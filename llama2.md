# LLaMA 模型实现架构拆解

## 📋 目录
1. [整体架构](#整体架构)
2. [数据结构](#数据结构)
3. [初始化流程](#初始化流程)
4. [前向传播流程](#前向传播流程)
5. [关键组件](#关键组件)
6. [代码文件组织](#代码文件组织)

---

## 整体架构

### 类层次结构

```
Model (基类)
  ↓ 继承
LLama2Model (LLaMA系列通用实现)
  ↓ 包含
LLama2Layers (所有层的容器)
```

### 核心职责

```cpp
class LLama2Model : public Model {
public:
  // 生命周期管理
  base::Status init(base::DeviceType device_type);

  // 推理接口
  base::Status forward(const tensor::Tensor& input,
                       const tensor::Tensor& pos_tensor,
                       int& next) const;
  base::Status predict(...) const;
  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const;

private:
  // 层管理
  std::unique_ptr<LLama2Layers> llama_layers_;
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
};
```

---

## 数据结构

### LLama2Layers：层的容器

```cpp
struct LLama2Layers {
  // ========== 无参数层（所有层共享） ==========
  std::shared_ptr<op::Layer> mha_layer_;      // 多头注意力
  std::shared_ptr<op::Layer> rope_layer_;     // 旋转位置编码
  std::shared_ptr<op::Layer> swiglu_layer_;   // SwiGLU激活
  std::shared_ptr<op::Layer> add_layer_;      // 向量加法（残差）

  // ========== 有参数层（每层独立） ==========
  // Attention权重
  std::vector<std::shared_ptr<op::Layer>> wq_layers_;  // Query投影 [layer_num]
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;  // Key投影
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;  // Value投影
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;  // Output投影

  // FFN权重
  std::vector<std::shared_ptr<op::Layer>> w1_layers_;  // 门控1
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;  // 输出
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;  // 门控2

  // 归一化层
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;  // [2*layer_num + 1]

  // ========== 其他 ==========
  std::shared_ptr<op::Layer> cls_layer_;       // 分类头（投影到词表）
  std::shared_ptr<op::Layer> embedding_layer_; // Embedding层
};
```

### 设计模式

**共享层 vs 独立层**：
```
无参数层（共享）:
  mha_layer_: 所有32层共用一个实例
    └─> 通过set_layer_idx(layer_idx)动态指定当前层

  rope_layer_: 所有32层共用一个实例

有参数层（独立）:
  wq_layers_[32]: 每层有独立的权重矩阵
    └─> wq_layers_[0] ~ wq_layers_[31]
```

---

## 初始化流程

### 完整流程图

```
init(device_type)
  │
  ├─> 1. 设备初始化
  │     ├─ CPU: 创建CPU分配器
  │     └─ CUDA: 创建CUDA配置、Stream
  │
  ├─> 2. 加载模型文件
  │     └─ gen_model_from_file()
  │        ├─ read_model_file()        // mmap权重文件
  │        ├─ create_encode_layer()    // 创建tokenizer
  │        └─ create_layers()          // 创建所有层
  │
  ├─> 3. 创建层的层级结构
  │     └─ create_layers()
  │        ├─ create_nonparam_layers()    // RoPE, MHA等
  │        ├─ create_param_layers()       // FP32权重层
  │        │  或 create_param_quant_layers()  // INT8量化层
  │        └─ init_mem()                  // 分配buffer
  │
  ├─> 4. 预计算位置编码缓存
  │     ├─ sin_cos_cache_calc_cpu()    // CPU版本
  │     └─ sin_cos_cache_calc_cu()     // CUDA版本
  │
  └─> 5. 创建采样器
     └─ ArgmaxSampler(device_type)
```

### 关键代码：init()

**位置**: `llama3.cpp:107-145`

```cpp
base::Status LLama2Model::init(base::DeviceType device_type) {
  // 1. 设备初始化
  device_type_ = device_type;
  if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
  }

  // 2. 加载模型
  Status read_status = gen_model_from_file();

  // 3. 分配buffer
  init_mem();

  // 4. 预计算RoPE缓存
  if (device_type == CPU) {
    kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
                                   sin_cache, cos_cache);
  } else {
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  sin_cache, cos_cache, cuda_config_->stream);
  }

  // 5. 创建采样器
  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return error::Success();
}
```

### 关键代码：create_layers()

**位置**: `llama3.cpp:502-598`

```cpp
base::Status LLama2Model::create_layers() {
  llama_layers_ = std::make_unique<LLama2Layers>();

  // 1. 创建无参数层（共享）
  create_nonparam_layers();

  // 2. 创建有参数层（每层独立）
  if (is_quant_model_) {
    create_param_quant_layers();  // INT8量化
  } else {
    create_param_layers();        // FP32
  }

  return error::Success();
}
```

### 关键代码：create_nonparam_layers()

**位置**: `llama3.cpp:169-182`

```cpp
void LLama2Model::create_nonparam_layers() {
  // RoPE层（旋转位置编码）
  llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

  // MHA层（多头注意力）
  llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0,                          // layer_index（初始值）
      config_->kv_mul_, config_->kv_dim_,       // GQA参数
      config_->seq_len_, config_->head_num_,    // 序列长度、头数
      config_->head_size_);                     // 头维度

  // 残差连接
  llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

  // SwiGLU激活
  llama_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
}
```

### 关键代码：create_param_layers()

**位置**: `llama3.cpp:290-424`

```cpp
void LLama2Model::create_param_layers() {
  size_t pos = 0;  // 权重文件偏移量
  int32_t dim = config_->dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // ========== 创建Attention权重 ==========

  // Query权重 [layer_num]
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wq->set_weight(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos += dim * dim;  // 移动偏移
  }

  // Key权重 [layer_num]
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wk->set_weight(0, {config_->kv_dim_, dim}, raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos += config_->kv_dim_ * dim;
  }

  // Value权重 [layer_num]
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wv->set_weight(0, {config_->kv_dim_, dim}, raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim;
  }

  // Output权重 [layer_num]
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wo->set_weight(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos += dim * dim;
  }

  // ========== 创建FFN权重 ==========

  // w1, w2, w3 [layer_num]
  // ... (类似逻辑)

  // ========== 创建其他层 ==========

  // Embedding层
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(...);

  // RMSNorm层 [2*layer_num + 1]
  // ... (创建所有RMSNorm层)

  // 分类头
  llama_layers_->cls_layer_ = std::make_shared<op::MatmulLayer>(...);
}
```

---

## 前向传播流程

### 单层Transformer Block

```
input [dim]
  │
  ├─> RMSNorm → rmsnorm [dim]
  │     │
  │     ├─> wq @ rmsnorm → query [dim]
  │     ├─> wk @ rmsnorm → key [kv_dim]
  │     ├─> wv @ rmsnorm → value [kv_dim]
  │     │
  │     └─> RoPE(query, key, pos) → q_rot, k_rot
  │            │
  │            └─> MHA(q_rot, K_cache, V_cache) → mha_out [dim]
  │                   │
  │                   └─> wo @ mha_out → attn_out [dim]
  │
  ├─> Residual: input + attn_out → input [dim]
  │
  ├─> RMSNorm → ffn_norm [dim]
  │     │
  │     ├─> w1 @ ffn_norm → w1_out [hidden_dim]
  │     ├─> w3 @ ffn_norm → w3_out [hidden_dim]
  │     │
  │     └─> SwiGLU(w1_out, w3_out) → swiglu_out [hidden_dim]
  │            │
  │            └─> w2 @ swiglu_out → ffn_out [dim]
  │
  └─> Residual: input + ffn_out → input [dim] (传递给下一层)
```

### 关键代码：forward()

**位置**: `llama3.cpp:147-167`

```cpp
base::Status LLama2Model::forward(const tensor::Tensor& input,
                                   const tensor::Tensor& pos_tensor,
                                   int& next) const {
  // 遍历所有Transformer层
  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    // 1. Attention RMSNorm
    attention_rms(layer_idx, input);

    // 2. QKV投影 + RoPE
    attention_qkv(layer_idx, pos_tensor);

    // 3. 多头注意力
    attention_mha(layer_idx, pos_tensor);

    // 4. 前馈网络
    feed_forward(layer_idx, input);
  }

  // 5. 最终分类
  cls_logits(input);

  return error::Success();
}
```

### 关键代码：attention_mha()

**位置**: `llama3.cpp:652-676`

```cpp
void LLama2Model::attention_mha(int32_t layer_idx,
                                 const tensor::Tensor& pos_tensor) const {
  // 1. 获取buffer
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
  tensor::Tensor query = get_buffer(ModelBufferType::kQuery);

  // 2. 设置MHA层参数
  int pos = pos_tensor.index<int32_t>(0);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer_)->set_pos(pos);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer_)->set_layer_idx(layer_idx);

  // 3. 执行MHA
  STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

  // 4. 输出投影
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = llama_layers_->wo_layers_.at(layer_idx);
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}
```

### 关键代码：feed_forward()

**位置**: `llama3.cpp:678-720`

```cpp
void LLama2Model::feed_forward(int32_t layer_idx,
                               const tensor::Tensor& input) const {
  // 1. 残差连接: input + attn_output
  CHECK_NE(llama_layers_->add_layer_, nullptr);
  STATUS_CHECK(
      llama_layers_->add_layer_->forward(input, get_buffer(kAttnOutput), input));

  // 2. FFN RMSNorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // 3. w1投影
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = llama_layers_->w1_layers_.at(layer_idx);
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // 4. w3投影
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = llama_layers_->w3_layers_.at(layer_idx);
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));

  // 5. SwiGLU激活
  CHECK_NE(llama_layers_->swiglu_layer_, nullptr);
  STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));

  // 6. w2投影
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = llama_layers_->w2_layers_.at(layer_idx);
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // 7. 残差连接: input + ffn_out
  CHECK_NE(llama_layers_->add_layer_, nullptr);
  STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_output, input));
}
```

---

## 关键组件

### 1. Buffer管理

**位置**: `llama3.cpp:425-500` (init_mem)

```cpp
void LLama2Model::init_mem() {
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();

  // KV Cache (最大的buffer)
  tensor::Tensor key_cache(base::DataType::kDataTypeFp32,
                           config_->layer_num_, config_->seq_len_, config_->kv_dim_,
                           true, alloc);
  insert_buffer(ModelBufferType::kKeyCache, key_cache);

  tensor::Tensor value_cache(...);
  insert_buffer(ModelBufferType::kValueCache, value_cache);

  // 中间计算buffer
  tensor::Tensor query(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  insert_buffer(ModelBufferType::kQuery, query);

  tensor::Tensor mha_output(...);
  insert_buffer(ModelBufferType::kOutputMHA, mha_output);

  tensor::Tensor score_storage(base::DataType::kDataTypeFp32,
                               config_->head_num_ * config_->seq_len_, true, alloc);
  insert_buffer(ModelBufferType::kScoreStorage, score_storage);

  // ... 其他buffer
}
```

### 2. RoPE位置编码

**调用位置**: `llama3.cpp:637` (attention_qkv)

```cpp
// 在attention_qkv中
rope_layer->forward(
    query, key, pos_tensor,
    get_buffer(kSinCache),
    get_buffer(kCosCache),
    tensor::Tensor{}  // value不需要RoPE
);
```

**RoPE缓存**: 在init时预计算
```cpp
// init()中
kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                               sin_cache, cos_cache, cuda_config_->stream);
```

### 3. KV Cache管理

**更新位置**: `llama3.cpp:617-632` (attention_qkv)

```cpp
void LLama2Model::attention_qkv(int32_t layer_idx,
                                 const tensor::Tensor& pos_tensor) const {
  int32_t pos = pos_tensor.index<int32_t>(0);

  // 获取KV Cache的当前时间片
  const auto& [key, val] = slice_kv_cache(layer_idx, pos);
  // key: Tensor [kv_dim]  (指向cache的第layer_idx层、第pos个位置)

  // 计算key/value
  key_layer->forward(rmsnorm_output, key);
  value_layer->forward(rmsnorm_output, val);

  // 应用RoPE
  rope_layer->forward(query, key, pos, sin_cache, cos_cache);
}
```

**slice_kv_cache**: `model.cpp:215-232`

```cpp
std::pair<Tensor, Tensor> Model::slice_kv_cache(int32_t layer_idx, int32_t token_pos) const {
  // KV Cache布局: [layer_num, seq_len, kv_dim]
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  // 返回视图（不复制数据）
  float* key_ptr = get_buffer(kKeyCache).ptr<float>() + cache_offset;
  float* val_ptr = get_buffer(kValueCache).ptr<float>() + cache_offset;

  Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr, key_ptr);
  Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr, val_ptr);

  return {key, val};
}
```

---

## 代码文件组织

### 头文件: `kuiper/include/model/llama3.h`

```
namespace model {
  struct LLama2Layers {
    // 所有层的容器
    std::shared_ptr<op::Layer> mha_layer_;
    std::vector<std::shared_ptr<op::Layer>> wq_layers_;
    // ...
  };

  class LLama2Model : public Model {
  public:
    // 公共接口
    base::Status init(base::DeviceType device_type) override;
    base::Status forward(...) override;
    base::Status predict(...) override;
    op::EmbeddingOutput embedding(...) override;

  private:
    // 初始化相关
    void init_mem() override;
    base::Status create_layers() override;
    void create_param_layers() override;
    void create_nonparam_layers() override;
    void create_param_quant_layers() override;

    // 推理相关
    void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;
    void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
    void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
    void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;
    void cls_logits(const tensor::Tensor& input) const;
    int32_t post_processing(...) const override;

    // 成员变量
    std::unique_ptr<LLama2Layers> llama_layers_;
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
  };
}
```

### 实现文件: `kuiper/source/model/llama3.cpp`

```
主要函数列表:

1. 生命周期管理
   - LLama2Model::LLama2Model()           [构造函数]
   - LLama2Model::~LLama2Model()          [析构函数]
   - LLama2Model::init()                  [初始化]

2. 推理接口
   - LLama2Model::forward()               [前向传播]
   - LLama2Model::predict()               [推理]
   - LLama2Model::embedding()             [Embedding]

3. 层创建
   - LLama2Model::create_layers()         [创建所有层]
   - LLama2Model::create_param_layers()   [创建FP32权重层]
   - LLama2Model::create_param_quant_layers()  [创建INT8权重层]
   - LLama2Model::create_nonparam_layers()[创建共享层]
   - LLama2Model::init_mem()              [分配buffer]

4. 推理组件
   - LLama2Model::attention_rms()         [Attention RMSNorm]
   - LLama2Model::attention_qkv()         [QKV投影+RoPE]
   - LLama2Model::attention_mha()         [多头注意力]
   - LLama2Model::feed_forward()          [前馈网络]
   - LLama2Model::cls_logits()            [分类]

5. 后处理
   - LLama2Model::post_processing()       [采样]
```

### 依赖关系

```
LLama2Model 依赖:
  ├─ op::MatmulLayer          (矩阵乘法)
  ├─ op::MultiHeadAttention   (多头注意力)
  ├─ op::RoPELayer            (旋转位置编码)
  ├─ op::RmsNormLayer         (归一化)
  ├─ op::SwiGLULayer          (激活函数)
  ├─ op::VecAddLayer          (向量加法)
  ├─ op::EmbeddingLayer       (Embedding)
  ├─ sampler::Sampler         (采样器)
  └─ tensor::Tensor           (张量)
```

---

## 设计亮点

### 1. 层的复用设计

**共享层通过参数复用**:
```cpp
// 创建单个MHA层
llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(...);

// 每次调用时动态设置层索引
for (int layer_idx = 0; layer_idx < 32; layer_idx++) {
  mha_layer_->set_layer_idx(layer_idx);
  mha_layer_->forward(...);  // 处理第layer_idx层
}
```

**好处**:
- 节省内存（32层共享1个MHA对象）
- 减少初始化开销

### 2. Buffer复用

**多个计算阶段共享buffer**:
```cpp
// 初始化时分配
tensor::Tensor temp(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);

insert_buffer(kOutputRMSNorm, temp);    // RMSNorm用
insert_buffer(kOutputMHA, temp);        // MHA用（复用）
insert_buffer(kW2Output, temp);         // FFN用（复用）
insert_buffer(kFFNRMSNorm, temp);       // RMSNorm用（复用）
```

**好处**:
- 减少内存占用
- 提高缓存命中率

### 3. 零拷贝视图

**KV Cache切片**:
```cpp
// 返回视图，不复制数据
std::pair<Tensor, Tensor> slice_kv_cache(int32_t layer_idx, int32_t token_pos) const {
  float* key_ptr = key_cache.ptr<float>() + offset;
  return {Tensor(view_of_data, key_ptr), ...};
}
```

**好处**:
- 避免数据复制
- 节省带宽

---

## 关键配置参数

### LLaMA-2 7B配置

```cpp
struct TransformerConfig {
  int32_t dim_ = 4096;           // 模型维度
  int32_t hidden_dim_ = 11008;   // FFN维度
  int32_t layer_num_ = 32;       // 层数
  int32_t head_num_ = 32;        // 注意力头数
  int32_t kv_head_num_ = 32;     // KV头数（GQA=1时与head_num相同）
  int32_t head_size_ = 128;      // 每个头维度 (4096/32)
  int32_t kv_dim_ = 4096;        // KV维度
  int32_t kv_mul_ = 1;           // GQA倍数（1=标准MHA）
  int32_t seq_len_ = 2048;       // 最大序列长度
  int32_t vocab_size_ = 32000;   // 词表大小
};
```

### 内存占用估算

```
KV Cache: 2 * 32 * 2048 * 4096 * 4 = 2GB
模型权重: ~13GB (FP32)
中间buffer: ~100MB

总内存: ~15GB (FP32)
```

---

## 总结

### 架构特点

1. **分层设计**: Model → Layers → Layer
2. **共享机制**: 无参数层在所有层间共享
3. **Buffer管理**: 预分配+复用
4. **零拷贝**: KV Cache视图、embedding视图

### 核心流程

```
初始化: init() → create_layers() → init_mem()
推理:   forward() → [for layer] → attention_mha() + feed_forward()
后处理: cls_logits() → post_processing() → next_token
```

### 关键文件

- `llama3.h`: 类定义（77行）
- `llama3.cpp`: 实现（746行）
- `model.h`: 基类定义
- `model.cpp`: 基类实现

---

**理解了LLama2Model，就理解了整个LLaMA推理框架的核心！**
