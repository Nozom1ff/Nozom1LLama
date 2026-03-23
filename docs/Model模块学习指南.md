# Model模块学习指南

## 目录
1. [模块概览](#模块概览)
2. [学习路径](#学习路径)
3. [各模块详解](#各模块详解)
4. [关键数据结构](#关键数据结构)
5. [推理流程](#推理流程)
6. [学习检查点](#学习检查点)

---

## 1. 模块概览

### 1.1 Model模块的职责

Model模块是整个推理框架的**指挥官**，负责：
- 加载模型权重文件
- 初始化Tokenizer
- 管理所有计算层（Layer）
- 协调推理过程中的数据流动
- 管理KV Cache等中间buffer

### 1.2 模块架构图

```
┌─────────────────────────────────────────────┐
│           Model (基类)                       │
│  - 模型加载                                   │
│  - 配置管理                                   │
│  - Buffer管理                                │
│  - 编码/解码                                  │
└──────────────┬──────────────────────────────┘
               │ 继承
               ├─────────────┬──────────────┬──────────────┐
               ▼             ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
        │LLama2Model│  │Qwen2Model│  │Qwen3Model│  │  其他...  │
        └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

### 1.3 文件结构

```
kuiper/include/model/
├── config.h              # 配置结构体（60行）
├── raw_model_data.h      # 模型文件加载（25行）
├── model.h              # Model基类（97行）
├── llama3.h             # LLaMA模型定义（77行）
├── qwen2.h              # Qwen2模型定义
└── qwen3.h              # Qwen3模型定义

kuiper/source/model/
├── model.cpp            # Model基类实现（260行）
├── raw_model_data.cpp   # 模型文件加载实现（22行）
├── llama3.cpp           # LLaMA模型实现（746行）★ 重点
├── qwen2.cpp            # Qwen2模型实现（751行）
└── qwen3.cpp            # Qwen3模型实现（644行）
```

---

## 2. 学习路径

### 📚 推荐学习顺序（从简单到复杂）

```
阶段1: 基础理解（1-2小时）
  ├─ 2.1 config.h          # 理解模型配置参数
  ├─ 2.2 raw_model_data.h  # 理解模型文件如何加载
  └─ 2.3 model.h (接口)    # 理解Model基类设计

阶段2: 核心流程（3-4小时）
  ├─ 2.4 model.cpp         # 模型初始化流程
  ├─ 2.5 demo/main.cpp     # 如何使用模型 ★ 先看这个！
  └─ 2.6 llama3.h         # LLaMA模型结构

阶段3: 深入实现（5-6小时）
  ├─ 2.7 llama3.cpp (初始化)  # init() → create_layers()
  ├─ 2.8 llama3.cpp (推理)    # forward() → predict()
  └─ 2.9 llama3.cpp (细节)    # attention_mha(), feed_forward()

阶段4: 扩展学习（可选）
  ├─ 2.10 qwen2.cpp        # 对比学习Qwen2
  └─ 2.11 qwen3.cpp        # 对比学习Qwen3
```

### 🎯 快速开始路径（如果时间有限）

**路径A: 理解整体流程（2小时）**
```
1. demo/main.cpp (10分钟)     # 如何使用
2. config.h (5分钟)            # 配置参数
3. llama3.h (15分钟)           # 数据结构
4. llama3.cpp关键函数:
   - init() (30分钟)           # 初始化
   - forward() (30分钟)        # 前向传播
   - attention_mha() (20分钟)  # MHA调用
   - feed_forward() (20分钟)   # FFN调用
```

**路径B: 理解数据流（3小时）**
```
1. raw_model_data.h/cpp (20分钟)  # 如何加载模型
2. model.cpp (40分钟)               # 基类实现
3. llama3.cpp (2小时)               # 完整推理流程
4. 调试追踪一遍推理过程 (20分钟)
```

---

## 3. 各模块详解

### 3.1 第一阶段：config.h（配置理解）

**学习目标**：理解LLaMA模型的核心配置参数

**关键概念**：

```cpp
// 文件格式配置（模型文件头部）
struct ModelConfig {
    int32_t dim;           // 模型隐藏维度，如4096
    int32_t hidden_dim;    // FFN隐藏维度，如11008
    int32_t layer_num;     // Transformer层数，如32
    int32_t head_num;      // 注意力头数，如32
    int32_t kv_head_num;   // KV头数（GQA），如4
    int32_t vocab_size;    // 词汇表大小，正数表示共享权重
    int32_t seq_len;       // 最大序列长度，如2048
};

// 运行时配置（计算出的派生参数）
struct TransformerConfig {
    int32_t dim_;          // = config.dim
    int32_t hidden_dim_;   // = config.hidden_dim
    int32_t layer_num_;    // = config.layer_num
    int32_t head_num_;     // = config.head_num
    int32_t kv_dim_;       // = dim * kv_head_num / head_num (计算!)
    int32_t kv_mul_;       // = head_num / kv_head_num (计算!)
    int32_t head_size_;    // = dim / head_num (计算!)
    int32_t seq_len_;      // = config.seq_len
    bool is_shared_weight_; // vocab_size > 0
};
```

**理解要点**：
- `kv_dim`是key/value的总维度，通常小于`dim`（GQA优化）
- `kv_mul`是query头与KV头的倍数，=1为标准MHA，>1为GQA
- `head_size`是每个注意力头的维度
- 这些参数决定了所有tensor的shape

**练习**：
- 给定：dim=4096, head_num=32, kv_head_num=4
- 计算：head_size=?, kv_dim=?, kv_mul=?

---

### 3.2 第二阶段：raw_model_data.h（模型加载）

**学习目标**：理解模型文件如何映射到内存

**核心机制**：

```
模型文件布局:
┌─────────────────┐
│ ModelConfig     │ ← 配置信息（固定大小）
├─────────────────┤
│ group_size      │ ← 量化参数（仅int8模型）
├─────────────────┤
│ Weights Data    │ ← 权重数据（巨大）
│ - token_embed   │
│ - layer_0_*     │
│ - layer_1_*     │
│ - ...           │
│ - rms_norm      │
└─────────────────┘

内存映射 (mmap):
整个文件映射到内存，无需全部加载到RAM
按需从文件读取（操作系统管理）
```

**关键数据结构**：

```cpp
struct RawModelData {
    int32_t fd;              // 文件描述符
    size_t file_size;        // 文件大小
    void* data;              // mmap映射的起始地址
    void* weight_data;       // 权重数据的起始地址

    virtual const void* weight(size_t offset) const = 0;
    // offset: 权重数据的偏移量
    // 返回: 对应权重的指针
};

// 两种实现
struct RawModelDataFp32 : RawModelData { ... };  // FP32模型
struct RawModelDataInt8 : RawModelData { ... };  // INT8量化模型
```

**理解要点**：
- 使用`mmap`而非`fread`，节省内存
- 权重数据不复制，只是指针偏移
- FP32和INT8模型的offset计算不同（INT8有额外的scale）

---

### 3.3 第三阶段：model.h（基类接口）

**学习目标**：理解Model提供的功能接口

**核心接口分类**：

```cpp
class Model {
public:
    // ====== 生命周期 ======
    virtual base::Status init(base::DeviceType device_type) = 0;
    // 功能：初始化模型，加载权重，创建层，分配内存

    // ====== 推理接口 ======
    virtual base::Status predict(..., int& next) const = 0;
    // 功能：执行一次推理，返回生成的token

    virtual base::Status forward(...) const = 0;
    // 功能：前向传播（内部调用）

    // ====== 编码/解码 ======
    virtual std::vector<int32_t> encode(const std::string& sentence) const;
    // 功能：文本→token ids

    virtual std::string decode(int32_t token_idx) const;
    // 功能：token id → 文本

    // ====== Buffer管理 ======
    virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);
    // 功能：获取中间计算buffer（如KV Cache）

    // ====== Embedding ======
    virtual op::EmbeddingOutput embedding(const std::vector<int>& tokens) const = 0;
    // 功能：token ids → 嵌入向量

protected:
    // ====== 虚函数（子类实现）=====

    // 创建层
    virtual base::Status create_layers() = 0;
    virtual void create_param_layers() = 0;      // 有权重的层
    virtual void create_nonparam_layers() = 0;   // 无权重的层
    virtual void create_param_quant_layers() = 0; // 量化层

    // 初始化内存
    virtual void init_mem() = 0;

    // 后处理
    virtual int32_t post_processing(...) const = 0;

    // ====== 已实现的功能 ======
    virtual base::Status read_model_file();
    // 功能：读取模型文件，mmap映射

    virtual base::Status gen_model_from_file();
    // 功能：从文件生成模型（调用create_layers）

    virtual base::Status create_encode_layer();
    // 功能：创建tokenizer层
};
```

**理解要点**：
- 模板方法模式：基类定义流程，子类实现细节
- `init()`调用一系列create函数，子类实现这些create函数
- `predict()`是对外接口，内部调用`forward()`

---

### 3.4 第四阶段：demo/main.cpp（使用示例）

**学习目标**：理解如何使用Model进行推理

**完整推理流程**：

```cpp
int main() {
    // 1. 创建模型
    model::LLama2Model model(
        base::TokenizerType::kEncodeSpe,  // tokenizer类型
        tokenizer_path,                    // tokenizer文件路径
        checkpoint_path,                   // 模型权重文件路径
        false                              // 是否量化模型
    );

    // 2. 初始化模型
    model.init(base::DeviceType::kDeviceCUDA);
    // 内部做了：
    //   - read_model_file()           // mmap模型文件
    //   - gen_model_from_file()       // 读取配置
    //   - create_encode_layer()       // 创建tokenizer
    //   - create_layers()             // 创建所有层
    //   - init_mem()                  // 分配buffer

    // 3. 编码输入文本
    auto tokens = model.encode("hello");
    // 结果: [15496, 2159] (假设)

    // 4. 获取prompt embedding
    const auto& prompt_embedding = model.embedding(tokens);
    // 结果: Tensor[2, 4096] (2个token，每个4096维)

    // 5. 自回归生成循环
    int pos = 0;
    int next = -1;
    bool is_prompt = true;

    while (pos < total_steps) {
        // 5.1 设置位置
        pos_tensor.index<int32_t>(0) = pos;

        // 5.2 准备输入
        if (pos < prompt_len - 1) {
            // Prompt阶段：使用预计算的embedding
            input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
        } else {
            // 生成阶段：只对上一个token计算embedding
            is_prompt = false;
            tokens = {next};
            token_embedding = model.embedding(tokens);
            input = model.fill_input(pos_tensor, token_embedding, is_prompt);
        }

        // 5.3 推理
        model.predict(input, pos_tensor, is_prompt, next);

        // 5.4 检查结束
        if (model.is_sentence_ending(next)) break;

        // 5.5 记录生成的token
        words.push_back(next);
        pos++;
    }

    // 6. 解码输出
    std::string output = model.decode(words);
    printf("%s\n", output.c_str());
}
```

**关键理解**：
- **Prompt阶段**：一次性处理所有prompt tokens，利用并行计算
- **生成阶段**：每次只处理一个token，利用KV Cache加速
- `pos_tensor`记录当前处理的位置，用于KV Cache索引

---

### 3.5 第五阶段：llama3.h（LLaMA模型结构）

**学习目标**：理解LLaMA模型的层组成

**核心数据结构**：

```cpp
// 一个LLaMA层的所有组件
struct LLama2Layers {
    // ====== Attention Block ======
    std::shared_ptr<op::Layer> mha_layer_;          // 多头注意力（无参数，共享）
    std::vector<std::shared_ptr<op::Layer>> wq_layers_;  // Query投影（每层独立）
    std::vector<std::shared_ptr<op::Layer>> wk_layers_;  // Key投影
    std::vector<std::shared_ptr<op::Layer>> wv_layers_;  // Value投影
    std::vector<std::shared_ptr<op::Layer>> wo_layers_;  // Output投影
    std::shared_ptr<op::Layer> rope_layer_;          // RoPE位置编码（共享）

    // ====== Feed-Forward Block ======
    std::vector<std::shared_ptr<op::Layer>> w1_layers_;  // FFN门控1
    std::vector<std::shared_ptr<op::Layer>> w2_layers_;  // FFN输出
    std::vector<std::shared_ptr<op::Layer>> w3_layers_;  // FFN门控2
    std::shared_ptr<op::Layer> swiglu_layer_;        // SwiGLU激活（共享）

    // ====== Normalization ======
    std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;  // 所有RMSNorm层
                                                                 // 顺序：[attn_0, attn_1, ..., ffn_0, ffn_1, ..., final]

    // ====== Other ======
    std::shared_ptr<op::Layer> add_layer_;           // 残差连接（共享）
    std::shared_ptr<op::Layer> cls_layer_;           // 分类头（投影到vocab）
    std::shared_ptr<op::Layer> embedding_layer_;     // Embedding层

    void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};
```

**类结构**：

```cpp
class LLama2Model : public Model {
public:
    // ====== 实现基类接口 ======
    base::Status init(base::DeviceType device_type) override;
    base::Status predict(...) const override;
    base::Status forward(...) const override;
    op::EmbeddingOutput embedding(...) const override;

private:
    // ====== 初始化相关 ======
    void init_mem() override;                    // 分配所有buffer
    base::Status create_layers() override;       // 创建所有层
    void create_param_layers() override;         // 创建有权重的层
    void create_nonparam_layers() override;      // 创建无权重的层
    void create_param_quant_layers() override;   // 创建量化层

    // ====== 推理相关 ======
    void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;
    void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
    void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
    void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;
    void cls_logits(const tensor::Tensor& input) const;

    int32_t post_processing(...) const override; // 采样处理
    std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(...) const;

private:
    std::unique_ptr<LLama2Layers> llama_layers_;  // 所有层
    std::shared_ptr<kernel::CudaConfig> cuda_config_;  // CUDA配置
};
```

**设计亮点**：
- **共享层**：mha、rope、swiglu、add层在所有layer间共享（通过参数区分）
- **权重层**：wq、wk、wv、wo等每层独立（不同权重）
- **Buffer管理**：通过`get_buffer(ModelBufferType)`访问中间结果

---

### 3.6 第六阶段：llama3.cpp（实现细节）

**学习目标**：理解LLaMA模型的完整实现

#### 3.6.1 初始化流程（init）

```
init(device_type)
  │
  ├─> [基类] read_model_file()
  │     ├─ 打开模型文件
  │     ├─ 读取ModelConfig
  │     ├─ mmap映射到内存
  │     └─ 设置weight_data指针
  │
  ├─> [基类] gen_model_from_file()
  │     ├─ generate_model_infos()     # 生成TransformerConfig
  │     ├─ create_encode_layer()      # 创建tokenizer
  │     └─ create_layers()            # 子类实现
  │
  ├─> create_layers()
  │     ├─ create_nonparam_layers()   # 创建共享层（mha, rope等）
  │     ├─ create_param_layers()      # 创建权重层（wq, wk等）
  │     │  或 create_param_quant_layers()  # 量化模型
  │     └─ init_mem()                 # 分配所有buffer
  │
  └─> 创建sampler
```

**关键点**：
- 所有层的创建都在初始化阶段完成
- 权重数据直接指向mmap的内存，不复制
- buffer按最大需求分配（如seq_len）

#### 3.6.2 Buffer管理（init_mem）

**定义的Buffer**（从ModelBufferType枚举）：

```cpp
enum class ModelBufferType {
    kInputTokens = 0,         // 输入token ids
    kInputEmbeddings = 1,     // 输入embedding
    kOutputRMSNorm = 2,       // RMSNorm输出
    kKeyCache = 3,            // KV Cache (Key)
    kValueCache = 4,          // KV Cache (Value)
    kQuery = 5,               // Query向量
    kInputPos = 6,            // 位置索引
    kScoreStorage = 7,        # Attention分数
    kOutputMHA = 8,           // MHA输出
    kAttnOutput = 9,          # Attention输出（wo之后）
    kW1Output = 10,           # FFN w1输出
    kW2Output = 11,           # FFN w2输出
    kW3Output = 12,           # FFN w3输出
    kFFNRMSNorm = 13,         # FFN RMSNorm输出
    kForwardOutput = 15,      # 最终输出
};
```

**Buffer分配**：

```cpp
void LLama2Model::init_mem() {
    // KV Cache: [layer_num, seq_len, kv_dim]
    insert_buffer(kKeyCache,
        Tensor({layer_num, seq_len, kv_dim}, device_type));
    insert_buffer(kValueCache,
        Tensor({layer_num, seq_len, kv_dim}, device_type));

    // Query: [dim]
    insert_buffer(kQuery,
        Tensor({dim}, device_type));

    // MHA输出: [dim]
    insert_buffer(kOutputMHA,
        Tensor({dim}, device_type));

    // Attention分数: [head_num * seq_len]
    insert_buffer(kScoreStorage,
        Tensor({head_num * seq_len}, device_type));

    // ... 其他buffer
}
```

**理解要点**：
- KV Cache是最大的buffer（layer_num × seq_len × kv_dim × 4字节）
- 其他buffer大多是单次计算用的临时空间
- 使用`insert_buffer`注册到map中，通过`get_buffer`访问

#### 3.6.3 前向传播流程（forward）

```
forward(input, pos_tensor, next)
  │
  ├─> for layer_idx in [0, layer_num):
  │     ├─> attention_rms(layer_idx, input)
  │     │    └─ RMSNorm(input) → rmsnorm_output
  │     │
  │     ├─> attention_qkv(layer_idx, pos_tensor)
  │     │    ├─ wq @ rmsnorm_output → query
  │     │    ├─ wk @ rmsnorm_output → key
  │     │    ├─ wv @ rmsnorm_output → value
  │     │    └─ RoPE(query, key, pos)
  │     │
  │     ├─> attention_mha(layer_idx, pos_tensor)  ★ 核心调用
  │     │    ├─ mha_layer->set_pos(pos)
  │     │    ├─ mha_layer->set_layer_idx(layer_idx)
  │     │    ├─ mha_layer->forward(...)
  │     │    │   └─ mha_kernel_cu(...)
  │     │    └─ wo @ mha_output → attn_output
  │     │
  │     └─> feed_forward(layer_idx, input)
  │          ├─ residual: input + attn_output
  │          ├─ RMSNorm(input) → ffn_norm
  │          ├─ w1 @ ffn_norm → w1_out
  │          ├─ w3 @ ffn_norm → w3_out
  │          ├─ SwiGLU(w1_out, w3_out) → swiglu_out
  │          ├─ w2 @ swiglu_out → w2_out
  │          └─ residual: input + w2_out
  │
  ├─> cls_logits(input)
  │     ├─ RMSNorm(input)
  │     └─ cls_layer @ input → logits
  │
  └─> post_processing(pos, is_prompt)
     └─ sampler->sample(logits) → next_token
```

**数据流转**（单层）：

```
input [dim]
  │
  ├─> RMSNorm → rmsnorm [dim]
  │     │
  │     ├─> wq → query [dim]
  │     ├─> wk → key [kv_dim]
  │     ├─> wv → value [kv_dim]
  │     │
  │     └─> RoPE → query_rot, key_rot
  │            │
  │            └─> MHA(query, K_cache, V_cache) → mha_out [dim]
  │                   │
  │                   └─> wo → attn_out [dim]
  │
  ├─> Residual: input + attn_out → input [dim]
  │
  ├─> RMSNorm → ffn_norm [dim]
  │     │
  │     ├─> w1 → w1_out [hidden_dim]
  │     ├─> w3 → w3_out [hidden_dim]
  │     │
  │     └─> SwiGLU(w1_out, w3_out) → swiglu_out [hidden_dim]
  │            │
  │            └─> w2 → w2_out [dim]
  │
  └─> Residual: input + w2_out → input [dim] (传递给下一层)
```

#### 3.6.4 关键函数解析

**attention_qkv**：

```cpp
void LLama2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
    // 1. 获取当前位置
    int32_t pos = pos_tensor.index<int32_t>(0);

    // 2. 获取KV Cache的当前时间片
    const auto& [key, val] = slice_kv_cache(layer_idx, pos);
    // key: Tensor [kv_dim]  (指向cache的第layer_idx层、第pos个位置)
    // val: Tensor [kv_dim]

    // 3. Query投影
    auto rmsnorm_output = get_buffer(kOutputRMSNorm);
    query_layer->forward(rmsnorm_output, query);  // [dim] → [dim]

    // 4. Key投影
    key_layer->forward(rmsnorm_output, key);      // [dim] → [kv_dim]

    // 5. Value投影
    value_layer->forward(rmsnorm_output, val);    // [dim] → [kv_dim]

    // 6. RoPE位置编码
    rope_layer->forward(query, key, pos, sin_cache, cos_cache);
}
```

**slice_kv_cache**：

```cpp
std::pair<Tensor, Tensor> LLama2Model::slice_kv_cache(int32_t layer_idx, int32_t token_pos) const {
    // KV Cache布局: [layer_num, seq_len, kv_dim]
    // 需要获取: 第layer_idx层、第token_pos位置的key/value

    tensor::Tensor key_cache = get_buffer(kKeyCache);
    tensor::Tensor val_cache = get_buffer(kValueCache);

    // 计算偏移量
    size_t key_offset = (layer_idx * seq_len_ + token_pos) * kv_dim_;
    size_t val_offset = (layer_idx * seq_len_ + token_pos) * kv_dim_;

    // 返回视图（不复制数据）
    Tensor key = key_cache.slice(key_offset, kv_dim_);
    Tensor val = val_cache.slice(val_offset, kv_dim_);

    return {key, val};
}
```

---

## 4. 关键数据结构

### 4.1 模型配置对比

| 参数 | LLaMA-7B | LLaMA-13B | LLaMA-70B | 说明 |
|------|----------|-----------|-----------|------|
| dim | 4096 | 5120 | 8192 | 模型隐藏维度 |
| hidden_dim | 11008 | 13824 | 28672 | FFN维度 |
| layer_num | 32 | 40 | 80 | Transformer层数 |
| head_num | 32 | 40 | 64 | 注意力头数 |
| kv_head_num | 32 | 40 | 8 | KV头数（GQA） |
| head_size | 128 | 128 | 128 | 每个头的维度 |
| kv_dim | 4096 | 5120 | 1024 | KV总维度 |
| kv_mul | 1 | 1 | 8 | GQA倍数 |
| vocab_size | 32000 | 32000 | 128000 | 词汇表大小 |
| seq_len | 2048 | 2048 | 2048 | 最大序列长度 |

### 4.2 KV Cache内存计算

```
KV Cache大小 = 2 × layer_num × seq_len × kv_dim × sizeof(float)

LLaMA-7B:
  = 2 × 32 × 2048 × 4096 × 4
  = 2,147,483,648 字节
  = 2 GB

LLaMA-70B (GQA优化):
  = 2 × 80 × 2048 × 1024 × 4
  = 1,342,177,280 字节
  = 1.25 GB
```

### 4.3 权重文件布局

```
文件偏移    内容                      大小
─────────────────────────────────────────────
0x0000     ModelConfig              sizeof(ModelConfig)
0x0040     group_size (仅int8)      sizeof(int32_t)
0x0044     token_embed              vocab_size × dim × 4
0x0044+    rms_norm_0               dim × 4
...        layer_0_wq               dim × dim × 4
...        layer_0_wk               kv_dim × dim × 4
...        layer_0_wv               kv_dim × dim × 4
...        layer_0_wo               dim × dim × 4
...        rms_norm_ffn_0           dim × 4
...        layer_0_w1               hidden_dim × dim × 4
...        layer_0_w2               dim × hidden_dim × 4
...        layer_0_w3               hidden_dim × dim × 4
...        (重复 layer_1 到 layer_N-1)
...        rms_norm_final           dim × 4
─────────────────────────────────────────────
```

---

## 5. 推理流程

### 5.1 完整推理时序图

```
用户输入: "hello"

编码阶段:
  "hello" → encode() → [15496, 2159]

Prompt处理 (并行):
  Token 0 (15496)                    Token 1 (2159)
       │                                  │
       ├─> embedding → embed[0]           ├─> embedding → embed[1]
       │                                  │
       └─> for layer in 0..31:           └─> for layer in 0..31:
            ├─> attention                      ├─> attention
            ├─> ffn                           ├─> ffn
            └─> output                       └─> output
                                            │
                                            ↓
                                    logits → sample() → token_x

自回归生成 (串行):
  Token 2 (token_x)
       │
       ├─> embedding → embed[2]
       │
       └─> for layer in 0..31:
            ├─> attention (使用KV Cache: embed[0], embed[1], embed[2])
            ├─> ffn
            └─> output
                                            │
                                            ↓
                                    logits → sample() → token_y

  循环直到<eos>或达到max_length
```

### 5.2 Buffer生命周期

```
初始化阶段:
  └─> init_mem() → 分配所有buffer (最大容量)

Prompt阶段:
  for pos in 0..prompt_len-1:
      ├─> fill_input() → kInputEmbeddings
      ├─> attention_qkv() → 更新 kKeyCache[pos], kValueCache[pos]
      ├─> attention_mha() → 读取 kKeyCache[0..pos]
      └─> forward() → 各个buffer被读写

生成阶段:
  while not eos:
      ├─> embedding(next) → kInputEmbeddings
      ├─> attention_qkv() → 更新 kKeyCache[pos], kValueCache[pos]
      ├─> attention_mha() → 读取 kKeyCache[0..pos]
      └─> forward() → 各个buffer被读写
```

---

## 6. 学习检查点

### ✅ 阶段1检查点

**应该能够回答**：
1. ModelConfig和TransformerConfig的区别是什么？
2. kv_dim和kv_mul是如何计算的？
3. 为什么使用mmap而不是fread加载模型？

**练习**：
- 给定一个模型的配置参数，计算所有派生参数
- 画出模型文件的内存布局图

---

### ✅ 阶段2检查点

**应该能够回答**：
1. Model基类提供了哪些功能接口？
2. `init()`函数的执行流程是什么？
3. `predict()`和`forward()`的关系是什么？

**练习**：
- 阅读demo/main.cpp，找到每个函数调用的位置
- 修改demo，打印出每次推理的时间

---

### ✅ 阶段3检查点

**应该能够回答**：
1. LLama2Layers包含哪些层？哪些是共享的？
2. KV Cache的索引是如何计算的？
3. attention_mha()中如何传递layer_idx？

**练习**：
- 添加一个自定义的buffer
- 打印每层的输出值

---

### ✅ 阶段4检查点（深入理解）

**应该能够回答**：
1. GQA是如何通过kv_mul实现的？
2. slice_kv_cache为什么返回的是视图而非副本？
3. 残差连接在哪里实现？

**练习**：
- 实现一个简单的CPU版本model（只支持1层）
- 对比CPU和CUDA版本的推理速度

---

### ✅ 综合项目

**目标**：添加一个新模型支持（如TinyLLaMA）

**步骤**：
1. 导出模型权重（使用export.py）
2. 创建TinyLlamaModel类（继承Model）
3. 实现create_layers()和forward()
4. 在demo中测试

---

## 7. 调试技巧

### 7.1 打印中间结果

```cpp
// 在forward()中添加
void LLama2Model::forward(...) const {
    for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
        attention_rms(layer_idx, input);

        // 打印RMSNorm输出
        auto rmsnorm_output = get_buffer(kOutputRMSNorm);
        if (layer_idx == 0) {
            float sum = 0;
            for (int i = 0; i < config_->dim_; i++) {
                sum += rmsnorm_output.index<float>(i);
            }
            LOG(INFO) << "Layer 0 RMSNorm output sum: " << sum;
        }

        attention_qkv(layer_idx, pos_tensor);
        // ...
    }
}
```

### 7.2 验证KV Cache

```cpp
// 在attention_qkv()后添加
void LLama2Model::attention_qkv(...) const {
    // ... 原有代码

    // 验证key cache是否正确写入
    tensor::Tensor key_cache = get_buffer(kKeyCache);
    float* key_ptr = key_cache.ptr<float>() + (layer_idx * seq_len_ + pos) * kv_dim_;
    LOG(INFO) << "Layer " << layer_idx << " pos " << pos
              << " key[0]: " << key_ptr[0];
}
```

### 7.3 性能分析

```cpp
#include <base/tick.h>

void LLama2Model::forward(...) const {
    base::Tick tick;
    tick.start();

    for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
        attention_rms(layer_idx, input);
        attention_qkv(layer_idx, pos_tensor);

        auto t1 = tick.elapse();
        attention_mha(layer_idx, pos_tensor);
        auto t2 = tick.elapse();

        feed_forward(layer_idx, input);

        if (layer_idx % 8 == 0) {
            LOG(INFO) << "Layer " << layer_idx
                      << " MHA: " << (t2 - t1) << "ms";
        }
    }
}
```

---

## 8. 常见问题

### Q1: 为什么MHA层是共享的？

**A**: MHA计算逻辑完全相同，只是：
- 输入数据不同（不同层的query/key/value）
- 访问的KV Cache区域不同（通过layer_offset区分）

通过`set_layer_idx(layer_idx)`动态设置，避免创建32个相同的MHA对象。

### Q2: KV Cache为什么是三维的？

**A**: `[layer_num, seq_len, kv_dim]`
- layer_num: 每层的KV独立
- seq_len: 历史位置（0到当前pos）
- kv_dim: 每个位置的key/value维度

### Q3: prompt阶段和生成阶段的区别？

**A**:
- **Prompt阶段**：一次性处理所有prompt tokens，利用GPU并行
- **生成阶段**：每次处理一个token，依赖KV Cache加速

### Q4: 如何支持新模型（如Qwen）？

**A**:
1. 继承Model基类
2. 实现create_layers()（创建模型特定的层）
3. 实现forward()（定义前向传播逻辑）
4. 如果有特殊算子，添加到op/模块

---

## 9. 推荐资源

### 论文
- *Attention Is All You Need* (Transformer)
- *LLaMA: Open and Efficient Foundation Language Models*
- *GQA: Training Generalized Multi-Query Transformer Models*

### 代码参考
- [nanoGPT](https://github.com/karpathy/nanoGPT) - 简洁的Transformer实现
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU推理优化

### 工具
- ncu (NVIDIA Compute Application) - CUDA性能分析
- nsight systems - 系统级性能追踪

---

**总结**：

Model模块的学习路径：
1. **配置** → 理解模型参数
2. **加载** → 理解文件映射
3. **接口** → 理解基类设计
4. **使用** → 理解推理流程
5. **实现** → 理解具体细节

建议按照顺序学习，每个阶段都要动手实验，打印中间结果，加深理解。

**下一步**：选择一个具体的模型（如LLaMA），完整追踪一次推理过程。
