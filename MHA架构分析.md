# KuiperLLama MHA（多头注意力）架构分析

## 目录
1. [MHA调用逻辑](#mha调用逻辑)
2. [Layer的含义](#layer的含义)
3. [MHA如何融入LLaMA](#mha如何融入llama)
4. [数据流转图](#数据流转图)
5. [关键代码位置](#关键代码位置)

---

## 1. MHA调用逻辑

### 1.1 整体调用流程

```
LLama2Model::forward()
  └─> for each layer_idx in [0, layer_num):
       ├─> attention_rms()           # RMS归一化
       ├─> attention_qkv()           # 计算Q,K,V并应用RoPE
       ├─> attention_mha()           # **核心：多头注意力计算**
       └─> feed_forward()            # 前馈神经网络
```

### 1.2 MHA详细调用步骤

#### Step 1: 初始化阶段（create_nonparam_layers）
**位置**: `kuiper/source/model/llama3.cpp:169-182`

```cpp
llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
    device_type_,
    0,                              // layer_index（初始值，后续会更新）
    config_->kv_mul_,               // GQA参数：查询头与KV头的倍数
    config_->kv_dim_,               // KV维度
    config_->seq_len_,              // 序列长度
    config_->head_num_,             // 注意力头数量
    config_->head_size_             // 每个头的维度
);
```

**参数说明**:
- `kv_mul_`: 用于GQA（Grouped Query Attention），当值为1时为标准MHA，大于1时为GQA
- `kv_dim_`: key和value的总维度 = head_size * (head_num / kv_mul)
- `head_num_`: 查询头的数量
- `head_size_`: 每个注意力头的维度

#### Step 2: 前向传播中的MHA调用
**位置**: `kuiper/source/model/llama3.cpp:652-676`

```cpp
void LLama2Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
    // 1. 获取必要的张量
    tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
    tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
    tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    tensor::Tensor query = get_buffer(ModelBufferType::kQuery);

    // 2. 设置动态参数
    int pos = pos_tensor.index<int32_t>(0);  // 当前token位置
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);

    // 3. 执行MHA前向传播
    STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

    // 4. 输出投影（wo矩阵）
    STATUS_CHECK(wo_layers_.at(layer_idx)->forward(mha_output, attn_output));
}
```

#### Step 3: MHA Layer内部处理
**位置**: `kuiper/source/op/mha.cpp:19-38`

```cpp
base::Status MultiHeadAttention::forward() {
    // 获取输入输出
    const tensor::Tensor& query_tensor = this->get_input(0);   // [head_num * head_size]
    const tensor::Tensor& score_tensor = this->get_input(1);   // [head_num * seq_len]
    const tensor::Tensor& key_cache_tensor = this->get_input(2); // KV Cache
    const tensor::Tensor& value_cache_tensor = this->get_input(3);
    const tensor::Tensor& mha_out = this->get_output(0);

    // 调用CUDA kernel
    kernel::get_mha_kernel(device_type_)(
        pos_,           // 当前位置
        head_num_,      // 注意力头数量
        layer_index_,   // 层索引
        seq_len_,       // 序列长度
        kv_dim_,        // KV维度
        kv_mul_,        // GQA倍数
        head_size_,     // 头维度
        mha_out, query_tensor, score_tensor,
        key_cache_tensor, value_cache_tensor,
        device_type_, cuda_config_
    );
}
```

---

## 2. Layer的含义

### 2.1 Transformer Layer的组成

在LLaMA模型中，一个**Layer**（Transformer层）是指一个完整的Transformer Block，包含：

```
Layer N (第N层)
├── Attention Block
│   ├── RMSNorm                    # 归一化
│   ├── QKV Projection             # Q,K,V投影（wq, wk, wv矩阵）
│   ├── RoPE (Rotary Positional Embedding)  # 位置编码
│   ├── **Multi-Head Attention**   # 多头注意力（本文重点）
│   └── Output Projection (wo)     # 输出投影
│
└── Feed-Forward Block
    ├── RMSNorm
    ├── SwiGLU Activation (w1, w3)
    └── Output Projection (w2)
    └── Residual Connection
```

### 2.2 Layer的层次关系

```
LLama2Model (模型)
└── config_->layer_num_ (例如：32层)
    └── Layer 0
        ├── wq_layers_[0], wk_layers_[0], wv_layers_[0]
        ├── mha_layer_ (共享，通过layer_idx参数区分)
        └── wo_layers_[0]
    └── Layer 1
        ├── wq_layers_[1], wk_layers_[1], wv_layers_[1]
        ├── mha_layer_ (同一实例，set_layer_idx(1))
        └── wo_layers_[1]
    ...
    └── Layer 31
        └── ...
```

**关键点**：
- 模型有`layer_num`个层（例如LLaMA-7B有32层）
- **MHA层是所有层共享的**，通过`set_layer_idx(layer_idx)`动态指定当前处理哪一层
- 每层有独立的权重矩阵（wq, wk, wv, wo），但MHA计算逻辑复用

### 2.3 Layer的配置参数

**位置**: `kuiper/include/model/config.h`

```cpp
struct TransformerConfig {
    int32_t dim_;           // 模型隐藏维度（例如：4096）
    int32_t hidden_dim_;    // FFN隐藏维度（例如：11008）
    int32_t layer_num_;     // 层数（例如：32）
    int32_t head_num_;      // 注意力头数（例如：32）
    int32_t kv_head_num_;   // KV头数（例如：4，用于GQA）
    int32_t head_size_;     // 每个头的维度（dim / head_num = 128）
    int32_t kv_dim_;        // KV维度（head_size * kv_head_num）
    int32_t kv_mul_;        // GQA倍数（head_num / kv_head_num = 8）
    int32_t seq_len_;       // 最大序列长度（例如：2048）
};
```

---

## 3. MHA如何融入LLaMA

### 3.1 完整的Attention Block数据流

```
输入: x [dim] (上一层的输出)

1. RMS归一化
   x_norm = RMSNorm(x)

2. QKV投影
   Q = x_norm @ wq.T  [dim] -> [dim]        (查询)
   K = x_norm @ wk.T  [dim] -> [kv_dim]     (键)
   V = x_norm @ wv.T  [dim] -> [kv_dim]     (值)

3. RoPE位置编码
   Q_rot = RoPE(Q, pos)
   K_rot = RoPE(K, pos)

4. KV Cache更新
   key_cache[layer_idx, pos] = K_rot
   value_cache[layer_idx, pos] = V

5. **Multi-Head Attention (MHA)**
   // CUDA kernel: multi_head_attention_kernel
   for each head in [0, head_num):
       for t in [0, pos]:  // 遍历所有历史位置
           score[head, t] = (Q[head] @ K_cache[t, head]) / sqrt(head_size)
       score[head] = Softmax(score[head])
       output[head] = score[head] @ V_cache[t, head]

6. 输出投影
   attn_out = MHA_output @ wo.T  [dim] -> [dim]

7. 残差连接
   residual_out = x + attn_out
```

### 3.2 CUDA Kernel实现细节

**位置**: `kuiper/source/op/kernels/cuda/mha_kernel.cu:49-106`

#### Kernel启动配置
```cpp
multi_head_attention_kernel<<<
    head_num,      // Grid维度：每个block处理一个注意力头
    thread_num,    // Block维度：每个block 256个线程
    head_size * sizeof(float),  // Shared memory：存储一个query头
    stream
>>>(
    pos, seq_len, query, score_ptr, output,
    key_cache, value_cache, kv_dim, kv_mul,
    head_num, head_size, layer_offset
);
```

#### Kernel核心逻辑
```cpp
__global__ void multi_head_attention_kernel(...) {
    int head = blockIdx.x;  // 每个block处理一个头

    // 1. 预加载query到共享内存
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        s_query_head[i] = query_head[i];
    }
    __syncthreads();

    // 2. 计算注意力分数（与所有历史位置）
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        float score = 0;
        // 使用float4向量化加载，每个循环处理4个float
        for (int i = 0; i < head_size; i += 4) {
            float4 key_val = *reinterpret_cast<float4*>(key_head + i);
            float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);
            score += key_val.x * query_val.x + ...;  // 点积
        }
        score_head[t] = score * scale;  // scale = 1/sqrt(head_size)
    }

    // 3. Softmax归一化
    softmax_gpu(score_head, pos + 1);

    // 4. 加权求和价值
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float value = 0;
        for (int t = 0; t <= pos; t++) {
            value += score_head[t] * value_cache[t][i];
        }
        output_head[i] = value;
    }
}
```

#### 关键优化技术

1. **共享内存优化**
   - 将query加载到共享内存，减少全局内存访问
   - `extern __shared__ float s_query_head[]`

2. **向量化内存访问**
   - 使用`float4`一次加载128位（4个float32）
   - 提高内存带宽利用率

3. **GQA支持**
   - `kv_mul_`参数实现Grouped Query Attention
   - 多个query头共享一组KV头
   - `head_offset = (head / kv_mul) * head_size`

4. **KV Cache分层**
   - `layer_offset = layer_index * seq_len * kv_dim`
   - 每层的KV Cache独立存储，避免层间干扰

### 3.3 GQA（Grouped Query Attention）机制

当`kv_mul_ > 1`时，使用GQA优化：

```
标准MHA (kv_mul=1):
  Query Heads:  [h0, h1, h2, ..., h31]
  Key Heads:    [h0, h1, h2, ..., h31]    32个KV头
  Value Heads:  [h0, h1, h2, ..., h31]

GQA (kv_mul=8):
  Query Heads:  [h0, h1, h2, ..., h31]     32个Query头
  Key Heads:    [H0, H0, H0, H0, H1, H1, H1, H1, ...]  4个KV头
  Value Heads:  [V0, V0, V0, V0, V1, V1, V1, V1, ...]  4个Value头

映射关系:
  Query h0-h3    -> KV H0 (每4个Query头共享1个KV头)
  Query h4-h7    -> KV H1
  Query h8-h11   -> KV H2
  ...
```

**代码实现** (`mha_kernel.cu:73`):
```cpp
int head_offset = (head / kv_mul) * head_size;
float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;
```

---

## 4. 数据流转图

### 4.1 单次推理的完整数据流

```
Token Embedding
    ↓
┌───────────────────────────────────────────────┐
│  Layer 0                                       │
│  ├─ RMSNorm(x) → x_norm                       │
│  ├─ wq@x_norm → Q, wk@x_norm → K, wv@x_norm → V │
│  ├─ RoPE(Q,K)                                  │
│  ├─ MHA(Q,K_cache,V_cache) → attn             │
│  ├─ wo@attn → attn_out                        │
│  ├─ x + attn_out (残差)                       │
│  ├─ RMSNorm → ffn_in                          │
│  ├─ w1@ffn_in, w3@ffn_in → SwiGLU            │
│  ├─ w2@SwiGLU_out → ffn_out                   │
│  └─ x + ffn_out (残差) → Layer 0 输出          │
└───────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────┐
│  Layer 1                                       │
│  └─ (相同结构...)                               │
└───────────────────────────────────────────────┘
    ↓
  ...
    ↓
┌───────────────────────────────────────────────┐
│  Layer 31                                      │
│  └─ (相同结构...)                               │
└───────────────────────────────────────────────┘
    ↓
Final RMSNorm
    ↓
Logits (vocab_size)
```

### 4.2 MHA内部数据流（单头）

```
输入: query [head_size]
      key_cache [seq_len, kv_dim]
      value_cache [seq_len, kv_dim]

for t = 0 to pos:
    ┌──────────────────────────────────────┐
    │  Attention Score Calculation         │
    │  score[t] = query @ key_cache[t]     │
    │            / sqrt(head_size)         │
    └──────────────────────────────────────┘
              ↓
    ┌──────────────────────────────────────┐
    │  Softmax Normalization               │
    │  prob[t] = exp(score[t]) / sum(exp)  │
    └──────────────────────────────────────┘
              ↓
    ┌──────────────────────────────────────┐
    │  Weighted Value Aggregation          │
    │  output += prob[t] * value_cache[t]  │
    └──────────────────────────────────────┘

输出: output [head_size]
```

---

## 5. 关键代码位置

### 5.1 文件结构

```
kuiper/
├── include/
│   ├── op/
│   │   ├── layer.h                    # Layer基类定义
│   │   └── mha.h                      # MHA Layer接口
│   └── model/
│       ├── llama3.h                   # LLaMA模型定义
│       └── config.h                   # 模型配置
│
└── source/
    ├── op/
    │   ├── layer.cpp                  # Layer基类实现
    │   ├── mha.cpp                    # MHA Layer实现
    │   └── kernels/
    │       ├── cuda/
    │       │   ├── mha_kernel.cu      # CUDA MHA kernel（本文重点）
    │       │   └── mha_kernel.cuh     # CUDA kernel接口
    │       └── cpu/
    │           ├── mha_kernel.cpp     # CPU MHA kernel
    │           └── mha_kernel.h
    │
    └── model/
        └── llama3.cpp                 # LLaMA模型实现
            ├── init()                 # 模型初始化
            ├── forward()              # 前向传播入口
            ├── create_nonparam_layers()  # 创建MHA层
            ├── attention_qkv()        # QKV计算
            └── attention_mha()        # MHA调用（关键）
```

### 5.2 关键函数速查表

| 函数 | 文件位置 | 行号 | 功能 |
|------|---------|------|------|
| `LLama2Model::forward()` | `llama3.cpp` | 147 | 模型前向传播入口 |
| `LLama2Model::create_nonparam_layers()` | `llama3.cpp` | 169 | 创建MHA层 |
| `LLama2Model::attention_mha()` | `llama3.cpp` | 652 | **MHA调用** |
| `MultiHeadAttention::forward()` | `mha.cpp` | 19 | MHA层前向传播 |
| `multi_head_attention_kernel()` | `mha_kernel.cu` | 49 | **CUDA核心** |
| `softmax_gpu()` | `mha_kernel.cu` | 9 | GPU Softmax |
| `get_mha_kernel()` | `kernels_interface.cpp` | - | Kernel获取 |

### 5.3 重要数据结构

```cpp
// MHA层定义
class MultiHeadAttention : public op::Layer {
    int32_t layer_index_;   // 当前层索引
    int32_t pos_;           // 当前token位置
    int32_t kv_mul_;        // GQA倍数
    int32_t kv_dim_;        // KV维度
    int32_t seq_len_;       // 序列长度
    int32_t head_num_;      // 注意力头数
    int32_t head_size_;     // 每个头的维度
};

// LLaMA层集合
struct LLama2Layers {
    std::shared_ptr<op::Layer> mha_layer_;         // 单个MHA层（共享）
    std::vector<std::shared_ptr<op::Layer>> wq_layers_;  // 每层的Query投影
    std::vector<std::shared_ptr<op::Layer>> wk_layers_;  // 每层的Key投影
    std::vector<std::shared_ptr<op::Layer>> wv_layers_;  // 每层的Value投影
    std::vector<std::shared_ptr<op::Layer>> wo_layers_;  // 每层的Output投影
    // ...
};

// Transformer配置
struct TransformerConfig {
    int32_t dim_;           // 模型维度
    int32_t layer_num_;     // 层数
    int32_t head_num_;      // 注意力头数
    int32_t head_size_;     // 每个头的维度
    int32_t kv_dim_;        // KV维度
    int32_t kv_mul_;        // GQA倍数
    int32_t seq_len_;       // 序列长度
};
```

---

## 6. 总结

### 6.1 核心要点

1. **Layer是Transformer的计算单元**
   - 每个Layer包含完整的Attention + FFN结构
   - LLaMA有多个Layer（如32层）
   - 通过遍历Layer实现深度Transformer

2. **MHA是Attention的核心**
   - 实现多头注意力机制
   - 使用CUDA并行计算加速
   - 支持GQA优化（减少KV Cache）

3. **高效设计**
   - MHA层在所有Layer间共享（通过layer_idx参数）
   - 每层有独立的权重矩阵
   - KV Cache分层存储，避免重复计算

4. **CUDA优化**
   - 每个注意力头一个Block
   - 共享内存缓存Query
   - float4向量化内存访问

### 6.2 MHA在推理中的角色

```
Prompt: "What is the capital of"

Token 0: "What"  → Layer 0 → Layer 1 → ... → Layer 31 → "is"
Token 1: "is"    → Layer 0 → Layer 1 → ... → Layer 31 → "the"
Token 2: "the"   → Layer 0 → Layer 1 → ... → Layer 31 → "capital"
Token 3: "capital"→ Layer 0 → Layer 1 → ... → Layer 31 → "of"
Token 4: "of"    → Layer 0 → Layer 1 → ... → Layer 31 → "France"

在每个Token的每个Layer中：
  MHA查看之前所有Token的KV Cache
  计算当前Token与历史Token的注意力
  聚合历史信息生成上下文表示
```

### 6.3 性能关键点

- **KV Cache大小**: `layer_num * seq_len * kv_dim * sizeof(float)`
  - LLaMA-7B: 32 * 2048 * 1024 * 4 = 256MB（单精度）

- **MHA计算复杂度**: O(pos² * head_size) 每层每Token
  - pos为当前序列长度，随推理增长

- **GQA优化**: 减少KV Cache至 `1/kv_mul`
  - kv_mul=8时，KV Cache减少87.5%

---

## 参考资料

1. 论文: *Attention Is All You Need* (Transformer原始论文)
2. 论文: *LLaMA: Open and Efficient Foundation Language Models*
3. 代码: https://github.com/wong-1994/KuiperLLama
4. CUDA编程指南: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

---

**文档生成时间**: 2026-03-22
**KuiperLLama版本**: 基于main分支
**作者**: AI技术分析
