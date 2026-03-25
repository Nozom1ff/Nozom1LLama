# Qwen与LLaMA模型对比分析

本文档详细分析了KuiperLLama项目中Qwen系列模型（Qwen2、Qwen3）与LLaMA系列模型（LLaMA2、LLaMA3）的实现差异。

## 1. 模型概述

| 模型 | 开发者 | 架构类型 | 本项目支持 |
|------|--------|----------|------------|
| LLaMA2 | Meta | Decoder-only Transformer | 是 |
| LLaMA3 | Meta | Decoder-only Transformer | 是 |
| Qwen2 | 阿里云 | Decoder-only Transformer | 是 |
| Qwen3 | 阿里云 | Decoder-only Transformer | 是 |

## 2. 架构对比

### 2.1 核心层结构

两个模型系列都采用类似的Transformer Decoder架构，包含以下核心层：

```
┌─────────────────────────────────────────────┐
│              Token Embedding                │
├─────────────────────────────────────────────┤
│  ┌───────────────────────────────────────┐  │
│  │         Attention Block               │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │ RMSNorm -> QKV Projection       │  │  │
│  │  │ -> RoPE -> Multi-Head Attention │  │  │
│  │  │ -> Output Projection -> Residual│  │  │
│  │  └─────────────────────────────────┘  │  │
│  └───────────────────────────────────────┘  │
├─────────────────────────────────────────────┤
│  ┌───────────────────────────────────────┐  │
│  │          FFN Block                    │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │ RMSNorm -> W1/W3 -> SwiGLU      │  │  │
│  │  │ -> W2 -> Residual               │  │  │
│  │  └─────────────────────────────────┘  │  │
│  └───────────────────────────────────────┘  │
├─────────────────────────────────────────────┤
│         Final RMSNorm -> LM Head            │
└─────────────────────────────────────────────┘
```

### 2.2 层结构定义对比

#### LLaMA层结构 (`kuiper/include/model/llama3.h:11-31`)
```cpp
struct LLama2Layers {
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  std::shared_ptr<op::Layer> cls_layer_;
  std::shared_ptr<op::Layer> embedding_layer_;
};
```

#### Qwen层结构 (`kuiper/include/model/qwen2.h:11-31`)
```cpp
struct Qwen2Layers {
  // 与LLaMA结构相同
  // ...相同的层定义
};
```

## 3. 关键差异分析

### 3.1 Attention Bias

**这是Qwen与LLaMA最重要的区别之一。**

#### LLaMA（无Bias）
```cpp
// kuiper/source/model/llama3.cpp:307
auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
// 注意：没有set_bias调用
```

#### Qwen2（有Bias）
```cpp
// kuiper/source/model/qwen2.cpp:307-312
auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false, true);
wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
pos += dim * dim;
wq->set_bias(0, dim, this->raw_model_data_->weight(pos), cpu_device_type);  // Qwen有bias
pos += dim;
```

**影响**：
- Qwen2的Q、K、V投影层包含bias参数
- LLaMA的Q、K、V投影层不包含bias参数
- 这导致权重文件格式和内存布局不同

### 3.2 配置结构差异

#### 基础配置 (`kuiper/include/model/config.h:4-15`)
```cpp
struct ModelConfig {
  int32_t dim = 0;
  int32_t hidden_dim = 0;
  int32_t layer_num = 0;
  int32_t head_num = 0;
  int32_t kv_head_num = 0;
  int32_t vocab_size = 0;
  int32_t seq_len = 0;
#ifdef QWEN3_SUPPORT
  int32_t immediate_dim_ = 0;  // Qwen3特有参数
#endif
};
```

#### Qwen3专用配置 (`kuiper/include/model/qwen3.h:10-24`)
```cpp
struct QWen3TransformerConfig {
  int32_t kv_dim_ = 0;
  int32_t kv_mul_ = 0;
  int32_t head_size_ = 0;
  int32_t immediate_size_ = 0;  // Qwen3特有：中间层维度
  int32_t vocab_size_ = 0;
  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;
  int32_t layer_num_ = 0;
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t seq_len_ = 0;
  bool is_shared_weight_ = false;
};
```

### 3.3 Tokenizer类型差异

| 模型 | Tokenizer类型 | 枚举值 | 文件格式 |
|------|---------------|--------|----------|
| LLaMA2 | SentencePiece | `kEncodeSpe` | `.model` |
| LLaMA3 | BPE/Tiktoken | `kEncodeBpe` | `.json` |
| Qwen2 | BPE/Tiktoken | `kEncodeBpe` | `.json` |
| Qwen3 | BPE/Tiktoken | `kEncodeBpe` | `.json` |

### 3.4 权重导出格式差异

#### LLaMA权重导出 (`tools/export_llama3.py:100-112`)
```python
# Attention weights (无bias)
for layer in model.layers:
    serialize_fp32(out_file, layer.attention.wq.weight)
for layer in model.layers:
    serialize_fp32(out_file, layer.attention.wk.weight)
for layer in model.layers:
    serialize_fp32(out_file, layer.attention.wv.weight)
for layer in model.layers:
    serialize_fp32(out_file, layer.attention.wo.weight)
```

#### Qwen2权重导出 (`tools/export_qwen2.py:103-111`)
```python
# Attention weights (包含bias)
for layer in model.layers:
    serialize_fp32(out_file, layer.attention.wq.weight)
    serialize_fp32(out_file, layer.attention.wq.bias)  # Qwen有bias
for layer in model.layers:
    serialize_fp32(out_file, layer.attention.wk.weight)
    serialize_fp32(out_file, layer.attention.wk.bias)  # Qwen有bias
for layer in model.layers:
    serialize_fp32(out_file, layer.attention.wv.weight)
    serialize_fp32(out_file, layer.attention.wv.bias)  # Qwen有bias
```

## 4. 权重内存布局对比

### 4.1 LLaMA权重布局

```
┌────────────────────────────────────────────┐
│           Embedding Weights                │
│         (vocab_size × dim)                 │
├────────────────────────────────────────────┤
│         Attention RMSNorm Weights          │
│          (layer_num × dim)                 │
├────────────────────────────────────────────┤
│              WQ Weights                    │
│          (layer_num × dim × dim)           │
├────────────────────────────────────────────┤
│              WK Weights                    │
│      (layer_num × kv_dim × dim)            │
├────────────────────────────────────────────┤
│              WV Weights                    │
│      (layer_num × kv_dim × dim)            │
├────────────────────────────────────────────┤
│              WO Weights                    │
│          (layer_num × dim × dim)           │
├────────────────────────────────────────────┤
│              ...FFN Weights...             │
└────────────────────────────────────────────┘
```

### 4.2 Qwen2权重布局

```
┌────────────────────────────────────────────┐
│           Embedding Weights                │
│         (vocab_size × dim)                 │
├────────────────────────────────────────────┤
│         Attention RMSNorm Weights          │
│          (layer_num × dim)                 │
├────────────────────────────────────────────┤
│         WQ Weights + Biases                │
│    (layer_num × (dim × dim + dim))         │
├────────────────────────────────────────────┤
│         WK Weights + Biases                │
│  (layer_num × (kv_dim × dim + kv_dim))     │
├────────────────────────────────────────────┤
│         WV Weights + Biases                │
│  (layer_num × (kv_dim × dim + kv_dim))     │
├────────────────────────────────────────────┤
│              WO Weights                    │
│          (layer_num × dim × dim)           │
├────────────────────────────────────────────┤
│              ...FFN Weights...             │
└────────────────────────────────────────────┘
```

## 5. 编译选项

```bash
# LLaMA2
cmake -DUSE_CPM=ON ..

# LLaMA3
cmake -DUSE_CPM=ON -DLLAMA3_SUPPORT=ON ..

# Qwen系列（使用LLaMA3的编译选项）
cmake -DUSE_CPM=ON -DLLAMA3_SUPPORT=ON ..

# Qwen3（需要额外宏定义）
cmake -DUSE_CPM=ON -DLLAMA3_SUPPORT=ON -DQWEN3_SUPPORT=ON ..
```

## 6. 模型类继承关系

```
                    ┌─────────────┐
                    │    Model    │ (基类)
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────┴──────┐ ┌──────┴──────┐ ┌──────┴──────┐
    │ LLama2Model │ │  Qwen2Model │ │  Qwen3Model │
    └─────────────┘ └─────────────┘ └─────────────┘
```

所有模型类都继承自`Model`基类，实现统一的接口：
- `init()` - 初始化模型
- `predict()` - 预测下一个token
- `forward()` - 前向传播
- `embedding()` - 词嵌入

## 7. 总结

### 主要差异

| 特性 | LLaMA系列 | Qwen系列 |
|------|-----------|----------|
| Q/K/V Bias | 无 | 有 |
| 配置参数 | 标准TransformerConfig | Qwen3有immediate_dim_ |
| Tokenizer | LLaMA2: SentencePiece<br>LLaMA3: BPE | BPE |
| 权重文件大小 | 相对较小 | 略大（包含bias） |

### 共同点

1. **核心架构相同**：都使用Decoder-only Transformer
2. **归一化方式**：都使用RMSNorm
3. **位置编码**：都使用RoPE（Rotary Position Embedding）
4. **激活函数**：都使用SwiGLU
5. **注意力机制**：都支持GQA（Grouped Query Attention）

### 开发建议

1. **权重转换**：从HuggingFace导出时，注意Qwen模型需要处理bias参数
2. **内存规划**：Qwen模型需要额外的内存存储bias参数
3. **兼容性**：如需支持两种模型，建议使用条件编译或运行时配置切换
