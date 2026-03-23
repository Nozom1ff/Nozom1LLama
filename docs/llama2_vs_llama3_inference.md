# Llama2 与 Llama3 推理对比

## 编译选项对比

| 项目 | Llama2 | Llama3 |
|------|--------|--------|
| CMake 选项 | `cmake -DUSE_CPM=ON ..` | `cmake -DUSE_CPM=ON -DLLAMA3_SUPPORT=ON ..` |
| 关键宏定义 | 无 | `LLAMA3_SUPPORT` |

## 代码修改

### Tokenizer 类型修改

**文件**: `demo/main.cpp:57`

| 模型 | Tokenizer 类型 | 说明 |
|------|---------------|------|
| Llama2 | `base::TokenizerType::kEncodeSpe` | 使用 SentencePiece (`.model` 文件) |
| Llama3 | `base::TokenizerType::kEncodeBpe` | 使用 BPE/Tiktoken (`.json` 文件) |

**Llama2 代码**:
```cpp
model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path, checkpoint_path, false);
```

**Llama3 代码**:
```cpp
model::LLama2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
```

## Tokenizer 文件格式

| 模型 | Tokenizer 文件 | 格式 |
|------|---------------|------|
| Llama2 | `tokenizer.model` | SentencePiece protobuf 格式 |
| Llama3 | `tokenizer.json` | HuggingFace JSON 格式 |

## 运行命令

### Llama2
```bash
cmake -DUSE_CPM=ON ..
make -j$(nproc)
./demo/llama_infer <model.bin> <tokenizer.model>
```

### Llama3
```bash
cmake -DUSE_CPM=ON -DLLAMA3_SUPPORT=ON ..
make -j$(nproc)
./demo/llama_infer <model.bin> <tokenizer.json>
```

## 实现细节

### Tokenizer 实现 (kuiper/source/op/encode.cpp)

- `SpeEncodeLayer`: SentencePiece 实现，用于 Llama2
  - 加载 `.model` 文件
  - 使用 `sentencepiece::SentencePieceProcessor`

- `BpeEncodeLayer`: BPE/Tiktoken 实现，用于 Llama3 (需要 `LLAMA3_SUPPORT` 宏)
  - 加载 `.json` 文件
  - 使用 `tiktoken::tiktoken`
  - 解析 JSON 中的 `added_tokens` 和 `model/vocab`

### 特殊 Token

| 模型 | BOS Token | EOS Token |
|------|-----------|-----------|
| Llama2 | `<s>` (id=1) | `</s>` (id=2) |
| Llama3 | `<|begin_of_text|>` (id=128000) | `<|end_of_text|>` (id=128001) |

## 采样策略问题

### 问题根源
当前使用 `ArgmaxSampler`（贪婪采样），总是选择概率最高的 token，导致：
1. **重复输出** - 模型陷入循环
2. **缺少多样性** - 输出缺乏随机性

### 解决方案
需要实现以下采样策略之一1. **Temperature Sampling** - 添加温度参数控制随机性
2. **Top-k Sampling** - 只从概率最高的 k 个 token 中采样
3. **Top-p (Nucleus) Sampling** - 从累积概率达到 p 的 token 中采样
4. **Repetition Penalty** - 惩罚已出现的 token

## 当前实现 (sampler.h)
```cpp
// ArgmaxSampler: 贪婪采样，virtual int32_t sample(const float* logits, int64_t size, void* stream = nullptr) override {
    // 找到最大概率的索引
    return argmax(logits, size, stream);
}
```

**ArgmaxSampler** 返回的是 logits 数组中最大值的索引。这对于确定性任务可能有用，但对于文本生成会导致重复。

## 推荐改进

实现 **Temperature Sampling + Softmax**:

```cpp
// TemperatureSampler: 温度采样
virtual int32_t sample(const float* logits, int64_t size, float temperature, void* stream) {
    // 1. 对 logits 应用温度缩放
    for (int i = 0; i < size; i++) {
        logits[i] = logits[i] / temperature;
    }
    // 2. Softmax 转为概率
    softmax(logits, size, stream);
    // 3. 按概率采样
    return random_sample(probabilities);
}
```

**温度参数** 控制随机性：
- `temperature = 1.0`: 贪婪采样（接近 argmax）
- `temperature = 0.7`: 平衡采样
- `temperature > 1.0`: 更随机/创造性

## 其他建议改进

1. **Top-k Sampling**: 只保留前 k 个最高概率的 token
2. **Top-p Sampling**: 累积概率达到 p 后停止
3. **Repetition Penalty**: 降低已出现 token 的概率
