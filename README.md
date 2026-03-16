# Nozom1LLama

> 从零开始构建的高性能 LLM 推理框架，支持 CUDA 加速

## 项目简介

Nozom1LLama 是一个学习型项目，目标是从零开始构建一个完整的 LLaMA 模型推理框架，支持：

- ✅ **现代 C++20** - 使用最新的 C++ 特性
- ✅ **CUDA 加速** - GPU 推理优化
- ✅ **FP16 支持** - 半精度浮点，减少内存占用
- ✅ **模块化设计** - 清晰的架构分层
- ✅ **完整测试** - 单元测试覆盖

## 项目结构

```
Nozom1LLama/
├── CMakeLists.txt                    # CMake 构建配置
├── README.md                         # 本文件
├── .clang-format                     # 代码格式化配置
├── nozom1/                           # 源代码目录
│   ├── include/                      # 头文件
│   │   ├── base/                     # 基础模块
│   │   │   ├── alloc.h               # 内存分配器
│   │   │   ├── base.h                # 基础类型定义
│   │   │   ├── buffer.h              # 缓冲区管理
│   │   │   ├── cuda_config.h         # CUDA 配置
│   │   │   └── unicode.h             # Unicode 处理
│   │   ├── tensor/                   # Tensor 模块
│   │   │   └── tensor.h              # Tensor 类定义
│   │   └── op/                       # 算子模块
│   │       ├── layer.h               # Layer 基类
│   │       ├── add.h                 # 向量加法层
│   │       └── embedding.h           # Embedding 层
│   └── source/                       # 源文件
│       ├── base/                     # 基础模块实现
│       ├── tensor/                   # Tensor 实现
│       ├── op/                       # 算子实现
│       │   ├── layer.cpp
│       │   ├── add.cpp
│       │   └── embedding.cpp
│       └── op/kernels/               # CUDA 内核
│           ├── kernels_interfaces.h  # 内核接口
│           └── cuda/                 # CUDA 实现
│               ├── add_kernel.cu
│               └── embeddingg_kernel.cu
└── tests/                            # 测试目录
    ├── test_base/                    # 基础模块测试
    │   ├── test_status.cpp
    │   ├── test_datatype.cpp
    │   ├── test_allocator.cpp
    │   └── test_buffer.cpp
    ├── test_tensor/                  # Tensor 测试
    │   └── test_tensor.cpp
    └── test_op/                      # 算子测试
        ├── test_add.cpp
        └── test_embedding.cpp
```

## 依赖

### 必需
- **CUDA Toolkit** 11.0+ (推荐 12.x，支持 Blackwell 架构 RTX 5090)
- **CMake** 3.16+
- **C++17** 兼容编译器 (GCC 9+ / Clang 10+)

### 第三方库
- **Google GTest** 1.11.0+ - 单元测试框架
- **Google GLog** 0.7.0+ - 日志系统
- **SentencePiece** 0.1.96+ - 分词器

## 快速开始

### 1. 安装依赖

#### 使用系统包管理器
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    libgtest-dev \
    libgoogle-glog-dev \
    libsentencepiece-dev \
    cuda-nvcc-12-x \
    cmake

# 或使用 CPM 自动下载（推荐用于开发）
cmake -DUSE_CPM=ON ..
```

#### 检查 CUDA 版本
```bash
nvcc --version  # 应该显示 11.0+ 或更高
nvidia-smi      # 检查 GPU 驱动
```

### 2. 编译项目

```bash
# 克隆仓库
git clone <repository-url>
cd Nozom1LLama

# 创建构建目录
mkdir build && cd build

# 配置（支持从 Turing 到 Blackwell 架构）
cmake -DCMAKE_BUILD_TYPE=Release ..

# 编译（使用所有 CPU 核心）
make -j$(nproc)
```

**编译选项：**
- `-DCMAKE_BUILD_TYPE=Debug` - 启用调试符号和禁用优化
- `-DCMAKE_BUILD_TYPE=Release` - 启用优化（默认）
- `-DBUILD_TESTS=OFF` - 禁用测试编译
- `-DUSE_CPM=ON` - 使用 CPM 自动下载依赖

### 3. 运行测试

```bash
cd build

# 运行所有测试
ctest --output-on-failure

# 或直接运行测试可执行文件
./test/test_base/test_status
./test/test_base/test_datatype
./test/test_base/test_allocator
./test/test_base/test_buffer
./test/test_tensor/test_tensor
./test/test_op/test_add
./test/test_op/test_embedding

# 运行特定测试用例
./test/test_op/test_embedding --gtest_filter=EmbeddingLayerTest.BasicEmbedding
```

## 开发路线

### ✅ 已完成
- [x] 基础架构设计
- [x] Status 错误处理系统
- [x] DataType 支持（FP32, FP16, Int8, Int32）
- [x] 单元测试框架集成
- [x] 内存管理系统
  - [x] CPU 内存分配器 (CPUDeviceAllocator)
  - [x] CUDA 内存分配器 (CUDADeviceAllocator)
  - [x] Buffer 管理
- [x] Tensor 类实现
  - [x] 多维张量支持
  - [x] CPU/CUDA 设备间数据传输
  - [x] 自动内存管理
- [x] Layer 基础框架
  - [x] Layer 基类
  - [x] LayerParam 派生类
  - [x] 输入/输出/权重管理
- [x] 算子实现
  - [x] VecAddLayer (向量加法，支持 CUDA)
  - [x] EmbeddingLayer (词嵌入层，支持 CUDA)

### 🚧 进行中
- [ ] 更多算子实现
  - [ ] MatMul (矩阵乘法)
  - [ ] RMSNorm (均方根层归一化)
  - [ ] SwiGLU (激活函数)

### 📋 计划中
- [ ] Multi-Head Attention (多头注意力机制)
- [ ] RoPE 位置编码
- [ ] Softmax 算子
- [ ] LLaMA 模型组装
- [ ] 文本生成流程
- [ ] KV Cache 优化
- [ ] Int8/FP4 量化
- [ ] Flash Attention

## 代码风格

项目使用 `.clang-format` 进行代码格式化：

```bash
# 格式化所有代码
find . -name "*.cpp" -o -name "*.h" | xargs clang-format -i
```

## 核心功能

### Tensor 模块

高性能多维张量实现，支持：
- 多维张量创建和操作
- CPU 和 CUDA 设备间无缝数据传输
- 自动内存管理
- 支持多种数据类型（FP32, FP16, Int32, Int8）

```cpp
// 创建一个 3x4 的 FP32 tensor on CUDA
Tensor tensor(DataType::kTypeFp32, {3, 4}, true, cuda_allocator);

// CPU -> CUDA 数据传输
tensor.to_cuda();

// 访问数据
float* data = tensor.ptr<float>();
```

### 算子模块

#### VecAddLayer (向量加法)
- 支持 CPU 和 CUDA 加速
- 自动输入验证
- 高性能并行计算

```cpp
VecAddLayer layer(DeviceType::kCUDA, DataType::kTypeFp32);
layer.set_cuda_config(cuda_config);
layer.set_input(0, input1);
layer.set_input(1, input2);
layer.set_output(0, output);
layer.forward();
```

#### EmbeddingLayer (词嵌入层)
- 高效词向量查找
- 支持任意词汇表大小和嵌入维度
- CUDA 加速，适合大规模批处理

```cpp
EmbeddingLayer layer(DeviceType::kCUDA, dim=128, seq_len=64, vocab_size=10000);
layer.set_cuda_config(cuda_config);
layer.set_input(0, token_ids);      // token ids (CPU)
layer.set_weight(0, embeddings);     // embedding matrix (CUDA)
layer.set_output(0, output);         // output embeddings (CUDA)
layer.forward();
```

### 性能指标

在 RTX 5090 (Blackwell 架构) 上的测试结果：

| 操作 | 规模 | 吞吐量 |
|------|------|--------|
| VecAdd | 1M 元素 | ~8.5B 元素/秒 |
| Embedding | 64 tokens, vocab=10K | ~1.0M tokens/秒 |
| Embedding | 64 tokens, vocab=1K | ~4.1M tokens/秒 |

## 测试

查看 [tests/README.md](./tests/README.md) 了解详细的测试说明。

### 运行特定测试

```bash
# 只运行 Status 测试
./test/test_base/test_status

# 运行特定测试用例
./test/test_op/test_embedding --gtest_filter=EmbeddingLayerTest.BasicEmbedding

# 运行性能测试
./test/test_op/test_embedding --gtest_filter=*PerformanceTest
```

## 学习资源

本项目参考了：
- [KuiperLLama](https://github.com/defysics/kuiper_infer) - 原始课程项目
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU 优化参考
- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention 参考

## 贡献

欢迎提交 Issue 和 Pull Request！

### 开发流程
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目仅用于学习目的。

## 作者

Nozom1

## 致谢

- KuiperInfer 课程
- CUDA 和深度学习社区
