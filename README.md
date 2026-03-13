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
├── CMakeLists.txt           # CMake 构建配置
├── README.md                # 本文件
├── .clang-format            # 代码格式化配置
├── nozom1/                  # 源代码目录
│   ├── include/             # 头文件
│   │   └── base/            # 基础模块
│   │       └── base.h       # 基础类型定义
│   └── source/              # 源文件
│       └── base/            # 基础模块实现
│           └── base.cpp     # Status 类实现
└── tests/                   # 测试目录
    ├── README.md            # 测试说明
    └── test_base/           # 基础模块测试
        ├── test_status.cpp  # Status 测试
        └── test_datatype.cpp # 数据类型测试
```

## 依赖

### 必需
- **CUDA Toolkit** 11.0+ (推荐 11.8 或 12.x)
- **CMake** 3.16+
- **C++17** 兼容编译器

### 第三方库
- **Google GTest** - 单元测试框架
- **Google GLog** - 日志系统
- **SentencePiece** - 分词器

## 快速开始

### 1. 安装依赖

#### 使用系统包管理器
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y libgtest-dev libgoogle-glog-dev libsentencepiece-dev

# 或使用 CPM 自动下载（推荐）
```

### 2. 编译项目

```bash
# 创建构建目录
mkdir build
cd build

# 配置（使用 CPM 自动下载依赖）
cmake -DUSE_CPM=ON ..

# 或使用系统依赖
cmake ..

# 编译
make -j$(nproc)
```

### 3. 运行测试

```bash
cd build

# 运行所有测试
ctest --output-on-failure

# 或直接运行测试可执行文件
./tests/test_status
./tests/test_datatype
```

## 开发路线

### ✅ 已完成
- [x] 基础架构设计
- [x] Status 错误处理
- [x] DataType 支持（FP32, FP16, Int8, Int32）
- [x] 单元测试框架

### 🚧 进行中
- [ ] Tensor 类实现
- [ ] 内存管理系统
- [ ] 算子实现

### 📋 计划中
- [ ] 矩阵乘法（CPU + CUDA）
- [ ] RMSNorm 层
- [ ] Multi-Head Attention
- [ ] RoPE 位置编码
- [ ] LLaMA 模型组装
- [ ] 文本生成流程
- [ ] Int8 量化
- [ ] Flash Attention

## 代码风格

项目使用 `.clang-format` 进行代码格式化：

```bash
# 格式化所有代码
find . -name "*.cpp" -o -name "*.h" | xargs clang-format -i
```

## 测试

查看 [tests/README.md](./tests/README.md) 了解详细的测试说明。

### 运行特定测试

```bash
# 只运行 Status 测试
./tests/test_status

# 运行特定测试用例
./tests/test_status --gtest_filter=StatusTest.DefaultConstructorShouldBeSuccess
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
