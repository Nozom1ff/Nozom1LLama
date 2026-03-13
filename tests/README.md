# Nozom1LLama 测试说明

## 测试目录结构

```
tests/
├── README.md                    # 本文件
├── test_base/                   # 基础模块测试
│   ├── test_status.cpp         # Status 类测试
│   ├── test_datatype.cpp       # 数据类型测试
│   ├── test_buffer.cpp         # Buffer 类测试（待添加）
│   └── test_allocator.cpp      # Allocator 测试（待添加）
├── test_tensor/                 # 张量模块测试（待添加）
├── test_op/                     # 算子测试（待添加）
└── test_model/                  # 模型测试（待添加）
```

## 编译和运行测试

### 1. 安装依赖

#### 使用系统包管理器
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y libgtest-dev libgoogle-glog-dev libsentencepiece-dev

# Fedora
sudo dnf install -y gtest-devel glog-devel sentencepiece-devel
```

#### 使用 CPM（推荐）
```bash
mkdir build
cd build
cmake -DUSE_CPM=ON ..
```

### 2. 编译项目

```bash
# 创建构建目录
mkdir build
cd build

# 配置 CMake
cmake ..

# 或者使用 CPM 自动下载依赖
cmake -DUSE_CPM=ON ..

# 编译
make -j$(nproc)
```

### 3. 运行所有测试

```bash
# 使用 CTest 运行所有测试
cd build
ctest --output-on-failure

# 或者直接运行测试可执行文件
./tests/test_status
./tests/test_datatype
```

### 4. 运行特定测试

```bash
# 只运行一个测试文件
./tests/test_status

# 运行特定测试用例
./tests/test_status --gtest_filter=StatusTest.DefaultConstructorShouldBeSuccess
```

## 测试编写规范

### 1. 文件命名

- 测试文件：`test_<module_name>.cpp`
- 例如：`test_status.cpp`, `test_tensor.cpp`

### 2. 测试结构

```cpp
/**
 * @file test_xxx.cpp
 * @brief 测试 xxx 功能
 */

#include <gtest/gtest.h>
#include "base/base.h"  // 或其他被测试的头文件

using namespace base;

// ==================== 测试组 ====================

/**
 * @test 测试描述
 */
TEST(TestGroupName, TestName) {
    // Arrange（准备）
    int expected = 42;

    // Act（执行）
    int actual = function_under_test();

    // Assert（断言）
    EXPECT_EQ(actual, expected);
}

// ==================== 主函数 ====================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

### 3. 断言宏

| 宏 | 说明 |
|---|---|
| `EXPECT_EQ(expected, actual)` | 相等（失败后继续） |
| `ASSERT_EQ(expected, actual)` | 相等（失败后终止） |
| `EXPECT_TRUE(condition)` | 条件为真 |
| `EXPECT_FALSE(condition)` | 条件为假 |
| `EXPECT_NE(val1, val2)` | 不相等 |
| `EXPECT_GT(val1, val2)` | 大于 |
| `EXPECT_LT(val1, val2)` | 小于 |

### 4. 测试命名规范

- 测试组名：`<ClassName>Test` 或 `<ModuleName>Test`
- 测试用例名：`<WhatBeingTested>_<ExpectedBehavior>`
- 例如：`StatusTest.DefaultConstructorShouldBeSuccess`

## 当前测试覆盖

### ✅ 已完成

- [x] `test_status.cpp` - Status 类完整测试
  - 构造和析构
  - 比较操作
  - 消息操作
  - error 命名空间函数

- [x] `test_datatype.cpp` - 数据类型测试
  - DataType 枚举和大小
  - DeviceType 枚举
  - ModelType 枚举
  - ModelBufferType 枚举
  - TokenizerType 枚举
  - NoCopyable 类

### 📝 待添加

- [ ] `test_buffer.cpp` - Buffer 类测试
- [ ] `test_allocator.cpp` - 内存分配器测试
- [ ] `test_tensor.cpp` - Tensor 类测试
- [ ] `test_matmul.cpp` - 矩阵乘法测试
- [ ] `test_rmsnorm.cpp` - RMSNorm 测试
- [ ] `test_mha.cpp` - Multi-Head Attention 测试
- [ ] `test_llama.cpp` - LLaMA 模型测试

## 测试目标

### 功能正确性
- ✅ 基础类型和枚举
- ✅ Status 错误处理
- ⏳ 内存管理
- ⏳ 张量操作
- ⏳ 算子计算
- ⏳ 模型推理

### 性能测试（可选）
- ⏳ 矩阵乘法性能
- ⏳ 内存分配速度
- ⏳ 整体推理速度

### 边界情况
- ⏳ 空输入
- ⏳ 超大张量
- ⏳ OOM 处理

## 持续集成（未来）

### GitHub Actions 示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: |
          sudo apt update
          sudo apt install -y libgtest-dev libgoogle-glog-dev libsentencepiece-dev
      - name: Build
        run: |
          mkdir build && cd build
          cmake ..
          make -j$(nproc)
      - name: Run tests
        run: |
          cd build
          ctest --output-on-failure
```

## 常见问题

### Q: 编译时找不到 gtest
**A**: 使用 `-DUSE_CPM=ON` 让 CMake 自动下载，或手动安装：
```bash
sudo apt install libgtest-dev
```

### Q: 运行测试时出现段错误
**A**: 使用 gdb 调试：
```bash
gdb ./tests/test_status
(gdb) run
(gdb) bt  # 查看堆栈
```

### Q: 如何添加新测试？
**A**: 1. 在对应目录创建 `test_xxx.cpp`
2. 编写测试代码
3. 重新编译 `make`
4. CMake 会自动发现并添加新测试

## 贡献指南

添加新测试时，请确保：

1. ✅ 每个测试只测试一个功能点
2. ✅ 测试名称清晰描述测试内容
3. ✅ 添加必要的注释说明测试目的
4. ✅ 使用 `EXPECT_*` 而非 `ASSERT_*`（除非必须终止）
5. ✅ 测试独立运行，不依赖其他测试

## 参考资料

- [Google Test Primer](https://google.github.io/googletest/primer.html)
- [Google Test Advanced Guide](https://google.github.io/googletest/advanced.html)
- [CMake Testing](https://cmake.org/cmake/help/latest/manual/ctest.1.html)
