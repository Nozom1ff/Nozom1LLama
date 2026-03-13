/**
 * @file test_tensor.cpp
 * @brief 测试 Tensor 类的功能
 */

#include <gtest/gtest.h>
#include "tensor/tensor.h"
#include "base/alloc.h"
#include <memory>
#include <cstring>
#include <vector>
#include <chrono>

using namespace tensor;

// ==================== 基础功能测试 ====================

/**
 * @test 测试默认构造函数
 */
TEST(TensorTest, DefaultConstructor) {
    Tensor tensor;

    EXPECT_EQ(tensor.size(), 0);
    EXPECT_EQ(tensor.dims_size(), 0);
    EXPECT_EQ(tensor.data_type(), base::DataType::kUnknown);
    EXPECT_EQ(tensor.get_buffer(), nullptr);
}

/**
 * @test 测试 1D Tensor 构造（不分配内存）
 */
TEST(TensorTest, Construct1DWithoutAllocation) {
    Tensor tensor(base::DataType::kTypeFp32, 10);

    EXPECT_EQ(tensor.size(), 10);
    EXPECT_EQ(tensor.dims_size(), 1);
    EXPECT_EQ(tensor.get_dim(0), 10);
    EXPECT_EQ(tensor.data_type(), base::DataType::kTypeFp32);
    EXPECT_EQ(tensor.get_buffer(), nullptr);  // 未分配内存
}

/**
 * @test 测试 1D Tensor 构造（分配内存）
 */
TEST(TensorTest, Construct1DWithAllocation) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp32, 10, true, allocator);

    EXPECT_EQ(tensor.size(), 10);
    EXPECT_EQ(tensor.dims_size(), 1);
    EXPECT_NE(tensor.get_buffer(), nullptr);
    EXPECT_NE(tensor.ptr<float>(), nullptr);
    EXPECT_EQ(tensor.device_type(), base::DeviceType::kCPU);
}

/**
 * @test 测试 2D Tensor 构造
 */
TEST(TensorTest, Construct2D) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp32, 2, 3, true, allocator);

    EXPECT_EQ(tensor.size(), 6);  // 2 * 3 = 6
    EXPECT_EQ(tensor.dims_size(), 2);
    EXPECT_EQ(tensor.get_dim(0), 2);
    EXPECT_EQ(tensor.get_dim(1), 3);
    EXPECT_NE(tensor.ptr<float>(), nullptr);
}

/**
 * @test 测试 3D Tensor 构造
 */
TEST(TensorTest, Construct3D) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp32, 2, 3, 4, true, allocator);

    EXPECT_EQ(tensor.size(), 24);  // 2 * 3 * 4 = 24
    EXPECT_EQ(tensor.dims_size(), 3);
    EXPECT_EQ(tensor.get_dim(0), 2);
    EXPECT_EQ(tensor.get_dim(1), 3);
    EXPECT_EQ(tensor.get_dim(2), 4);
    EXPECT_NE(tensor.ptr<float>(), nullptr);
}

/**
 * @test 测试 4D Tensor 构造
 */
TEST(TensorTest, Construct4D) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp32, 2, 3, 4, 5, true, allocator);

    EXPECT_EQ(tensor.size(), 120);  // 2 * 3 * 4 * 5 = 120
    EXPECT_EQ(tensor.dims_size(), 4);
    EXPECT_EQ(tensor.get_dim(0), 2);
    EXPECT_EQ(tensor.get_dim(1), 3);
    EXPECT_EQ(tensor.get_dim(2), 4);
    EXPECT_EQ(tensor.get_dim(3), 5);
    EXPECT_NE(tensor.ptr<float>(), nullptr);
}

/**
 * @test 测试使用 vector 构造 Tensor
 */
TEST(TensorTest, ConstructWithVector) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims = {2, 3, 4};
    Tensor tensor(base::DataType::kTypeFp32, dims, true, allocator);

    EXPECT_EQ(tensor.size(), 24);
    EXPECT_EQ(tensor.dims_size(), 3);
    EXPECT_EQ(tensor.get_dim(0), 2);
    EXPECT_EQ(tensor.get_dim(1), 3);
    EXPECT_EQ(tensor.get_dim(2), 4);
}

// ==================== 数据类型测试 ====================

/**
 * @test 测试 Float32 Tensor
 */
TEST(TensorTest, DataTypeFloat32) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp32, 10, true, allocator);

    EXPECT_EQ(tensor.data_type(), base::DataType::kTypeFp32);
    EXPECT_EQ(tensor.byte_size(), 10 * sizeof(float));
    EXPECT_NE(tensor.ptr<float>(), nullptr);
}

/**
 * @test 测试 Int32 Tensor
 */
TEST(TensorTest, DataTypeInt32) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeInt32, 10, true, allocator);

    EXPECT_EQ(tensor.data_type(), base::DataType::kTypeInt32);
    EXPECT_EQ(tensor.byte_size(), 10 * sizeof(int32_t));
    EXPECT_NE(tensor.ptr<int32_t>(), nullptr);
}

/**
 * @test 测试 Int8 Tensor
 */
TEST(TensorTest, DataTypeInt8) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeInt8, 10, true, allocator);

    EXPECT_EQ(tensor.data_type(), base::DataType::kTypeInt8);
    EXPECT_EQ(tensor.byte_size(), 10 * sizeof(int8_t));
    EXPECT_NE(tensor.ptr<int8_t>(), nullptr);
}

/**
 * @test 测试 Float16 (__half) Tensor
 */
TEST(TensorTest, DataTypeFloat16) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp16, 10, true, allocator);

    EXPECT_EQ(tensor.data_type(), base::DataType::kTypeFp16);
    EXPECT_EQ(tensor.byte_size(), 10 * sizeof(__half));
    EXPECT_NE(tensor.ptr<__half>(), nullptr);
}

/**
 * @test 测试 __half 数据写入和读取
 */
TEST(TensorTest, HalfDataWriteAndRead) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp16, 10, true, allocator);

    // 写入数据：将 float 转换为 __half
    __half* data = tensor.ptr<__half>();
    for (int i = 0; i < 10; ++i) {
        float fval = static_cast<float>(i) * 0.5f;
        data[i] = __float2half(fval);  // CUDA 内置函数
    }

    // 读取数据：将 __half 转换回 float 验证
    for (int i = 0; i < 10; ++i) {
        __half hval = data[i];
        float fval = __half2float(hval);  // CUDA 内置函数
        EXPECT_NEAR(fval, static_cast<float>(i) * 0.5f, 0.001f);
    }
}

/**
 * @test 测试 __half index 访问
 */
TEST(TensorTest, HalfIndexAccess) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp16, 10, true, allocator);

    // 使用 index 写入
    for (int i = 0; i < 10; ++i) {
        float fval = static_cast<float>(i) * 1.5f;
        tensor.index<__half>(i) = __float2half(fval);
    }

    // 使用 index 读取
    for (int i = 0; i < 10; ++i) {
        __half hval = tensor.index<__half>(i);
        float fval = __half2float(hval);
        EXPECT_NEAR(fval, static_cast<float>(i) * 1.5f, 0.001f);
    }
}

/**
 * @test 测试 __half const ptr 访问
 */
TEST(TensorTest, HalfConstPtr) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp16, 10, true, allocator);

    // 写入数据
    __half* data = tensor.ptr<__half>();
    for (int i = 0; i < 10; ++i) {
        float fval = static_cast<float>(i);
        data[i] = __float2half(fval);
    }

    // 测试 const 版本
    const Tensor& const_tensor = tensor;
    const __half* const_data = const_tensor.ptr<__half>();

    for (int i = 0; i < 10; ++i) {
        float fval = __half2float(const_data[i]);
        EXPECT_NEAR(fval, static_cast<float>(i), 0.001f);
    }
}

/**
 * @test 测试 __half 内存大小对比 Float32
 */
TEST(TensorTest, HalfVsFloatMemorySize) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();

    Tensor float_tensor(base::DataType::kTypeFp32, 1000, true, allocator);
    Tensor half_tensor(base::DataType::kTypeFp16, 1000, true, allocator);

    // __half 占用内存应该是 float 的一半
    EXPECT_EQ(half_tensor.byte_size(), float_tensor.byte_size() / 2);
    EXPECT_EQ(half_tensor.size(), float_tensor.size());
}

/**
 * @test 测试 __half 2D Tensor 数据布局
 */
TEST(TensorTest, HalfDataLayout2D) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp16, 2, 3, true, allocator);

    __half* data = tensor.ptr<__half>();

    // 填充数据
    // Row 0: [0.5, 1.5, 2.5]
    // Row 1: [10.5, 11.5, 12.5]
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            float fval = static_cast<float>(i * 10 + j) + 0.5f;
            data[i * 3 + j] = __float2half(fval);
        }
    }

    // 验证数据
    float val00 = __half2float(data[0]);
    float val01 = __half2float(data[1]);
    float val02 = __half2float(data[2]);
    float val10 = __half2float(data[3]);
    float val11 = __half2float(data[4]);
    float val12 = __half2float(data[5]);

    EXPECT_NEAR(val00, 0.5f, 0.001f);
    EXPECT_NEAR(val01, 1.5f, 0.001f);
    EXPECT_NEAR(val02, 2.5f, 0.001f);
    EXPECT_NEAR(val10, 10.5f, 0.001f);
    EXPECT_NEAR(val11, 11.5f, 0.001f);
    EXPECT_NEAR(val12, 12.5f, 0.001f);
}

// ==================== 数据访问测试 ====================

/**
 * @test 测试数据写入和读取
 */
TEST(TensorTest, DataWriteAndRead) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp32, 10, true, allocator);

    // 写入数据
    float* data = tensor.ptr<float>();
    for (int i = 0; i < 10; ++i) {
        data[i] = static_cast<float>(i) * 1.5f;
    }

    // 读取数据
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i) * 1.5f);
    }
}

/**
 * @test 测试 index 访问元素
 */
TEST(TensorTest, IndexAccess) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp32, 10, true, allocator);

    // 使用 index 写入
    for (int i = 0; i < 10; ++i) {
        tensor.index<float>(i) = static_cast<float>(i * 2);
    }

    // 使用 index 读取
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(tensor.index<float>(i), static_cast<float>(i * 2));
    }
}

/**
 * @test 测试带索引的 ptr 访问
 */
TEST(TensorTest, PtrWithIndex) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp32, 10, true, allocator);

    float* data = tensor.ptr<float>();
    for (int i = 0; i < 10; ++i) {
        data[i] = static_cast<float>(i);
    }

    // 使用带索引的 ptr
    float* ptr_at_5 = tensor.ptr<float>(5);
    EXPECT_FLOAT_EQ(*ptr_at_5, 5.0f);
    EXPECT_FLOAT_EQ(ptr_at_5[1], 6.0f);
}

/**
 * @test 测试 const ptr
 */
TEST(TensorTest, ConstPtr) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp32, 10, true, allocator);

    float* data = tensor.ptr<float>();
    for (int i = 0; i < 10; ++i) {
        data[i] = static_cast<float>(i);
    }

    // 测试 const 版本
    const Tensor& const_tensor = tensor;
    const float* const_data = const_tensor.ptr<float>();

    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(const_data[i], static_cast<float>(i));
    }
}

// ==================== 2D Tensor 数据布局测试 ====================

/**
 * @test 测试 2D Tensor 行主序存储
 */
TEST(TensorTest, DataLayout2D) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp32, 2, 3, true, allocator);

    float* data = tensor.ptr<float>();

    // 填充数据
    // Row 0: [0, 1, 2]
    // Row 1: [10, 11, 12]
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            data[i * 3 + j] = static_cast<float>(i * 10 + j);
        }
    }

    // 验证数据
    EXPECT_FLOAT_EQ(data[0], 0.0f);   // [0,0]
    EXPECT_FLOAT_EQ(data[1], 1.0f);   // [0,1]
    EXPECT_FLOAT_EQ(data[2], 2.0f);   // [0,2]
    EXPECT_FLOAT_EQ(data[3], 10.0f);  // [1,0]
    EXPECT_FLOAT_EQ(data[4], 11.0f);  // [1,1]
    EXPECT_FLOAT_EQ(data[5], 12.0f);  // [1,2]
}

// ==================== Strides 测试 ====================

/**
 * @test 测试 1D Tensor strides
 */
TEST(TensorTest, Strides1D) {
    Tensor tensor(base::DataType::kTypeFp32, 10);
    auto strides = tensor.strides();

    ASSERT_EQ(strides.size(), 1);
    EXPECT_EQ(strides[0], 1);
}

/**
 * @test 测试 2D Tensor strides
 */
TEST(TensorTest, Strides2D) {
    Tensor tensor(base::DataType::kTypeFp32, 2, 3);
    auto strides = tensor.strides();

    ASSERT_EQ(strides.size(), 2);
    EXPECT_EQ(strides[0], 3);  // 第一维步长：第二维大小
    EXPECT_EQ(strides[1], 1);  // 最后一维步长：1
}

/**
 * @test 测试 3D Tensor strides
 */
TEST(TensorTest, Strides3D) {
    Tensor tensor(base::DataType::kTypeFp32, 2, 3, 4);
    auto strides = tensor.strides();

    ASSERT_EQ(strides.size(), 3);
    EXPECT_EQ(strides[0], 12);  // 3 * 4
    EXPECT_EQ(strides[1], 4);   // 4
    EXPECT_EQ(strides[2], 1);   // 1
}

/**
 * @test 测试 4D Tensor strides
 */
TEST(TensorTest, Strides4D) {
    Tensor tensor(base::DataType::kTypeFp32, 2, 3, 4, 5);
    auto strides = tensor.strides();

    ASSERT_EQ(strides.size(), 4);
    EXPECT_EQ(strides[0], 60);  // 3 * 4 * 5
    EXPECT_EQ(strides[1], 20);  // 4 * 5
    EXPECT_EQ(strides[2], 5);   // 5
    EXPECT_EQ(strides[3], 1);   // 1
}

// ==================== Reshape 测试 ====================

/**
 * @test 测试 reshape（无内存分配）
 */
TEST(TensorTest, ReshapeWithoutBuffer) {
    Tensor tensor(base::DataType::kTypeFp32, 12);

    EXPECT_EQ(tensor.size(), 12);
    EXPECT_EQ(tensor.dims_size(), 1);

    // Reshape to 3x4
    tensor.reshape({3, 4});

    EXPECT_EQ(tensor.size(), 12);
    EXPECT_EQ(tensor.dims_size(), 2);
    EXPECT_EQ(tensor.get_dim(0), 3);
    EXPECT_EQ(tensor.get_dim(1), 4);
}

/**
 * @test 测试 reshape（有内存分配）
 */
TEST(TensorTest, ReshapeWithBuffer) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp32, 12, true, allocator);

    // 写入初始数据
    float* data = tensor.ptr<float>();
    for (int i = 0; i < 12; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Reshape to 3x4（大小相同，不需要重新分配）
    tensor.reshape({3, 4});

    EXPECT_EQ(tensor.size(), 12);
    EXPECT_EQ(tensor.dims_size(), 2);

    // 验证数据保留
    data = tensor.ptr<float>();
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
    }
}

// ==================== Clone 测试 ====================

/**
 * @test 测试 Tensor 克隆
 */
TEST(TensorTest, Clone) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp32, 2, 3, true, allocator);

    // 填充数据
    float* data = tensor.ptr<float>();
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i * 10);
    }

    // 克隆
    Tensor cloned = tensor.clone();

    // 验证属性相同
    EXPECT_EQ(cloned.size(), tensor.size());
    EXPECT_EQ(cloned.dims_size(), tensor.dims_size());
    EXPECT_EQ(cloned.data_type(), tensor.data_type());
    EXPECT_NE(cloned.get_buffer(), tensor.get_buffer());  // 不同的 buffer

    // 验证数据相同
    float* cloned_data = cloned.ptr<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(cloned_data[i], data[i]);
    }

    // 验证独立性：修改克隆不影响原 tensor
    cloned_data[0] = 999.0f;
    EXPECT_FLOAT_EQ(data[0], 0.0f);  // 原数据不变
}

// ==================== Reset 测试 ====================

/**
 * @test 测试 reset 方法
 */
TEST(TensorTest, Reset) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp32, 10, true, allocator);

    EXPECT_NE(tensor.get_buffer(), nullptr);
    EXPECT_EQ(tensor.size(), 10);

    // Reset 为新的形状和类型
    tensor.reset(base::DataType::kTypeInt32, {5, 6});

    EXPECT_EQ(tensor.size(), 30);
    EXPECT_EQ(tensor.dims_size(), 2);
    EXPECT_EQ(tensor.data_type(), base::DataType::kTypeInt32);
    EXPECT_EQ(tensor.get_buffer(), nullptr);  // buffer 被清空
}

// ==================== Assign 测试 ====================

/**
 * @test 测试 assign buffer
 */
TEST(TensorTest, AssignBuffer) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();

    // 创建两个 tensor
    Tensor tensor1(base::DataType::kTypeFp32, 10, true, allocator);
    Tensor tensor2(base::DataType::kTypeFp32, 10);

    // 填充 tensor1 的数据
    float* data1 = tensor1.ptr<float>();
    for (int i = 0; i < 10; ++i) {
        data1[i] = static_cast<float>(i);
    }

    // 将 tensor1 的 buffer 赋给 tensor2
    bool success = tensor2.assign(tensor1.get_buffer());
    EXPECT_TRUE(success);

    // 验证 tensor2 现在指向同一个 buffer
    EXPECT_EQ(tensor2.get_buffer(), tensor1.get_buffer());

    // 验证可以通过 tensor2 访问数据
    float* data2 = tensor2.ptr<float>();
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(data2[i], static_cast<float>(i));
    }
}

/**
 * @test 测试 assign 不同类型的 buffer（应失败）
 */
TEST(TensorTest, AssignIncompatibleBuffer) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();

    Tensor tensor1(base::DataType::kTypeFp32, 10, true, allocator);
    Tensor tensor2(base::DataType::kTypeFp32, 20);  // 大小不匹配

    bool success = tensor2.assign(tensor1.get_buffer());
    EXPECT_FALSE(success);  // tensor2 需要更大的 buffer
}

// ==================== 边界情况测试 ====================

/**
 * @test 测试零维 Tensor
 */
TEST(TensorTest, ZeroDimTensor) {
    Tensor tensor(base::DataType::kTypeFp32, std::vector<int32_t>{});

    EXPECT_EQ(tensor.size(), 0);
    EXPECT_EQ(tensor.dims_size(), 0);
}

/**
 * @test 测试零大小 Tensor
 */
TEST(TensorTest, ZeroSizeTensor) {
    Tensor tensor(base::DataType::kTypeFp32, 0);

    EXPECT_EQ(tensor.size(), 0);
    EXPECT_EQ(tensor.dims_size(), 1);
    EXPECT_EQ(tensor.get_dim(0), 0);
}

/**
 * @test 测试大维度 Tensor（不分配内存）
 */
TEST(TensorTest, LargeDimTensor) {
    // 创建大维度但不分配内存
    Tensor tensor(base::DataType::kTypeFp32, 1000, 1000);

    EXPECT_EQ(tensor.size(), 1000000);
    EXPECT_EQ(tensor.dims_size(), 2);
    EXPECT_EQ(tensor.get_buffer(), nullptr);  // 未分配内存
}

/**
 * @test 测试访问空 Tensor
 */
TEST(TensorTest, AccessEmptyTensor) {
    Tensor tensor;

    EXPECT_EQ(tensor.ptr<float>(), nullptr);
    EXPECT_EQ(tensor.dims_size(), 0);
    EXPECT_EQ(tensor.size(), 0);
}

// ==================== 性能测试 ====================

/**
 * @test 测试大 Tensor 创建性能
 */
TEST(TensorTest, LargeTensorCreation) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    size_t size = 1000000;  // 1M 元素

    auto start = std::chrono::high_resolution_clock::now();

    Tensor tensor(base::DataType::kTypeFp32, size, true, allocator);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    EXPECT_NE(tensor.ptr<float>(), nullptr);
    EXPECT_EQ(tensor.size(), size);

    LOG(INFO) << "Allocated 1M float tensor in " << duration.count() << " microseconds";
}

/**
 * @test 测试数据填充性能
 */
TEST(TensorTest, DataFillPerformance) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    Tensor tensor(base::DataType::kTypeFp32, 1000000, true, allocator);

    auto start = std::chrono::high_resolution_clock::now();

    float* data = tensor.ptr<float>();
    for (size_t i = 0; i < tensor.size(); ++i) {
        data[i] = static_cast<float>(i);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    LOG(INFO) << "Filled 1M elements in " << duration.count() << " milliseconds";
}
