#include <gtest/gtest.h>
#include <op/add.h>
#include <base/cuda_config.h>
#include <base/alloc.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>

using namespace op;
using namespace base;
using namespace tensor;

class VecAddLayerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 初始化 CUDA allocator
        cuda_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();

        // 创建 CUDA config
        cuda_config_ = std::make_shared<kernel::CudaConfig>();
        cuda_config_->stream = nullptr; // 使用默认 stream

        // 创建 layer
        layer_ = std::make_unique<VecAddLayer>(DeviceType::kCUDA, DataType::kTypeFp32);
        layer_->set_cuda_config(cuda_config_);
    }

    void TearDown() override
    {
        layer_.reset();
        cuda_config_.reset();
    }

    // 创建 CUDA tensor（带随机数据）
    Tensor create_cuda_tensor(const std::vector<int32_t> &dims, const std::vector<float> &data)
    {
        CHECK_EQ(data.size(), std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>()));

        // 1. 先创建 CPU tensor
        auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
        Tensor cpu_tensor(DataType::kTypeFp32, dims, true, cpu_alloc);

        // 拷贝数据
        memcpy(cpu_tensor.ptr<float>(), data.data(), data.size() * sizeof(float));

        // 2. 转移到 CUDA
        cpu_tensor.to_cuda();
        // 同步确保拷贝完成
        cudaDeviceSynchronize();
        return cpu_tensor;
    }

    // 创建空的 CUDA tensor
    Tensor create_empty_cuda_tensor(const std::vector<int32_t> &dims)
    {
        return Tensor(DataType::kTypeFp32, dims, true, cuda_alloc_);
    }

    // 验证结果
    void verify_result(const Tensor &output, const std::vector<float> &expected)
    {
        // 拷贝回 CPU
        Tensor cpu_output = output.clone();
        cpu_output.to_cpu();

        // 验证
        ASSERT_EQ(output.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_FLOAT_EQ(cpu_output.ptr<float>()[i], expected[i])
                << "Mismatch at index " << i;
        }
    }

    std::shared_ptr<base::DeviceAllocator> cuda_alloc_;
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
    std::unique_ptr<VecAddLayer> layer_;
};

// ==================== 基础功能测试 ====================

TEST_F(VecAddLayerTest, BasicAddition)
{
    // 测试简单加法：[1, 2, 3] + [4, 5, 6] = [5, 7, 9]
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> data2 = {4.0f, 5.0f, 6.0f};
    std::vector<float> expected = {5.0f, 7.0f, 9.0f};

    Tensor input1 = create_cuda_tensor({3}, data1);
    Tensor input2 = create_cuda_tensor({3}, data2);
    Tensor output = create_empty_cuda_tensor({3});

    layer_->set_input(0, input1);
    layer_->set_input(1, input2);
    layer_->set_output(0, output);

    Status status = layer_->forward();
    ASSERT_TRUE(status);

    // CUDA 同步
    cudaDeviceSynchronize();

    // 验证
    verify_result(output, expected);
}

TEST_F(VecAddLayerTest, LargeTensorAddition)
{
    // 测试大规模 tensor 加法
    const int32_t size = 1000000;
    std::vector<float> data1(size, 1.0f);
    std::vector<float> data2(size, 2.0f);
    std::vector<float> expected(size, 3.0f);

    Tensor input1 = create_cuda_tensor({size}, data1);
    Tensor input2 = create_cuda_tensor({size}, data2);
    Tensor output = create_empty_cuda_tensor({size});

    layer_->set_input(0, input1);
    layer_->set_input(1, input2);
    layer_->set_output(0, output);

    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    Status status = layer_->forward();
    auto end = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(status);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Large tensor addition (1M elements): " << duration.count() << " us\n";

    // 验证部分结果
    verify_result(output, expected);
}

TEST_F(VecAddLayerTest, NegativeNumbers)
{
    // 测试负数：[-1, -2, 3] + [4, -5, 6] = [3, -7, 9]
    std::vector<float> data1 = {-1.0f, -2.0f, 3.0f};
    std::vector<float> data2 = {4.0f, -5.0f, 6.0f};
    std::vector<float> expected = {3.0f, -7.0f, 9.0f};

    Tensor input1 = create_cuda_tensor({3}, data1);
    Tensor input2 = create_cuda_tensor({3}, data2);
    Tensor output = create_empty_cuda_tensor({3});

    layer_->set_input(0, input1);
    layer_->set_input(1, input2);
    layer_->set_output(0, output);

    Status status = layer_->forward();
    ASSERT_TRUE(status);

    verify_result(output, expected);
}

TEST_F(VecAddLayerTest, FloatingPointPrecision)
{
    // 测试浮点精度
    std::vector<float> data1 = {0.1f, 0.2f, 0.3f};
    std::vector<float> data2 = {0.4f, 0.5f, 0.6f};
    std::vector<float> expected = {0.5f, 0.7f, 0.9f};

    Tensor input1 = create_cuda_tensor({3}, data1);
    Tensor input2 = create_cuda_tensor({3}, data2);
    Tensor output = create_empty_cuda_tensor({3});

    layer_->set_input(0, input1);
    layer_->set_input(1, input2);
    layer_->set_output(0, output);

    Status status = layer_->forward();
    ASSERT_TRUE(status);

    // 允许一定的浮点误差
    Tensor cpu_output = output.clone();
    cpu_output.to_cpu();
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(cpu_output.ptr<float>()[i], expected[i], 1e-6f)
            << "Mismatch at index " << i;
    }
}

// ==================== 错误处理测试 ====================

TEST_F(VecAddLayerTest, EmptyTensor)
{
    // 测试空 tensor
    std::vector<float> data1 = {1.0f, 2.0f};
    std::vector<float> data2 = {3.0f, 4.0f};

    Tensor input1 = create_cuda_tensor({2}, data1);
    Tensor input2 = create_cuda_tensor({2}, data2);
    Tensor output; // 空 tensor

    layer_->set_input(0, input1);
    layer_->set_input(1, input2);
    layer_->set_output(0, output);

    Status status = layer_->forward();
    ASSERT_FALSE(status);
}

TEST_F(VecAddLayerTest, SizeMismatch)
{
    // 测试大小不匹配
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> data2 = {4.0f, 5.0f}; // 大小不同

    Tensor input1 = create_cuda_tensor({3}, data1);
    Tensor input2 = create_cuda_tensor({2}, data2);
    Tensor output = create_empty_cuda_tensor({3});

    layer_->set_input(0, input1);
    layer_->set_input(1, input2);
    layer_->set_output(0, output);

    // check() 会失败
    Status status = layer_->check();
    ASSERT_FALSE(status);
}

TEST_F(VecAddLayerTest, WrongDeviceType)
{
    // 测试错误的设备类型
    auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();

    std::vector<float> data1 = {1.0f, 2.0f};
    std::vector<float> data2 = {3.0f, 4.0f};

    // 创建 CPU tensor（错误的设备类型）
    Tensor input1(DataType::kTypeFp32, {2}, true, cpu_alloc);
    Tensor input2(DataType::kTypeFp32, {2}, true, cpu_alloc);

    memcpy(input1.ptr<float>(), data1.data(), data1.size() * sizeof(float));
    memcpy(input2.ptr<float>(), data2.data(), data2.size() * sizeof(float));

    Tensor output = create_empty_cuda_tensor({2});

    layer_->set_input(0, input1);
    layer_->set_input(1, input2);
    layer_->set_output(0, output);

    // check() 会失败
    Status status = layer_->check();
    ASSERT_FALSE(status);
}

TEST_F(VecAddLayerTest, WrongDataType)
{
    // 测试错误的数据类型（如果添加了其他类型支持）
    // 目前只有 fp32，这个测试为未来扩展准备

    std::vector<float> data1 = {1.0f, 2.0f};
    std::vector<float> data2 = {3.0f, 4.0f};

    Tensor input1 = create_cuda_tensor({2}, data1);
    Tensor input2 = create_cuda_tensor({2}, data2);
    Tensor output = create_empty_cuda_tensor({2});

    layer_->set_input(0, input1);
    layer_->set_input(1, input2);
    layer_->set_output(0, output);

    Status status = layer_->forward();
    ASSERT_TRUE(status); // fp32 应该成功
}

// ==================== API 测试 ====================

TEST_F(VecAddLayerTest, VectorInputAPI)
{
    // 测试 vector 输入 API（通过 Layer 基类）
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> data2 = {4.0f, 5.0f, 6.0f};
    std::vector<float> expected = {5.0f, 7.0f, 9.0f};

    Tensor input1 = create_cuda_tensor({3}, data1);
    Tensor input2 = create_cuda_tensor({3}, data2);
    Tensor output = create_empty_cuda_tensor({3});

    // 使用 vector API（Layer 基类提供）
    std::vector<Tensor> inputs = {input1, input2};
    std::vector<Tensor> outputs = {output};

    Layer *base_layer = layer_.get();
    Status status = base_layer->forward(inputs, outputs);
    ASSERT_TRUE(status);

    verify_result(output, expected);
}

TEST_F(VecAddLayerTest, BatchSetInputOutput)
{
    // 测试批量设置输入输出
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> data2 = {4.0f, 5.0f, 6.0f};
    std::vector<float> expected = {5.0f, 7.0f, 9.0f};

    Tensor input1 = create_cuda_tensor({3}, data1);
    Tensor input2 = create_cuda_tensor({3}, data2);
    Tensor output = create_empty_cuda_tensor({3});

    // 使用批量设置 API
    std::vector<Tensor> inputs = {input1, input2};
    std::vector<Tensor> outputs = {output};

    layer_->set_input(inputs);
    layer_->set_output(outputs);

    Status status = layer_->forward();
    ASSERT_TRUE(status);

    verify_result(output, expected);
}

TEST_F(VecAddLayerTest, GetSetInputOutput)
{
    // 测试 get/set API
    std::vector<float> data1 = {1.0f, 2.0f};
    std::vector<float> data2 = {3.0f, 4.0f};

    Tensor input1 = create_cuda_tensor({2}, data1);
    Tensor input2 = create_cuda_tensor({2}, data2);
    Tensor output = create_empty_cuda_tensor({2});

    // 设置单个输入
    layer_->set_input(0, input1);
    layer_->set_input(1, input2);

    // 获取输入
    ASSERT_EQ(layer_->input_size(), 2);
    ASSERT_EQ(layer_->output_size(), 1);

    // 验证获取的 tensor
    Tensor &get_input1 = layer_->get_input(0);
    Tensor &get_input2 = layer_->get_input(1);
    ASSERT_EQ(get_input1.size(), 2);
    ASSERT_EQ(get_input2.size(), 2);
}

// ==================== 性能测试 ====================

TEST_F(VecAddLayerTest, PerformanceTest)
{
    // 性能测试
    const std::vector<int32_t> sizes = {1000, 10000, 100000, 1000000};

    std::cout << "\n=== Performance Test ===\n";

    for (int32_t size : sizes)
    {
        std::vector<float> data1(size, 1.0f);
        std::vector<float> data2(size, 2.0f);

        Tensor input1 = create_cuda_tensor({size}, data1);
        Tensor input2 = create_cuda_tensor({size}, data2);
        Tensor output = create_empty_cuda_tensor({size});

        layer_->set_input(0, input1);
        layer_->set_input(1, input2);
        layer_->set_output(0, output);

        // 预热
        layer_->forward();

        // 测试 10 次
        const int iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            layer_->forward();
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time = static_cast<double>(duration.count()) / iterations;

        std::cout << "Size: " << size
                  << ", Avg time: " << avg_time << " us"
                  << ", Throughput: " << (size / avg_time * 1e6) << " elements/s\n";
    }
}

// ==================== 主函数 ====================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
