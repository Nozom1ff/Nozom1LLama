#include <gtest/gtest.h>
#include <op/swiglu.h>
#include <base/cuda_config.h>
#include <base/alloc.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <chrono>

using namespace op;
using namespace base;
using namespace tensor;

class SwiGLULayerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cuda_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();

        cuda_config_ = std::make_shared<kernel::CudaConfig>();
        cuda_config_->stream = nullptr;
    }

    void TearDown() override
    {
        cuda_config_.reset();
    }

    // 创建 CUDA tensor
    Tensor create_cuda_tensor(const std::vector<int32_t> &dims, const std::vector<float> &data)
    {
        CHECK_EQ(data.size(), std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>()));

        Tensor cpu_tensor(DataType::kTypeFp32, dims, true, cpu_alloc_);
        memcpy(cpu_tensor.ptr<float>(), data.data(), data.size() * sizeof(float));

        cpu_tensor.to_cuda();
        cudaDeviceSynchronize();
        return cpu_tensor;
    }

    // 创建空的 CUDA tensor
    Tensor create_empty_cuda_tensor(const std::vector<int32_t> &dims)
    {
        return Tensor(DataType::kTypeFp32, dims, true, cuda_alloc_);
    }

    // CPU SwiGLU 实现（用于验证）
    float cpu_sigmoid(float x)
    {
        // Clamp to avoid overflow
        x = fmaxf(fminf(x, 88.3762626647949f), -88.3762626647949f);
        return 1.0f / (1.0f + exp(-x));
    }

    void cpu_swiglu(const std::vector<float> &input1,
                   const std::vector<float> &input2,
                   std::vector<float> &output)
    {
        ASSERT_EQ(input1.size(), input2.size());
        output.resize(input1.size());

        for (size_t i = 0; i < input1.size(); ++i)
        {
            float sigmoid_val = cpu_sigmoid(input1[i]);
            output[i] = input1[i] * sigmoid_val * input2[i];
        }
    }

    // 验证结果
    void verify_result(const Tensor &output, const std::vector<float> &expected, float tolerance = 1e-5f)
    {
        Tensor cpu_output = output.clone();
        cpu_output.to_cpu();

        ASSERT_EQ(output.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_NEAR(cpu_output.ptr<float>()[i], expected[i], tolerance)
                << "Mismatch at index " << i;
        }
    }

    std::shared_ptr<base::DeviceAllocator> cuda_alloc_;
    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
};

// ==================== 基础功能测试 ====================

TEST_F(SwiGLULayerTest, BasicSwiGLU)
{
    // 测试基本的 SwiGLU: SiGLU(A) * B
    // A = [0.0, 1.0, 2.0], B = [1.0, 2.0, 3.0]
    // sigmoid(0) ≈ 0.5, sigmoid(1) ≈ 0.73, sigmoid(2) ≈ 0.88
    // expected ≈ [0*0.5*1, 1*0.73*2, 2*0.88*3]

    const int32_t hidden_dim = 3;
    std::vector<float> input1_data = {0.0f, 1.0f, 2.0f};
    std::vector<float> input2_data = {1.0f, 2.0f, 3.0f};

    Tensor input1 = create_cuda_tensor({hidden_dim}, input1_data);
    Tensor input2 = create_cuda_tensor({hidden_dim}, input2_data);
    Tensor output = create_empty_cuda_tensor({hidden_dim});

    SwiGLULayer layer(DeviceType::kCUDA, hidden_dim);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input1);
    layer.set_input(1, input2);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    std::vector<float> expected;
    cpu_swiglu(input1_data, input2_data, expected);
    verify_result(output, expected, 1e-4f);
}

TEST_F(SwiGLULayerTest, SingleElement)
{
    // 测试单个元素
    const int32_t hidden_dim = 1;
    std::vector<float> input1_data = {0.5f};
    std::vector<float> input2_data = {2.0f};

    Tensor input1 = create_cuda_tensor({hidden_dim}, input1_data);
    Tensor input2 = create_cuda_tensor({hidden_dim}, input2_data);
    Tensor output = create_empty_cuda_tensor({hidden_dim});

    SwiGLULayer layer(DeviceType::kCUDA, hidden_dim);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input1);
    layer.set_input(1, input2);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    std::vector<float> expected;
    cpu_swiglu(input1_data, input2_data, expected);
    verify_result(output, expected);
}

TEST_F(SwiGLULayerTest, LargeVector)
{
    // 测试大向量
    const int32_t hidden_dim = 10000;
    std::vector<float> input1_data(hidden_dim);
    std::vector<float> input2_data(hidden_dim);

    std::random_device rd;
    std::mt19937 gen(42); // 固定种子以获得可重复的结果
    std::uniform_real_distribution<float> dis(-5.0f, 5.0f);

    for (int i = 0; i < hidden_dim; ++i)
    {
        input1_data[i] = dis(gen);
        input2_data[i] = dis(gen);
    }

    Tensor input1 = create_cuda_tensor({hidden_dim}, input1_data);
    Tensor input2 = create_cuda_tensor({hidden_dim}, input2_data);
    Tensor output = create_empty_cuda_tensor({hidden_dim});

    SwiGLULayer layer(DeviceType::kCUDA, hidden_dim);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input1);
    layer.set_input(1, input2);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    std::vector<float> expected;
    cpu_swiglu(input1_data, input2_data, expected);
    verify_result(output, expected, 1e-4f);
}

TEST_F(SwiGLULayerTest, AllZeros)
{
    // 测试全零输入
    const int32_t hidden_dim = 5;
    std::vector<float> input1_data(5, 0.0f);
    std::vector<float> input2_data(5, 1.0f);

    Tensor input1 = create_cuda_tensor({hidden_dim}, input1_data);
    Tensor input2 = create_cuda_tensor({hidden_dim}, input2_data);
    Tensor output = create_empty_cuda_tensor({hidden_dim});

    SwiGLULayer layer(DeviceType::kCUDA, hidden_dim);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input1);
    layer.set_input(1, input2);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    // SiGLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    std::vector<float> expected(5, 0.0f);
    verify_result(output, expected, 1e-6f);
}

TEST_F(SwiGLULayerTest, NegativeValues)
{
    // 测试负值
    const int32_t hidden_dim = 3;
    std::vector<float> input1_data = {-1.0f, -2.0f, -0.5f};
    std::vector<float> input2_data = {1.0f, 1.0f, 1.0f};

    Tensor input1 = create_cuda_tensor({hidden_dim}, input1_data);
    Tensor input2 = create_cuda_tensor({hidden_dim}, input2_data);
    Tensor output = create_empty_cuda_tensor({hidden_dim});

    SwiGLULayer layer(DeviceType::kCUDA, hidden_dim);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input1);
    layer.set_input(1, input2);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    std::vector<float> expected;
    cpu_swiglu(input1_data, input2_data, expected);
    verify_result(output, expected, 1e-4f);
}

TEST_F(SwiGLULayerTest, ExtremeValues)
{
    // 测试极端值（接近 exp 溢出边界）
    const int32_t hidden_dim = 4;
    std::vector<float> input1_data = {88.0f, -88.0f, 0.0f, 50.0f};
    std::vector<float> input2_data = {1.0f, 1.0f, 1.0f, 1.0f};

    Tensor input1 = create_cuda_tensor({hidden_dim}, input1_data);
    Tensor input2 = create_cuda_tensor({hidden_dim}, input2_data);
    Tensor output = create_empty_cuda_tensor({hidden_dim});

    SwiGLULayer layer(DeviceType::kCUDA, hidden_dim);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input1);
    layer.set_input(1, input2);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    std::vector<float> expected;
    cpu_swiglu(input1_data, input2_data, expected);
    verify_result(output, expected, 1e-3f);
}

// ==================== 错误处理测试 ====================

TEST_F(SwiGLULayerTest, EmptyInput)
{
    // 测试空输入
    const int32_t hidden_dim = 3;

    Tensor input1; // 空 tensor
    Tensor input2;
    Tensor output = create_empty_cuda_tensor({hidden_dim});

    SwiGLULayer layer(DeviceType::kCUDA, hidden_dim);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input1);
    layer.set_input(1, input2);
    layer.set_output(0, output);

    Status status = layer.check();
    ASSERT_FALSE(status);
}

TEST_F(SwiGLULayerTest, WrongDeviceType)
{
    // 测试错误的设备类型
    const int32_t hidden_dim = 3;
    std::vector<float> input1_data(3, 1.0f);
    std::vector<float> input2_data(3, 1.0f);

    // 创建 CPU tensor
    Tensor input1(DataType::kTypeFp32, {hidden_dim}, true, cpu_alloc_);
    Tensor input2(DataType::kTypeFp32, {hidden_dim}, true, cpu_alloc_);
    memcpy(input1.ptr<float>(), input1_data.data(), input1_data.size() * sizeof(float));
    memcpy(input2.ptr<float>(), input2_data.data(), input2_data.size() * sizeof(float));

    Tensor output = create_empty_cuda_tensor({hidden_dim});

    SwiGLULayer layer(DeviceType::kCUDA, hidden_dim);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input1);
    layer.set_input(1, input2);
    layer.set_output(0, output);

    Status status = layer.check();
    ASSERT_FALSE(status);
}

TEST_F(SwiGLULayerTest, WrongShape)
{
    // 测试错误的形状
    const int32_t hidden_dim = 3;
    std::vector<float> input1_data = {1.0f, 2.0f, 3.0f};
    std::vector<float> input2_data = {1.0f, 2.0f}; // 错误的大小

    Tensor input1 = create_cuda_tensor({hidden_dim}, input1_data);
    Tensor input2 = create_cuda_tensor({2}, input2_data);
    Tensor output = create_empty_cuda_tensor({hidden_dim});

    SwiGLULayer layer(DeviceType::kCUDA, hidden_dim);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input1);
    layer.set_input(1, input2);
    layer.set_output(0, output);

    Status status = layer.check();
    ASSERT_FALSE(status);
}

TEST_F(SwiGLULayerTest, WrongOutputShape)
{
    // 测试错误的输出形状
    const int32_t hidden_dim = 3;
    std::vector<float> input1_data(3, 1.0f);
    std::vector<float> input2_data(3, 1.0f);

    Tensor input1 = create_cuda_tensor({hidden_dim}, input1_data);
    Tensor input2 = create_cuda_tensor({hidden_dim}, input2_data);
    Tensor output = create_empty_cuda_tensor({hidden_dim + 1}); // 错误的输出大小

    SwiGLULayer layer(DeviceType::kCUDA, hidden_dim);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input1);
    layer.set_input(1, input2);
    layer.set_output(0, output);

    Status status = layer.check();
    ASSERT_FALSE(status);
}

// ==================== API 测试 ====================

TEST_F(SwiGLULayerTest, GetSetInputOutput)
{
    // 测试 get/set API
    const int32_t hidden_dim = 5;
    std::vector<float> input1_data(5, 1.0f);
    std::vector<float> input2_data(5, 2.0f);

    SwiGLULayer layer(DeviceType::kCUDA, hidden_dim);
    layer.set_cuda_config(cuda_config_);

    ASSERT_EQ(layer.input_size(), 2);
    ASSERT_EQ(layer.output_size(), 1);

    Tensor input1 = create_cuda_tensor({hidden_dim}, input1_data);
    Tensor input2 = create_cuda_tensor({hidden_dim}, input2_data);
    Tensor output = create_empty_cuda_tensor({hidden_dim});

    layer.set_input(0, input1);
    layer.set_input(1, input2);
    layer.set_output(0, output);

    ASSERT_EQ(layer.get_input(0).size(), hidden_dim);
    ASSERT_EQ(layer.get_input(1).size(), hidden_dim);
    ASSERT_EQ(layer.get_output(0).size(), hidden_dim);
}

// ==================== 性能测试 ====================

TEST_F(SwiGLULayerTest, PerformanceTest)
{
    // 性能测试
    const std::vector<int32_t> hidden_dims = {1000, 10000, 100000, 1000000};

    std::cout << "\n=== SwiGLU Performance Test ===\n";

    for (int32_t hidden_dim : hidden_dims)
    {
        std::vector<float> input1_data(hidden_dim, 1.0f);
        std::vector<float> input2_data(hidden_dim, 2.0f);

        Tensor input1 = create_cuda_tensor({hidden_dim}, input1_data);
        Tensor input2 = create_cuda_tensor({hidden_dim}, input2_data);
        Tensor output = create_empty_cuda_tensor({hidden_dim});

        SwiGLULayer layer(DeviceType::kCUDA, hidden_dim);
        layer.set_cuda_config(cuda_config_);

        layer.set_input(0, input1);
        layer.set_input(1, input2);
        layer.set_output(0, output);

        // 预热
        layer.forward();

        // 测试 10 次
        const int iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            layer.forward();
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time = static_cast<double>(duration.count()) / iterations;

        std::cout << "Hidden dim: " << hidden_dim
                  << ", Avg time: " << avg_time << " us"
                  << ", Throughput: " << (hidden_dim / avg_time * 1e6) << " elements/s\n";
    }
}

// ==================== 主函数 ====================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
