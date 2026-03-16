#include <gtest/gtest.h>
#include <op/rmsnorm.h>
#include <base/cuda_config.h>
#include <base/alloc.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <chrono>
#include <numeric>

using namespace op;
using namespace base;
using namespace tensor;

class RmsNormLayerTest : public ::testing::Test
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

    // CPU RMSNorm 实现（用于验证）
    float cpu_rmsnorm(const std::vector<float> &input,
                      const std::vector<float> &weight,
                      std::vector<float> &output,
                      float eps = 1e-6f)
    {
        CHECK_EQ(input.size(), weight.size());
        output.resize(input.size());

        // 计算 mean(x^2)
        float sum_sq = 0.0f;
        for (size_t i = 0; i < input.size(); ++i)
        {
            sum_sq += input[i] * input[i];
        }
        float mean_sq = sum_sq / static_cast<float>(input.size());

        // 计算 RMS: sqrt(mean_sq + eps)
        float rms = sqrtf(mean_sq + eps);

        // RMSNorm(x) = x / RMS * weight
        for (size_t i = 0; i < input.size(); ++i)
        {
            output[i] = (input[i] / rms) * weight[i];
        }

        return rms;
    }

    // 多维 RMSNorm（沿着最后一维归一化）
    void cpu_rmsnorm_dim(const std::vector<float> &input,
                         const std::vector<float> &weight,
                         std::vector<float> &output,
                         const std::vector<int32_t> &dims,
                         float eps = 1e-6f)
    {
        size_t total_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
        int32_t last_dim = dims.back();
        int32_t num_rows = total_size / last_dim;

        CHECK_EQ(weight.size(), last_dim);
        output.resize(total_size);

        for (int32_t row = 0; row < num_rows; ++row)
        {
            // 提取当前行
            std::vector<float> row_input(last_dim);
            std::vector<float> row_output(last_dim);

            for (int32_t i = 0; i < last_dim; ++i)
            {
                row_input[i] = input[row * last_dim + i];
            }

            // 计算 RMSNorm
            cpu_rmsnorm(row_input, weight, row_output, eps);

            // 复制回输出
            for (int32_t i = 0; i < last_dim; ++i)
            {
                output[row * last_dim + i] = row_output[i];
            }
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

// ==================== 基础功能测试 (1D tensor) ====================

TEST_F(RmsNormLayerTest, BasicRmsNorm1D)
{
    // 测试基本的 RMSNorm: output = (input / RMS) * weight
    // input = [1.0, 2.0, 3.0, 4.0], weight = [0.1, 0.2, 0.3, 0.4]
    const int32_t dim = 4;
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> weight_data = {0.1f, 0.2f, 0.3f, 0.4f};

    Tensor input = create_cuda_tensor({dim}, input_data);
    Tensor weight = create_cuda_tensor({dim}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim});

    RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    std::vector<float> expected;
    cpu_rmsnorm(input_data, weight_data, expected);
    verify_result(output, expected, 1e-4f);
}

TEST_F(RmsNormLayerTest, SingleElement)
{
    // 测试单个元素
    const int32_t dim = 1;
    std::vector<float> input_data = {5.0f};
    std::vector<float> weight_data = {2.0f};

    Tensor input = create_cuda_tensor({dim}, input_data);
    Tensor weight = create_cuda_tensor({dim}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim});

    RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    std::vector<float> expected;
    cpu_rmsnorm(input_data, weight_data, expected);
    verify_result(output, expected);
}

TEST_F(RmsNormLayerTest, LargeVector1D)
{
    // 测试大向量
    const int32_t dim = 10000;
    std::vector<float> input_data(dim);
    std::vector<float> weight_data(dim);

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-5.0f, 5.0f);

    for (int i = 0; i < dim; ++i)
    {
        input_data[i] = dis(gen);
        weight_data[i] = dis(gen) * 0.1f + 1.0f; // weight 接近 1
    }

    Tensor input = create_cuda_tensor({dim}, input_data);
    Tensor weight = create_cuda_tensor({dim}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim});

    RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    std::vector<float> expected;
    cpu_rmsnorm(input_data, weight_data, expected);
    verify_result(output, expected, 1e-4f);
}

TEST_F(RmsNormLayerTest, AllZeros)
{
    // 测试全零输入
    const int32_t dim = 5;
    std::vector<float> input_data(5, 0.0f);
    std::vector<float> weight_data(5, 1.0f);

    Tensor input = create_cuda_tensor({dim}, input_data);
    Tensor weight = create_cuda_tensor({dim}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim});

    RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    // RMSNorm(0) = 0 / sqrt(0 + eps) * weight = 0
    std::vector<float> expected(5, 0.0f);
    verify_result(output, expected, 1e-6f);
}

TEST_F(RmsNormLayerTest, NegativeValues)
{
    // 测试负值
    const int32_t dim = 4;
    std::vector<float> input_data = {-1.0f, -2.0f, -3.0f, -4.0f};
    std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f};

    Tensor input = create_cuda_tensor({dim}, input_data);
    Tensor weight = create_cuda_tensor({dim}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim});

    RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    std::vector<float> expected;
    cpu_rmsnorm(input_data, weight_data, expected);
    verify_result(output, expected, 1e-4f);
}

// ==================== 多维张量测试 ====================

TEST_F(RmsNormLayerTest, BasicRmsNorm2D)
{
    // 测试 2D tensor (batch_size=2, dim=4)
    const int32_t dim = 4;
    std::vector<int32_t> input_dims = {2, dim};
    std::vector<float> input_data = {
        1.0f, 2.0f, 3.0f, 4.0f,   // row 0
        5.0f, 6.0f, 7.0f, 8.0f    // row 1
    };
    std::vector<float> weight_data = {0.1f, 0.2f, 0.3f, 0.4f};

    Tensor input = create_cuda_tensor(input_dims, input_data);
    Tensor weight = create_cuda_tensor({dim}, weight_data);
    Tensor output = create_empty_cuda_tensor(input_dims);

    RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    std::vector<float> expected;
    cpu_rmsnorm_dim(input_data, weight_data, expected, input_dims);
    verify_result(output, expected, 1e-4f);
}

TEST_F(RmsNormLayerTest, RmsNorm3D)
{
    // 测试 3D tensor (batch=2, seq=3, dim=4)
    const int32_t dim = 4;
    std::vector<int32_t> input_dims = {2, 3, dim};
    std::vector<float> input_data;
    std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f};

    // 生成数据
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.1f, 2.0f);

    int total_size = 2 * 3 * 4;
    for (int i = 0; i < total_size; ++i)
    {
        input_data.push_back(dis(gen));
    }

    Tensor input = create_cuda_tensor(input_dims, input_data);
    Tensor weight = create_cuda_tensor({dim}, weight_data);
    Tensor output = create_empty_cuda_tensor(input_dims);

    RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    std::vector<float> expected;
    cpu_rmsnorm_dim(input_data, weight_data, expected, input_dims);
    verify_result(output, expected, 1e-4f);
}

TEST_F(RmsNormLayerTest, Large2DBatch)
{
    // 测试大批量 2D tensor (batch=128, dim=256)
    const int32_t batch = 128;
    const int32_t dim = 256;
    std::vector<int32_t> input_dims = {batch, dim};
    std::vector<float> input_data(batch * dim);
    std::vector<float> weight_data(dim, 1.0f);

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < input_data.size(); ++i)
    {
        input_data[i] = dis(gen);
    }

    Tensor input = create_cuda_tensor(input_dims, input_data);
    Tensor weight = create_cuda_tensor({dim}, weight_data);
    Tensor output = create_empty_cuda_tensor(input_dims);

    RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    std::vector<float> expected;
    cpu_rmsnorm_dim(input_data, weight_data, expected, input_dims);
    verify_result(output, expected, 1e-4f);
}

// ==================== 错误处理测试 ====================

TEST_F(RmsNormLayerTest, EmptyInput)
{
    // 测试空输入
    const int32_t dim = 3;

    Tensor input; // 空 tensor
    Tensor weight = create_cuda_tensor({dim}, {1.0f, 1.0f, 1.0f});
    Tensor output = create_empty_cuda_tensor({dim});

    RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    Status status = layer.check();
    ASSERT_FALSE(status);
}

TEST_F(RmsNormLayerTest, WrongDeviceType)
{
    // 测试错误的设备类型（input）
    const int32_t dim = 3;
    std::vector<float> input_data(3, 1.0f);
    std::vector<float> weight_data(3, 1.0f);

    // 创建 CPU tensor for input
    Tensor input(DataType::kTypeFp32, {dim}, true, cpu_alloc_);
    memcpy(input.ptr<float>(), input_data.data(), input_data.size() * sizeof(float));

    Tensor weight = create_cuda_tensor({dim}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim});

    RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    // check() 会失败因为 input 是 CPU 而 layer 期望 CUDA
    Status status = layer.check();
    ASSERT_FALSE(status);
}

TEST_F(RmsNormLayerTest, WrongShape1D)
{
    // 测试错误的形状 (1D)
    const int32_t dim = 4;
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f}; // 错误的大小
    std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f};

    Tensor input = create_cuda_tensor({3}, input_data);
    Tensor weight = create_cuda_tensor({dim}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim});

    RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    Status status = layer.check();
    ASSERT_FALSE(status);
}

TEST_F(RmsNormLayerTest, WrongShape2D)
{
    // 测试错误的形状 (2D - 最后一维不匹配)
    const int32_t dim = 4;
    std::vector<float> input_data(8, 1.0f); // (2, 4)
    std::vector<float> weight_data(3, 1.0f); // 错误的 weight 维度

    Tensor input = create_cuda_tensor({2, 4}, input_data);
    Tensor weight = create_cuda_tensor({3}, weight_data);
    Tensor output = create_empty_cuda_tensor({2, 4});

    RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    Status status = layer.check();
    ASSERT_FALSE(status);
}

TEST_F(RmsNormLayerTest, WrongLastDim2D)
{
    // 测试 2D tensor 最后一维不匹配
    const int32_t dim = 4;
    std::vector<float> input_data(6, 1.0f); // (2, 3) - 最后一维是 3
    std::vector<float> weight_data(4, 1.0f);

    Tensor input = create_cuda_tensor({2, 3}, input_data);
    Tensor weight = create_cuda_tensor({dim}, weight_data);
    Tensor output = create_empty_cuda_tensor({2, 3});

    RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    Status status = layer.check();
    ASSERT_FALSE(status);
}

// ==================== API 测试 ====================

TEST_F(RmsNormLayerTest, GetSetInputOutputWeight)
{
    // 测试 get/set API
    const int32_t dim = 5;
    std::vector<float> input_data(5, 1.0f);
    std::vector<float> weight_data(5, 2.0f);

    RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    ASSERT_EQ(layer.input_size(), 1);
    ASSERT_EQ(layer.output_size(), 1);
    ASSERT_EQ(layer.weight_size(), 1);

    Tensor input = create_cuda_tensor({dim}, input_data);
    Tensor weight = create_cuda_tensor({dim}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim});

    layer.set_input(0, input);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    ASSERT_EQ(layer.get_input(0).size(), dim);
    ASSERT_EQ(layer.get_weight(0).size(), dim);
    ASSERT_EQ(layer.get_output(0).size(), dim);
}

// ==================== 性能测试 ====================

TEST_F(RmsNormLayerTest, PerformanceTest1D)
{
    // 1D tensor 性能测试
    const std::vector<int32_t> dims = {1000, 10000, 100000, 1000000};

    std::cout << "\n=== RMSNorm 1D Performance Test ===\n";

    for (int32_t dim : dims)
    {
        std::vector<float> input_data(dim, 1.0f);
        std::vector<float> weight_data(dim, 1.0f);

        Tensor input = create_cuda_tensor({dim}, input_data);
        Tensor weight = create_cuda_tensor({dim}, weight_data);
        Tensor output = create_empty_cuda_tensor({dim});

        RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
        layer.set_cuda_config(cuda_config_);

        layer.set_input(0, input);
        layer.set_weight(0, weight);
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

        std::cout << "Dim: " << dim
                  << ", Avg time: " << avg_time << " us"
                  << ", Throughput: " << (dim / avg_time * 1e6) << " elements/s\n";
    }
}

TEST_F(RmsNormLayerTest, PerformanceTest2D)
{
    // 2D tensor 性能测试
    const std::vector<std::pair<int32_t, int32_t>> shapes = {
        {32, 128},
        {64, 256},
        {128, 512},
        {256, 1024}
    };

    std::cout << "\n=== RMSNorm 2D Performance Test ===\n";

    for (auto [batch, dim] : shapes)
    {
        std::vector<float> input_data(batch * dim, 1.0f);
        std::vector<float> weight_data(dim, 1.0f);

        Tensor input = create_cuda_tensor({batch, dim}, input_data);
        Tensor weight = create_cuda_tensor({dim}, weight_data);
        Tensor output = create_empty_cuda_tensor({batch, dim});

        RmsNormLayer layer(DeviceType::kCUDA, dim, DataType::kTypeFp32);
        layer.set_cuda_config(cuda_config_);

        layer.set_input(0, input);
        layer.set_weight(0, weight);
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

        std::cout << "Shape: (" << batch << ", " << dim << ")"
                  << ", Avg time: " << avg_time << " us"
                  << ", Throughput: " << (batch * dim / avg_time * 1e6) << " elements/s\n";
    }
}

// ==================== 主函数 ====================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
