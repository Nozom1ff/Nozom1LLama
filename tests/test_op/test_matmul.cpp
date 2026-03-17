#include <gtest/gtest.h>
#include <op/matmul.h>
#include <base/cuda_config.h>
#include <base/alloc.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <cmath>

using namespace op;
using namespace base;
using namespace tensor;

class MatmulLayerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 初始化 CUDA allocator
        cuda_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();

        // 创建 CUDA config
        cuda_config_ = std::make_shared<kernel::CudaConfig>();
        cuda_config_->stream = nullptr; // 使用默认 stream
    }

    void TearDown() override
    {
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

    // 验证结果（允许一定误差）
    void verify_result(const Tensor &output, const std::vector<float> &expected, float epsilon = 1e-3f)
    {
        // 拷贝回 CPU
        Tensor cpu_output = output.clone();
        cpu_output.to_cpu();

        // 验证
        ASSERT_EQ(output.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_NEAR(cpu_output.ptr<float>()[i], expected[i], epsilon)
                << "Mismatch at index " << i;
        }
    }

    // 计算预期的 matmul 结果
    std::vector<float> compute_expected(const std::vector<float> &input,
                                        const std::vector<float> &weight,
                                        int32_t dim0, int32_t dim1,
                                        const std::vector<float> *bias = nullptr)
    {
        std::vector<float> output(dim0, 0.0f);
        for (int i = 0; i < dim0; ++i)
        {
            for (int j = 0; j < dim1; ++j)
            {
                output[i] += input[j] * weight[i * dim1 + j];
            }
            if (bias)
            {
                output[i] += (*bias)[i];
            }
        }
        return output;
    }

    std::shared_ptr<base::DeviceAllocator> cuda_alloc_;
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
};

// ==================== 基础功能测试 ====================

TEST_F(MatmulLayerTest, BasicMatmul)
{
    // 测试基本的矩阵乘法
    // input: [1, 2] (dim1=2)
    // weight: [[1, 2], [3, 4]] (dim0=2, dim1=2)
    // expected: [1*1+2*2, 1*3+2*4] = [5, 11]
    const int32_t dim0 = 2;
    const int32_t dim1 = 2;

    std::vector<float> input_data = {1.0f, 2.0f};
    std::vector<float> weight_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> expected = {5.0f, 11.0f};

    auto layer = std::make_unique<MatmulLayer>(DeviceType::kCUDA, dim0, dim1, false, false, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);

    Tensor input = create_cuda_tensor({dim1}, input_data);
    Tensor weight = create_cuda_tensor({dim0, dim1}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim0});

    layer->set_input(0, input);
    layer->set_weight(0, weight);
    layer->set_output(0, output);

    Status status = layer->forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    verify_result(output, expected);
}

TEST_F(MatmulLayerTest, MatmulWithBias)
{
    // 测试带 bias 的矩阵乘法
    const int32_t dim0 = 2;
    const int32_t dim1 = 3;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
    std::vector<float> weight_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> bias_data = {10.0f, 20.0f};

    // 手动计算预期结果
    std::vector<float> expected = compute_expected(input_data, weight_data, dim0, dim1, &bias_data);

    auto layer = std::make_unique<MatmulLayer>(DeviceType::kCUDA, dim0, dim1, false, true, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);

    Tensor input = create_cuda_tensor({dim1}, input_data);
    Tensor weight = create_cuda_tensor({dim0, dim1}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim0});

    layer->set_input(0, input);
    layer->set_weight(0, weight);
    layer->set_output(0, output);

    // 设置 bias - 先用 CPU 内存，然后 layer 会处理转移到 CUDA
    int32_t bias_dim = dim0;
    layer->set_bias(0, bias_dim, bias_data.data(), DeviceType::kUnknown);

    // 将 layer 转移到 CUDA（包括 bias）
    layer->to_cuda();

    Status status = layer->forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    verify_result(output, expected);
}

TEST_F(MatmulLayerTest, LargerMatmul)
{
    // 测试较大的矩阵乘法
    const int32_t dim0 = 128;
    const int32_t dim1 = 256;

    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::vector<float> input_data(dim1);
    std::vector<float> weight_data(dim0 * dim1);

    for (int i = 0; i < dim1; ++i)
    {
        input_data[i] = dis(gen);
    }
    for (int i = 0; i < dim0 * dim1; ++i)
    {
        weight_data[i] = dis(gen);
    }

    std::vector<float> expected = compute_expected(input_data, weight_data, dim0, dim1);

    auto layer = std::make_unique<MatmulLayer>(DeviceType::kCUDA, dim0, dim1, false, false, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);

    Tensor input = create_cuda_tensor({dim1}, input_data);
    Tensor weight = create_cuda_tensor({dim0, dim1}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim0});

    layer->set_input(0, input);
    layer->set_weight(0, weight);
    layer->set_output(0, output);

    Status status = layer->forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    verify_result(output, expected, 1e-2f); // 较大的误差容限
}

TEST_F(MatmulLayerTest, ZeroInput)
{
    // 测试零输入
    const int32_t dim0 = 4;
    const int32_t dim1 = 3;

    std::vector<float> input_data(dim1, 0.0f);
    std::vector<float> weight_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> expected(dim0, 0.0f);

    auto layer = std::make_unique<MatmulLayer>(DeviceType::kCUDA, dim0, dim1, false, false, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);

    Tensor input = create_cuda_tensor({dim1}, input_data);
    Tensor weight = create_cuda_tensor({dim0, dim1}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim0});

    layer->set_input(0, input);
    layer->set_weight(0, weight);
    layer->set_output(0, output);

    Status status = layer->forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    verify_result(output, expected);
}

TEST_F(MatmulLayerTest, ZeroWeight)
{
    // 测试零权重
    const int32_t dim0 = 3;
    const int32_t dim1 = 4;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> weight_data(dim0 * dim1, 0.0f);
    std::vector<float> expected(dim0, 0.0f);

    auto layer = std::make_unique<MatmulLayer>(DeviceType::kCUDA, dim0, dim1, false, false, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);

    Tensor input = create_cuda_tensor({dim1}, input_data);
    Tensor weight = create_cuda_tensor({dim0, dim1}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim0});

    layer->set_input(0, input);
    layer->set_weight(0, weight);
    layer->set_output(0, output);

    Status status = layer->forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    verify_result(output, expected);
}

TEST_F(MatmulLayerTest, IdentityMatrix)
{
    // 测试单位矩阵
    const int32_t dim0 = 4;
    const int32_t dim1 = 4;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> weight_data = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    std::vector<float> expected = input_data; // 单位矩阵不改变输入

    auto layer = std::make_unique<MatmulLayer>(DeviceType::kCUDA, dim0, dim1, false, false, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);

    Tensor input = create_cuda_tensor({dim1}, input_data);
    Tensor weight = create_cuda_tensor({dim0, dim1}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim0});

    layer->set_input(0, input);
    layer->set_weight(0, weight);
    layer->set_output(0, output);

    Status status = layer->forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    verify_result(output, expected);
}

// ==================== 错误处理测试 ====================

TEST_F(MatmulLayerTest, WrongInputDim)
{
    // 测试错误的输入维度
    const int32_t dim0 = 3;
    const int32_t dim1 = 4;

    std::vector<float> input_data(5, 1.0f); // 错误的维度
    std::vector<float> weight_data(dim0 * dim1, 1.0f);

    auto layer = std::make_unique<MatmulLayer>(DeviceType::kCUDA, dim0, dim1, false, false, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);

    Tensor input = create_cuda_tensor({5}, input_data); // 错误维度
    Tensor weight = create_cuda_tensor({dim0, dim1}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim0});

    layer->set_input(0, input);
    layer->set_weight(0, weight);
    layer->set_output(0, output);

    Status status = layer->check();
    ASSERT_FALSE(status);
}

TEST_F(MatmulLayerTest, WrongWeightDim)
{
    // 测试错误的权重维度
    const int32_t dim0 = 3;
    const int32_t dim1 = 4;

    std::vector<float> input_data(dim1, 1.0f);
    std::vector<float> weight_data(dim0 * 5, 1.0f); // 错误的维度

    auto layer = std::make_unique<MatmulLayer>(DeviceType::kCUDA, dim0, dim1, false, false, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);

    Tensor input = create_cuda_tensor({dim1}, input_data);
    Tensor weight = create_cuda_tensor({dim0, 5}, weight_data); // 错误维度
    Tensor output = create_empty_cuda_tensor({dim0});

    layer->set_input(0, input);
    layer->set_weight(0, weight);
    layer->set_output(0, output);

    Status status = layer->check();
    ASSERT_FALSE(status);
}

TEST_F(MatmulLayerTest, WrongOutputDim)
{
    // 测试错误的输出维度
    const int32_t dim0 = 3;
    const int32_t dim1 = 4;

    std::vector<float> input_data(dim1, 1.0f);
    std::vector<float> weight_data(dim0 * dim1, 1.0f);

    auto layer = std::make_unique<MatmulLayer>(DeviceType::kCUDA, dim0, dim1, false, false, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);

    Tensor input = create_cuda_tensor({dim1}, input_data);
    Tensor weight = create_cuda_tensor({dim0, dim1}, weight_data);
    Tensor output = create_empty_cuda_tensor({5}); // 错误维度

    layer->set_input(0, input);
    layer->set_weight(0, weight);
    layer->set_output(0, output);

    Status status = layer->check();
    ASSERT_FALSE(status);
}

// ==================== 性能测试 ====================

TEST_F(MatmulLayerTest, PerformanceTest)
{
    // 性能测试
    const std::vector<std::pair<int32_t, int32_t>> sizes = {
        {64, 64},
        {128, 128},
        {256, 256},
        {512, 512},
        {1024, 1024}
    };

    std::cout << "\n=== Performance Test ===\n";

    for (auto [dim0, dim1] : sizes)
    {
        std::vector<float> input_data(dim1, 1.0f);
        std::vector<float> weight_data(dim0 * dim1, 1.0f);

        auto layer = std::make_unique<MatmulLayer>(DeviceType::kCUDA, dim0, dim1, false, false, DataType::kTypeFp32);
        layer->set_cuda_config(cuda_config_);

        Tensor input = create_cuda_tensor({dim1}, input_data);
        Tensor weight = create_cuda_tensor({dim0, dim1}, weight_data);
        Tensor output = create_empty_cuda_tensor({dim0});

        layer->set_input(0, input);
        layer->set_weight(0, weight);
        layer->set_output(0, output);

        // 预热
        layer->forward();
        cudaDeviceSynchronize();

        // 测试 100 次
        const int iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            layer->forward();
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time = static_cast<double>(duration.count()) / iterations;

        int64_t flops = 2LL * dim0 * dim1; // 乘加操作数
        double gflops = (flops / avg_time) / 1000.0;

        std::cout << "Dim0: " << dim0 << ", Dim1: " << dim1
                  << ", Avg time: " << avg_time << " us"
                  << ", GFLOPS: " << gflops << "\n";
    }
}

TEST_F(MatmulLayerTest, LargeScaleMatmul)
{
    // 大规模矩阵乘法测试
    const int32_t dim0 = 4096;
    const int32_t dim1 = 4096;

    std::vector<float> input_data(dim1, 1.0f);
    std::vector<float> weight_data(dim0 * dim1, 1.0f);

    // 计算预期结果（每个输出元素都是 dim1）
    std::vector<float> expected(dim0, static_cast<float>(dim1));

    auto layer = std::make_unique<MatmulLayer>(DeviceType::kCUDA, dim0, dim1, false, false, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);

    Tensor input = create_cuda_tensor({dim1}, input_data);
    Tensor weight = create_cuda_tensor({dim0, dim1}, weight_data);
    Tensor output = create_empty_cuda_tensor({dim0});

    layer->set_input(0, input);
    layer->set_weight(0, weight);
    layer->set_output(0, output);

    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    Status status = layer->forward();
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(status);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Large scale matmul (4096x4096): " << duration.count() << " ms\n";

    // 验证部分结果
    Tensor cpu_output = output.clone();
    cpu_output.to_cpu();

    for (int i = 0; i < std::min(10, dim0); ++i)
    {
        EXPECT_NEAR(cpu_output.ptr<float>()[i], expected[i], 1e-2f)
            << "Mismatch at index " << i;
    }
}

// ==================== 主函数 ====================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
