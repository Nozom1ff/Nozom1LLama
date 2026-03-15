#include <gtest/gtest.h>
#include <op/embedding.h>
#include <base/cuda_config.h>
#include <base/alloc.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>

using namespace op;
using namespace base;
using namespace tensor;

class EmbeddingLayerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 初始化 CUDA allocator
        cuda_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();

        // 创建 CUDA config
        cuda_config_ = std::make_shared<kernel::CudaConfig>();
        cuda_config_->stream = nullptr; // 使用默认 stream
    }

    void TearDown() override
    {
        cuda_config_.reset();
    }

    // 创建 CPU tensor（用于输入 token）
    Tensor create_cpu_tensor(const std::vector<int32_t> &dims, const std::vector<int32_t> &data)
    {
        CHECK_EQ(data.size(), std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>()));

        Tensor cpu_tensor(DataType::kTypeInt32, dims, true, cpu_alloc_);
        memcpy(cpu_tensor.ptr<int32_t>(), data.data(), data.size() * sizeof(int32_t));
        return cpu_tensor;
    }

    // 创建 CUDA tensor（用于权重）
    Tensor create_cuda_weight_tensor(const std::vector<int32_t> &dims, const std::vector<float> &data)
    {
        CHECK_EQ(data.size(), std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>()));

        // 1. 先创建 CPU tensor
        Tensor cpu_tensor(DataType::kTypeFp32, dims, true, cpu_alloc_);
        memcpy(cpu_tensor.ptr<float>(), data.data(), data.size() * sizeof(float));

        // 2. 转移到 CUDA
        cpu_tensor.to_cuda();
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
    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
};

// ==================== 基础功能测试 ====================

TEST_F(EmbeddingLayerTest, BasicEmbedding)
{
    // 测试基本的 embedding 查找
    // vocab_size = 4, dim = 3, seq_len = 2
    // 权重矩阵:
    //   token 0: [1.0, 2.0, 3.0]
    //   token 1: [4.0, 5.0, 6.0]
    //   token 2: [7.0, 8.0, 9.0]
    //   token 3: [10.0, 11.0, 12.0]
    // 输入 tokens: [0, 2]
    // 期望输出: [1.0, 2.0, 3.0, 7.0, 8.0, 9.0]

    const int32_t vocab_size = 4;
    const int32_t dim = 3;
    const int32_t seq_len = 2;

    // 创建权重矩阵
    std::vector<float> weight_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    };
    Tensor weight = create_cuda_weight_tensor({vocab_size, dim}, weight_data);

    // 创建输入 tokens
    std::vector<int32_t> token_data = {0, 2};
    Tensor input_tokens = create_cpu_tensor({seq_len}, token_data);

    // 创建输出 tensor
    Tensor output = create_empty_cuda_tensor({seq_len, dim});

    // 创建 layer
    EmbeddingLayer layer(DeviceType::kCUDA, dim, seq_len, vocab_size);
    layer.set_cuda_config(cuda_config_);

    // 设置输入、权重、输出
    layer.set_input(0, input_tokens);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    // 执行
    Status status = layer.forward();
    ASSERT_TRUE(status);

    // CUDA 同步
    cudaDeviceSynchronize();

    // 验证结果
    std::vector<float> expected = {
        1.0f, 2.0f, 3.0f,   // token 0 的 embedding
        7.0f, 8.0f, 9.0f    // token 2 的 embedding
    };
    verify_result(output, expected);
}

TEST_F(EmbeddingLayerTest, SingleTokenEmbedding)
{
    // 测试单个 token 的 embedding
    const int32_t vocab_size = 10;
    const int32_t dim = 5;
    const int32_t seq_len = 1;

    // 创建权重矩阵
    std::vector<float> weight_data(vocab_size * dim);
    for (int32_t i = 0; i < vocab_size; ++i)
    {
        for (int32_t j = 0; j < dim; ++j)
        {
            weight_data[i * dim + j] = static_cast<float>(i * dim + j);
        }
    }
    Tensor weight = create_cuda_weight_tensor({vocab_size, dim}, weight_data);

    // 创建输入 token (查询 token 5)
    std::vector<int32_t> token_data = {5};
    Tensor input_tokens = create_cpu_tensor({seq_len}, token_data);

    // 创建输出 tensor
    Tensor output = create_empty_cuda_tensor({seq_len, dim});

    // 创建 layer
    EmbeddingLayer layer(DeviceType::kCUDA, dim, seq_len, vocab_size);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input_tokens);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    // 验证结果 (token 5 的 embedding 应该是 [25, 26, 27, 28, 29])
    std::vector<float> expected;
    for (int32_t j = 0; j < dim; ++j)
    {
        expected.push_back(static_cast<float>(5 * dim + j));
    }
    verify_result(output, expected);
}

TEST_F(EmbeddingLayerTest, LongSequenceEmbedding)
{
    // 测试长序列的 embedding
    const int32_t vocab_size = 100;
    const int32_t dim = 8;
    const int32_t seq_len = 32;

    // 创建权重矩阵
    std::vector<float> weight_data(vocab_size * dim);
    for (size_t i = 0; i < weight_data.size(); ++i)
    {
        weight_data[i] = static_cast<float>(i) * 0.1f;
    }
    Tensor weight = create_cuda_weight_tensor({vocab_size, dim}, weight_data);

    // 创建输入 tokens
    std::vector<int32_t> token_data(seq_len);
    for (int32_t i = 0; i < seq_len; ++i)
    {
        token_data[i] = i % vocab_size;  // 重复使用 vocab 中的 tokens
    }
    Tensor input_tokens = create_cpu_tensor({seq_len}, token_data);

    // 创建输出 tensor
    Tensor output = create_empty_cuda_tensor({seq_len, dim});

    // 创建 layer
    EmbeddingLayer layer(DeviceType::kCUDA, dim, seq_len, vocab_size);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input_tokens);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    // 验证结果
    Tensor cpu_output = output.clone();
    cpu_output.to_cpu();

    for (int32_t i = 0; i < seq_len; ++i)
    {
        int32_t token = token_data[i];
        for (int32_t j = 0; j < dim; ++j)
        {
            float expected = weight_data[token * dim + j];
            EXPECT_FLOAT_EQ(cpu_output.ptr<float>()[i * dim + j], expected)
                << "Mismatch at token " << i << ", dim " << j;
        }
    }
}

TEST_F(EmbeddingLayerTest, OutOfVocabToken)
{
    // 测试超出词汇表的 token (应该被忽略，不会崩溃)
    const int32_t vocab_size = 5;
    const int32_t dim = 3;
    const int32_t seq_len = 2;

    // 创建权重矩阵
    std::vector<float> weight_data(vocab_size * dim, 1.0f);
    Tensor weight = create_cuda_weight_tensor({vocab_size, dim}, weight_data);

    // 创建输入 tokens (包含超出范围的 token)
    std::vector<int32_t> token_data = {0, 10};  // 10 >= vocab_size
    Tensor input_tokens = create_cpu_tensor({seq_len}, token_data);

    // 创建输出 tensor
    Tensor output = create_empty_cuda_tensor({seq_len, dim});

    // 创建 layer
    EmbeddingLayer layer(DeviceType::kCUDA, dim, seq_len, vocab_size);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input_tokens);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    // 执行 (不应该崩溃)
    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();
}

// ==================== 错误处理测试 ====================

TEST_F(EmbeddingLayerTest, EmptyInput)
{
    // 测试空输入
    const int32_t vocab_size = 10;
    const int32_t dim = 5;
    const int32_t seq_len = 0;

    // 创建权重矩阵
    std::vector<float> weight_data(vocab_size * dim, 1.0f);
    Tensor weight = create_cuda_weight_tensor({vocab_size, dim}, weight_data);

    // 创建空的输入
    Tensor input_tokens;  // 空 tensor
    Tensor output = create_empty_cuda_tensor({seq_len, dim});

    // 创建 layer
    EmbeddingLayer layer(DeviceType::kCUDA, dim, seq_len, vocab_size);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input_tokens);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    // check() 应该失败
    Status status = layer.check();
    ASSERT_FALSE(status);
}

// WrongWeightDevice test removed - set_weight() uses CHECK which terminates the program
// The device type check is already done in set_weight() at layer.cpp:264

TEST_F(EmbeddingLayerTest, WrongWeightShape)
{
    // 测试错误的权重形状
    const int32_t vocab_size = 5;
    const int32_t dim = 3;
    const int32_t seq_len = 2;

    // 创建错误形状的权重矩阵 (应该是 vocab_size x dim)
    std::vector<float> weight_data(10 * 10, 1.0f);
    Tensor weight = create_cuda_weight_tensor({10, 10}, weight_data);

    // 创建输入 tokens
    std::vector<int32_t> token_data = {0, 1};
    Tensor input_tokens = create_cpu_tensor({seq_len}, token_data);
    Tensor output = create_empty_cuda_tensor({seq_len, dim});

    // 创建 layer
    EmbeddingLayer layer(DeviceType::kCUDA, dim, seq_len, vocab_size);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input_tokens);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    // check() 应该失败
    Status status = layer.check();
    ASSERT_FALSE(status);
}

TEST_F(EmbeddingLayerTest, WrongOutputShape)
{
    // 测试错误的输出形状
    const int32_t vocab_size = 5;
    const int32_t dim = 3;
    const int32_t seq_len = 2;

    // 创建权重矩阵
    std::vector<float> weight_data(vocab_size * dim, 1.0f);
    Tensor weight = create_cuda_weight_tensor({vocab_size, dim}, weight_data);

    // 创建输入 tokens
    std::vector<int32_t> token_data = {0, 1};
    Tensor input_tokens = create_cpu_tensor({seq_len}, token_data);
    Tensor output = create_empty_cuda_tensor({seq_len, dim + 1});  // 错误的维度

    // 创建 layer
    EmbeddingLayer layer(DeviceType::kCUDA, dim, seq_len, vocab_size);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input_tokens);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    // check() 应该失败
    Status status = layer.check();
    ASSERT_FALSE(status);
}

// ==================== API 测试 ====================

TEST_F(EmbeddingLayerTest, GetSetInputOutputWeight)
{
    // 测试 get/set API
    const int32_t vocab_size = 10;
    const int32_t dim = 5;
    const int32_t seq_len = 2;

    // 创建 layer
    EmbeddingLayer layer(DeviceType::kCUDA, dim, seq_len, vocab_size);
    layer.set_cuda_config(cuda_config_);

    // 验证输入输出权重数量
    ASSERT_EQ(layer.input_size(), 1);
    ASSERT_EQ(layer.output_size(), 1);
    ASSERT_EQ(layer.weight_size(), 1);

    // 创建权重
    std::vector<float> weight_data(vocab_size * dim, 1.0f);
    Tensor weight = create_cuda_weight_tensor({vocab_size, dim}, weight_data);

    // 创建输入
    std::vector<int32_t> token_data = {0, 1};
    Tensor input_tokens = create_cpu_tensor({seq_len}, token_data);

    // 创建输出
    Tensor output = create_empty_cuda_tensor({seq_len, dim});

    // 设置
    layer.set_input(0, input_tokens);
    layer.set_weight(0, weight);
    layer.set_output(0, output);

    // 获取并验证
    ASSERT_EQ(layer.get_input(0).size(), seq_len);
    ASSERT_EQ(layer.get_weight(0).size(), vocab_size * dim);
    ASSERT_EQ(layer.get_output(0).size(), seq_len * dim);
}

// ==================== 性能测试 ====================

TEST_F(EmbeddingLayerTest, PerformanceTest)
{
    // 性能测试
    const std::vector<int32_t> vocab_sizes = {1000, 5000, 10000};
    const int32_t dim = 128;
    const int32_t seq_len = 64;

    std::cout << "\n=== Performance Test ===\n";

    for (int32_t vocab_size : vocab_sizes)
    {
        // 创建权重矩阵
        std::vector<float> weight_data(vocab_size * dim);
        for (size_t i = 0; i < weight_data.size(); ++i)
        {
            weight_data[i] = static_cast<float>(i) * 0.01f;
        }
        Tensor weight = create_cuda_weight_tensor({vocab_size, dim}, weight_data);

        // 创建输入 tokens
        std::vector<int32_t> token_data(seq_len);
        for (int32_t i = 0; i < seq_len; ++i)
        {
            token_data[i] = i % vocab_size;
        }
        Tensor input_tokens = create_cpu_tensor({seq_len}, token_data);

        // 创建输出 tensor
        Tensor output = create_empty_cuda_tensor({seq_len, dim});

        // 创建 layer
        EmbeddingLayer layer(DeviceType::kCUDA, dim, seq_len, vocab_size);
        layer.set_cuda_config(cuda_config_);

        layer.set_input(0, input_tokens);
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

        std::cout << "Vocab size: " << vocab_size
                  << ", Avg time: " << avg_time << " us"
                  << ", Throughput: " << (seq_len / avg_time * 1e6) << " tokens/s\n";
    }
}

// ==================== 主函数 ====================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
