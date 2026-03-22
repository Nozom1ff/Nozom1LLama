#include <gtest/gtest.h>
#include <op/mha.h>
#include <base/cuda_config.h>
#include <base/alloc.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <cmath>
#include <vector>
#include <numeric>

using namespace op;
using namespace base;
using namespace tensor;

class MultiHeadAttentionTest : public ::testing::Test
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

    // CPU 参考实现：计算 attention score
    std::vector<float> cpu_attention_score(const std::vector<float> &query,
                                           const std::vector<float> &key_cache,
                                           int32_t pos, int32_t head_size,
                                           int32_t kv_dim, int32_t head_offset)
    {
        std::vector<float> scores(pos + 1, 0.0f);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_size));

        for (int t = 0; t <= pos; ++t)
        {
            float score = 0.0f;
            for (int i = 0; i < head_size; ++i)
            {
                score += query[i] * key_cache[t * kv_dim + head_offset + i];
            }
            scores[t] = score * scale;
        }
        return scores;
    }

    // CPU 参考实现：softmax
    std::vector<float> cpu_softmax(const std::vector<float> &scores)
    {
        std::vector<float> result(scores.size());

        // 找最大值
        float max_val = *std::max_element(scores.begin(), scores.end());

        // 计算 exp(sum)
        float sum = 0.0f;
        for (size_t i = 0; i < scores.size(); ++i)
        {
            result[i] = std::exp(scores[i] - max_val);
            sum += result[i];
        }

        // 归一化
        for (size_t i = 0; i < scores.size(); ++i)
        {
            result[i] /= sum;
        }
        return result;
    }

    // CPU 参考实现：计算 attention output
    std::vector<float> cpu_attention_output(const std::vector<float> &scores,
                                            const std::vector<float> &value_cache,
                                            int32_t pos, int32_t head_size,
                                            int32_t kv_dim, int32_t head_offset)
    {
        std::vector<float> output(head_size, 0.0f);
        for (int i = 0; i < head_size; ++i)
        {
            float value = 0.0f;
            for (int t = 0; t <= pos; ++t)
            {
                value += scores[t] * value_cache[t * kv_dim + head_offset + i];
            }
            output[i] = value;
        }
        return output;
    }

    // 完整的 CPU 参考实现
    std::vector<float> cpu_mha_reference(const std::vector<float> &query,
                                         const std::vector<float> &key_cache,
                                         const std::vector<float> &value_cache,
                                         int32_t pos, int32_t head_size,
                                         int32_t kv_dim, int32_t head_offset)
    {
        // 1. 计算 attention scores
        auto scores = cpu_attention_score(query, key_cache, pos, head_size, kv_dim, head_offset);

        // 2. Softmax
        auto attn_weights = cpu_softmax(scores);

        // 3. 计算 output
        return cpu_attention_output(attn_weights, value_cache, pos, head_size, kv_dim, head_offset);
    }

    std::shared_ptr<base::DeviceAllocator> cuda_alloc_;
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
};

// ==================== 基础功能测试 ====================

TEST_F(MultiHeadAttentionTest, BasicMHASingleHead)
{
    // 测试基础 MHA：单头，pos=0
    const int32_t layer_index = 0;
    const int32_t kv_mul = 1;  // MHA (多头注意力)
    const int32_t kv_dim = 8;  // head_size * head_num
    const int32_t seq_len = 4;
    const int32_t head_num = 1;
    const int32_t head_size = 8;
    const int32_t pos = 0;

    auto layer = std::make_unique<MultiHeadAttention>(
        DeviceType::kCUDA, layer_index, kv_mul, kv_dim, seq_len,
        head_num, head_size, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);
    layer->set_pos(pos);

    // 准备输入数据
    // query: [head_num * head_size] = [8]
    std::vector<float> query_data = {1.0f, 0.5f, 0.2f, 0.8f, 0.3f, 0.9f, 0.1f, 0.6f};

    // key_cache: [layer_index * seq_len * kv_dim] = [0 * 4 * 8]
    std::vector<float> key_cache_data(seq_len * kv_dim, 0.0f);
    // 填充 position 0 的 key
    for (int i = 0; i < kv_dim; ++i)
    {
        key_cache_data[i] = 0.5f + i * 0.1f;
    }

    // value_cache: [layer_index * seq_len * kv_dim] = [0 * 4 * 8]
    std::vector<float> value_cache_data(seq_len * kv_dim, 0.0f);
    // 填充 position 0 的 value
    for (int i = 0; i < kv_dim; ++i)
    {
        value_cache_data[i] = 0.3f + i * 0.15f;
    }

    // score tensor: [head_num * seq_len] = [1 * 4]
    std::vector<float> score_data(head_num * seq_len, 0.0f);

    // 创建 tensor
    Tensor query = create_cuda_tensor({head_num * head_size}, query_data);
    Tensor key_cache = create_cuda_tensor({seq_len * kv_dim}, key_cache_data);
    Tensor value_cache = create_cuda_tensor({seq_len * kv_dim}, value_cache_data);
    Tensor score = create_cuda_tensor({head_num * seq_len}, score_data);
    Tensor output = create_empty_cuda_tensor({head_num * head_size});

    layer->set_input(0, query);
    layer->set_input(1, score);
    layer->set_input(2, key_cache);
    layer->set_input(3, value_cache);
    layer->set_output(0, output);

    Status status = layer->forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    // 计算预期结果
    auto expected = cpu_mha_reference(query_data, key_cache_data, value_cache_data,
                                      pos, head_size, kv_dim, 0);

    verify_result(output, expected, 1e-2f);
}

TEST_F(MultiHeadAttentionTest, MHAMultiHead)
{
    // 测试多头 MHA
    const int32_t layer_index = 0;
    const int32_t kv_mul = 1;  // MHA
    const int32_t kv_dim = 32; // head_size * head_num = 8 * 4
    const int32_t seq_len = 8;
    const int32_t head_num = 4;
    const int32_t head_size = 8;
    const int32_t pos = 2;

    auto layer = std::make_unique<MultiHeadAttention>(
        DeviceType::kCUDA, layer_index, kv_mul, kv_dim, seq_len,
        head_num, head_size, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);
    layer->set_pos(pos);

    // 准备输入数据
    std::vector<float> query_data(head_num * head_size);
    std::vector<float> key_cache_data(seq_len * kv_dim);
    std::vector<float> value_cache_data(seq_len * kv_dim);
    std::vector<float> score_data(head_num * seq_len, 0.0f);

    // 填充随机数据
    std::mt19937 gen(42);  // 固定种子
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < query_data.size(); ++i)
    {
        query_data[i] = dis(gen);
    }
    for (size_t i = 0; i < key_cache_data.size(); ++i)
    {
        key_cache_data[i] = dis(gen);
    }
    for (size_t i = 0; i < value_cache_data.size(); ++i)
    {
        value_cache_data[i] = dis(gen);
    }

    Tensor query = create_cuda_tensor({head_num * head_size}, query_data);
    Tensor key_cache = create_cuda_tensor({seq_len * kv_dim}, key_cache_data);
    Tensor value_cache = create_cuda_tensor({seq_len * kv_dim}, value_cache_data);
    Tensor score = create_cuda_tensor({head_num * seq_len}, score_data);
    Tensor output = create_empty_cuda_tensor({head_num * head_size});

    layer->set_input(0, query);
    layer->set_input(1, score);
    layer->set_input(2, key_cache);
    layer->set_input(3, value_cache);
    layer->set_output(0, output);

    Status status = layer->forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    // 验证每个 head
    Tensor cpu_output = output.clone();
    cpu_output.to_cpu();
    const float *output_ptr = cpu_output.ptr<float>();

    for (int head = 0; head < head_num; ++head)
    {
        // 提取当前 head 的 query
        std::vector<float> head_query(head_size);
        for (int i = 0; i < head_size; ++i)
        {
            head_query[i] = query_data[head * head_size + i];
        }

        // 计算预期结果
        int32_t head_offset = head * head_size;
        auto expected = cpu_mha_reference(head_query, key_cache_data, value_cache_data,
                                          pos, head_size, kv_dim, head_offset);

        // 验证当前 head 的输出
        for (int i = 0; i < head_size; ++i)
        {
            EXPECT_NEAR(output_ptr[head * head_size + i], expected[i], 1e-2f)
                << "Mismatch at head " << head << ", index " << i;
        }
    }
}

TEST_F(MultiHeadAttentionTest, MHAGroupedQueryAttention)
{
    // 测试 GQA (Grouped Query Attention)
    const int32_t layer_index = 0;
    const int32_t kv_mul = 2;  // 每 2 个 query head 共享 1 个 key/value head
    const int32_t kv_dim = 16; // head_size * (head_num / kv_mul) = 8 * 2
    const int32_t seq_len = 8;
    const int32_t head_num = 4;
    const int32_t head_size = 8;
    const int32_t pos = 3;

    auto layer = std::make_unique<MultiHeadAttention>(
        DeviceType::kCUDA, layer_index, kv_mul, kv_dim, seq_len,
        head_num, head_size, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);
    layer->set_pos(pos);

    // 准备输入数据
    std::vector<float> query_data(head_num * head_size);
    std::vector<float> key_cache_data(seq_len * kv_dim);
    std::vector<float> value_cache_data(seq_len * kv_dim);
    std::vector<float> score_data(head_num * seq_len, 0.0f);

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < query_data.size(); ++i)
    {
        query_data[i] = dis(gen);
    }
    for (size_t i = 0; i < key_cache_data.size(); ++i)
    {
        key_cache_data[i] = dis(gen);
    }
    for (size_t i = 0; i < value_cache_data.size(); ++i)
    {
        value_cache_data[i] = dis(gen);
    }

    Tensor query = create_cuda_tensor({head_num * head_size}, query_data);
    Tensor key_cache = create_cuda_tensor({seq_len * kv_dim}, key_cache_data);
    Tensor value_cache = create_cuda_tensor({seq_len * kv_dim}, value_cache_data);
    Tensor score = create_cuda_tensor({head_num * seq_len}, score_data);
    Tensor output = create_empty_cuda_tensor({head_num * head_size});

    layer->set_input(0, query);
    layer->set_input(1, score);
    layer->set_input(2, key_cache);
    layer->set_input(3, value_cache);
    layer->set_output(0, output);

    Status status = layer->forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    // 验证每个 head
    Tensor cpu_output = output.clone();
    cpu_output.to_cpu();
    const float *output_ptr = cpu_output.ptr<float>();

    for (int head = 0; head < head_num; ++head)
    {
        // 提取当前 head 的 query
        std::vector<float> head_query(head_size);
        for (int i = 0; i < head_size; ++i)
        {
            head_query[i] = query_data[head * head_size + i];
        }

        // GQA: head_offset = (head / kv_mul) * head_size
        int32_t head_offset = (head / kv_mul) * head_size;
        auto expected = cpu_mha_reference(head_query, key_cache_data, value_cache_data,
                                          pos, head_size, kv_dim, head_offset);

        // 验证当前 head 的输出
        for (int i = 0; i < head_size; ++i)
        {
            EXPECT_NEAR(output_ptr[head * head_size + i], expected[i], 1e-2f)
                << "Mismatch at head " << head << ", index " << i;
        }
    }
}

TEST_F(MultiHeadAttentionTest, MHADifferentPositions)
{
    // 测试不同序列位置
    const int32_t layer_index = 0;
    const int32_t kv_mul = 1;
    const int32_t kv_dim = 16;
    const int32_t seq_len = 16;
    const int32_t head_num = 2;
    const int32_t head_size = 8;

    std::vector<float> query_data(head_num * head_size);
    std::vector<float> key_cache_data(seq_len * kv_dim);
    std::vector<float> value_cache_data(seq_len * kv_dim);

    std::mt19937 gen(456);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < query_data.size(); ++i)
    {
        query_data[i] = dis(gen);
    }
    for (size_t i = 0; i < key_cache_data.size(); ++i)
    {
        key_cache_data[i] = dis(gen);
    }
    for (size_t i = 0; i < value_cache_data.size(); ++i)
    {
        value_cache_data[i] = dis(gen);
    }

    // 测试不同的位置
    std::vector<int32_t> positions = {0, 1, 3, 7, 15};

    for (int32_t pos : positions)
    {
        auto layer = std::make_unique<MultiHeadAttention>(
            DeviceType::kCUDA, layer_index, kv_mul, kv_dim, seq_len,
            head_num, head_size, DataType::kTypeFp32);
        layer->set_cuda_config(cuda_config_);
        layer->set_pos(pos);

        std::vector<float> score_data(head_num * seq_len, 0.0f);

        Tensor query = create_cuda_tensor({head_num * head_size}, query_data);
        Tensor key_cache = create_cuda_tensor({seq_len * kv_dim}, key_cache_data);
        Tensor value_cache = create_cuda_tensor({seq_len * kv_dim}, value_cache_data);
        Tensor score = create_cuda_tensor({head_num * seq_len}, score_data);
        Tensor output = create_empty_cuda_tensor({head_num * head_size});

        layer->set_input(0, query);
        layer->set_input(1, score);
        layer->set_input(2, key_cache);
        layer->set_input(3, value_cache);
        layer->set_output(0, output);

        Status status = layer->forward();
        ASSERT_TRUE(status) << "Failed at pos " << pos;

        cudaDeviceSynchronize();

        // 验证第一个 head
        Tensor cpu_output = output.clone();
        cpu_output.to_cpu();
        const float *output_ptr = cpu_output.ptr<float>();

        std::vector<float> head_query(head_size);
        for (int i = 0; i < head_size; ++i)
        {
            head_query[i] = query_data[i];
        }

        auto expected = cpu_mha_reference(head_query, key_cache_data, value_cache_data,
                                          pos, head_size, kv_dim, 0);

        for (int i = 0; i < head_size; ++i)
        {
            EXPECT_NEAR(output_ptr[i], expected[i], 1e-2f)
                << "Mismatch at pos " << pos << ", index " << i;
        }
    }
}

TEST_F(MultiHeadAttentionTest, MHADifferentLayers)
{
    // 测试不同层 - 验证不同层索引不会导致崩溃
    const int32_t kv_mul = 1;
    const int32_t kv_dim = 16;
    const int32_t seq_len = 8;
    const int32_t head_num = 2;
    const int32_t head_size = 8;
    const int32_t pos = 2;
    const int32_t num_layers = 3;

    // 使用简单的确定性数据
    std::vector<float> query_data(head_num * head_size);
    for (size_t i = 0; i < query_data.size(); ++i)
    {
        query_data[i] = 0.1f * (i + 1);
    }

    for (int32_t layer_index = 0; layer_index < num_layers; ++layer_index)
    {
        auto layer = std::make_unique<MultiHeadAttention>(
            DeviceType::kCUDA, layer_index, kv_mul, kv_dim, seq_len,
            head_num, head_size, DataType::kTypeFp32);
        layer->set_cuda_config(cuda_config_);
        layer->set_pos(pos);

        std::vector<float> score_data(head_num * seq_len, 0.0f);

        // 为每一层创建简单的数据（注意：实际使用时，cache 应该包含所有层的数据）
        std::vector<float> layer_key_cache(seq_len * kv_dim, 0.1f);
        std::vector<float> layer_value_cache(seq_len * kv_dim, 0.2f);

        Tensor query = create_cuda_tensor({head_num * head_size}, query_data);
        Tensor key_cache = create_cuda_tensor({seq_len * kv_dim}, layer_key_cache);
        Tensor value_cache = create_cuda_tensor({seq_len * kv_dim}, layer_value_cache);
        Tensor score = create_cuda_tensor({head_num * seq_len}, score_data);
        Tensor output = create_empty_cuda_tensor({head_num * head_size});

        layer->set_input(0, query);
        layer->set_input(1, score);
        layer->set_input(2, key_cache);
        layer->set_input(3, value_cache);
        layer->set_output(0, output);

        Status status = layer->forward();
        ASSERT_TRUE(status) << "Failed at layer " << layer_index;

        cudaDeviceSynchronize();

        // 验证输出存在且不是全部为零
        Tensor cpu_output = output.clone();
        cpu_output.to_cpu();
        const float *output_ptr = cpu_output.ptr<float>();

        bool all_zero = true;
        for (int i = 0; i < head_num * head_size; ++i)
        {
            if (std::abs(output_ptr[i]) > 1e-6f)
            {
                all_zero = false;
                break;
            }
        }
        EXPECT_FALSE(all_zero) << "Output is all zero for layer " << layer_index;
    }
}

// ==================== 边界情况测试 ====================

TEST_F(MultiHeadAttentionTest, MHALargeHeadSize)
{
    // 测试较大的 head_size
    const int32_t layer_index = 0;
    const int32_t kv_mul = 1;
    const int32_t kv_dim = 128; // head_size * head_num = 128 * 1
    const int32_t seq_len = 32;
    const int32_t head_num = 1;
    const int32_t head_size = 128;
    const int32_t pos = 10;

    auto layer = std::make_unique<MultiHeadAttention>(
        DeviceType::kCUDA, layer_index, kv_mul, kv_dim, seq_len,
        head_num, head_size, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);
    layer->set_pos(pos);

    std::vector<float> query_data(head_num * head_size);
    std::vector<float> key_cache_data(seq_len * kv_dim);
    std::vector<float> value_cache_data(seq_len * kv_dim);
    std::vector<float> score_data(head_num * seq_len, 0.0f);

    std::mt19937 gen(999);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < query_data.size(); ++i)
    {
        query_data[i] = dis(gen);
    }
    for (size_t i = 0; i < key_cache_data.size(); ++i)
    {
        key_cache_data[i] = dis(gen);
    }
    for (size_t i = 0; i < value_cache_data.size(); ++i)
    {
        value_cache_data[i] = dis(gen);
    }

    Tensor query = create_cuda_tensor({head_num * head_size}, query_data);
    Tensor key_cache = create_cuda_tensor({seq_len * kv_dim}, key_cache_data);
    Tensor value_cache = create_cuda_tensor({seq_len * kv_dim}, value_cache_data);
    Tensor score = create_cuda_tensor({head_num * seq_len}, score_data);
    Tensor output = create_empty_cuda_tensor({head_num * head_size});

    layer->set_input(0, query);
    layer->set_input(1, score);
    layer->set_input(2, key_cache);
    layer->set_input(3, value_cache);
    layer->set_output(0, output);

    Status status = layer->forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    // 验证部分结果
    Tensor cpu_output = output.clone();
    cpu_output.to_cpu();
    const float *output_ptr = cpu_output.ptr<float>();

    std::vector<float> head_query(head_size);
    for (int i = 0; i < head_size; ++i)
    {
        head_query[i] = query_data[i];
    }

    auto expected = cpu_mha_reference(head_query, key_cache_data, value_cache_data,
                                      pos, head_size, kv_dim, 0);

    // 验证前 32 个元素
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_NEAR(output_ptr[i], expected[i], 1e-2f)
            << "Mismatch at index " << i;
    }
}

TEST_F(MultiHeadAttentionTest, MHAManyHeads)
{
    // 测试多个头
    const int32_t layer_index = 0;
    const int32_t kv_mul = 1;
    const int32_t kv_dim = 256; // head_size * head_num = 32 * 8
    const int32_t seq_len = 16;
    const int32_t head_num = 8;
    const int32_t head_size = 32;
    const int32_t pos = 5;

    auto layer = std::make_unique<MultiHeadAttention>(
        DeviceType::kCUDA, layer_index, kv_mul, kv_dim, seq_len,
        head_num, head_size, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);
    layer->set_pos(pos);

    std::vector<float> query_data(head_num * head_size);
    std::vector<float> key_cache_data(seq_len * kv_dim);
    std::vector<float> value_cache_data(seq_len * kv_dim);
    std::vector<float> score_data(head_num * seq_len, 0.0f);

    std::mt19937 gen(111);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < query_data.size(); ++i)
    {
        query_data[i] = dis(gen);
    }
    for (size_t i = 0; i < key_cache_data.size(); ++i)
    {
        key_cache_data[i] = dis(gen);
    }
    for (size_t i = 0; i < value_cache_data.size(); ++i)
    {
        value_cache_data[i] = dis(gen);
    }

    Tensor query = create_cuda_tensor({head_num * head_size}, query_data);
    Tensor key_cache = create_cuda_tensor({seq_len * kv_dim}, key_cache_data);
    Tensor value_cache = create_cuda_tensor({seq_len * kv_dim}, value_cache_data);
    Tensor score = create_cuda_tensor({head_num * seq_len}, score_data);
    Tensor output = create_empty_cuda_tensor({head_num * head_size});

    layer->set_input(0, query);
    layer->set_input(1, score);
    layer->set_input(2, key_cache);
    layer->set_input(3, value_cache);
    layer->set_output(0, output);

    Status status = layer->forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    // 验证每个 head 的部分元素
    Tensor cpu_output = output.clone();
    cpu_output.to_cpu();
    const float *output_ptr = cpu_output.ptr<float>();

    for (int head = 0; head < head_num; ++head)
    {
        std::vector<float> head_query(head_size);
        for (int i = 0; i < head_size; ++i)
        {
            head_query[i] = query_data[head * head_size + i];
        }

        int32_t head_offset = head * head_size;
        auto expected = cpu_mha_reference(head_query, key_cache_data, value_cache_data,
                                          pos, head_size, kv_dim, head_offset);

        // 验证前 8 个元素
        for (int i = 0; i < 8; ++i)
        {
            EXPECT_NEAR(output_ptr[head * head_size + i], expected[i], 1e-2f)
                << "Mismatch at head " << head << ", index " << i;
        }
    }
}

TEST_F(MultiHeadAttentionTest, MHAUniformQuery)
{
    // 测试 uniform query (所有元素相同) - 应该产生均匀的 attention weights
    const int32_t layer_index = 0;
    const int32_t kv_mul = 1;
    const int32_t kv_dim = 8;
    const int32_t seq_len = 4;
    const int32_t head_num = 1;
    const int32_t head_size = 8;
    const int32_t pos = 2;

    auto layer = std::make_unique<MultiHeadAttention>(
        DeviceType::kCUDA, layer_index, kv_mul, kv_dim, seq_len,
        head_num, head_size, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);
    layer->set_pos(pos);

    // 使用相同的 query 值
    std::vector<float> query_data(head_num * head_size, 1.0f);
    std::vector<float> key_cache_data(seq_len * kv_dim, 1.0f);
    std::vector<float> value_cache_data(seq_len * kv_dim, 1.0f);
    std::vector<float> score_data(head_num * seq_len, 0.0f);

    Tensor query = create_cuda_tensor({head_num * head_size}, query_data);
    Tensor key_cache = create_cuda_tensor({seq_len * kv_dim}, key_cache_data);
    Tensor value_cache = create_cuda_tensor({seq_len * kv_dim}, value_cache_data);
    Tensor score = create_cuda_tensor({head_num * seq_len}, score_data);
    Tensor output = create_empty_cuda_tensor({head_num * head_size});

    layer->set_input(0, query);
    layer->set_input(1, score);
    layer->set_input(2, key_cache);
    layer->set_input(3, value_cache);
    layer->set_output(0, output);

    Status status = layer->forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    // 验证输出存在且不为零
    Tensor cpu_output = output.clone();
    cpu_output.to_cpu();
    const float *output_ptr = cpu_output.ptr<float>();

    for (int i = 0; i < head_num * head_size; ++i)
    {
        EXPECT_GT(std::abs(output_ptr[i]), 0.0f) << "Output should not be zero at index " << i;
    }
}

// ==================== API 测试 ====================

TEST_F(MultiHeadAttentionTest, SetPosAndLayerIdx)
{
    // 测试 set_pos 和 set_layer_idx API
    const int32_t layer_index = 0;
    const int32_t kv_mul = 1;
    const int32_t kv_dim = 8;
    const int32_t seq_len = 4;
    const int32_t head_num = 1;
    const int32_t head_size = 8;

    auto layer = std::make_unique<MultiHeadAttention>(
        DeviceType::kCUDA, layer_index, kv_mul, kv_dim, seq_len,
        head_num, head_size, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);

    // 测试 set_pos
    layer->set_pos(2);
    layer->set_layer_idx(1);

    // 验证 layer 可以正常 forward
    std::vector<float> query_data(head_num * head_size, 1.0f);
    std::vector<float> key_cache_data(seq_len * kv_dim, 0.5f);
    std::vector<float> value_cache_data(seq_len * kv_dim, 0.3f);
    std::vector<float> score_data(head_num * seq_len, 0.0f);

    Tensor query = create_cuda_tensor({head_num * head_size}, query_data);
    Tensor key_cache = create_cuda_tensor({seq_len * kv_dim}, key_cache_data);
    Tensor value_cache = create_cuda_tensor({seq_len * kv_dim}, value_cache_data);
    Tensor score = create_cuda_tensor({head_num * seq_len}, score_data);
    Tensor output = create_empty_cuda_tensor({head_num * head_size});

    layer->set_input(0, query);
    layer->set_input(1, score);
    layer->set_input(2, key_cache);
    layer->set_input(3, value_cache);
    layer->set_output(0, output);

    Status status = layer->forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();
}

// ==================== 性能测试 ====================

TEST_F(MultiHeadAttentionTest, PerformanceTest)
{
    // 性能测试
    const int32_t layer_index = 0;
    const int32_t kv_mul = 1;
    const int32_t seq_len = 128;
    const int32_t head_size = 64;

    std::cout << "\n=== MHA Performance Test ===\n";

    std::vector<std::pair<int32_t, int32_t>> configs = {
        {8, 8},    // 8 heads, 64 dim
        {16, 16},  // 16 heads, 128 dim
        {32, 32},  // 32 heads, 256 dim
    };

    for (auto [head_num, kv_dim] : configs)
    {
        auto layer = std::make_unique<MultiHeadAttention>(
            DeviceType::kCUDA, layer_index, kv_mul, kv_dim, seq_len,
            head_num, head_size, DataType::kTypeFp32);
        layer->set_cuda_config(cuda_config_);
        layer->set_pos(seq_len - 1);

        std::vector<float> query_data(head_num * head_size, 1.0f);
        std::vector<float> key_cache_data(seq_len * kv_dim, 0.5f);
        std::vector<float> value_cache_data(seq_len * kv_dim, 0.3f);
        std::vector<float> score_data(head_num * seq_len, 0.0f);

        Tensor query = create_cuda_tensor({head_num * head_size}, query_data);
        Tensor key_cache = create_cuda_tensor({seq_len * kv_dim}, key_cache_data);
        Tensor value_cache = create_cuda_tensor({seq_len * kv_dim}, value_cache_data);
        Tensor score = create_cuda_tensor({head_num * seq_len}, score_data);
        Tensor output = create_empty_cuda_tensor({head_num * head_size});

        layer->set_input(0, query);
        layer->set_input(1, score);
        layer->set_input(2, key_cache);
        layer->set_input(3, value_cache);
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

        std::cout << "Heads: " << head_num
                  << ", KV dim: " << kv_dim
                  << ", Avg time: " << avg_time << " us\n";
    }
}

TEST_F(MultiHeadAttentionTest, LargeScaleMHATest)
{
    // 大规模 MHA 测试
    const int32_t layer_index = 0;
    const int32_t kv_mul = 4;  // GQA
    const int32_t kv_dim = 512; // 32 key/value heads * 16 head_size
    const int32_t seq_len = 2048;
    const int32_t head_num = 32; // 32 query heads
    const int32_t head_size = 64;  // 每个头 64 维
    const int32_t pos = 1023;

    auto layer = std::make_unique<MultiHeadAttention>(
        DeviceType::kCUDA, layer_index, kv_mul, kv_dim, seq_len,
        head_num, head_size, DataType::kTypeFp32);
    layer->set_cuda_config(cuda_config_);
    layer->set_pos(pos);

    std::vector<float> query_data(head_num * head_size, 1.0f);
    std::vector<float> key_cache_data(seq_len * kv_dim, 0.5f);
    std::vector<float> value_cache_data(seq_len * kv_dim, 0.3f);
    std::vector<float> score_data(head_num * seq_len, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();

    Tensor query = create_cuda_tensor({head_num * head_size}, query_data);
    Tensor key_cache = create_cuda_tensor({seq_len * kv_dim}, key_cache_data);
    Tensor value_cache = create_cuda_tensor({seq_len * kv_dim}, value_cache_data);
    Tensor score = create_cuda_tensor({head_num * seq_len}, score_data);
    Tensor output = create_empty_cuda_tensor({head_num * head_size});

    layer->set_input(0, query);
    layer->set_input(1, score);
    layer->set_input(2, key_cache);
    layer->set_input(3, value_cache);
    layer->set_output(0, output);

    Status status = layer->forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Large scale MHA (32 heads, seq_len=2048, pos=1023): "
              << duration.count() << " ms\n";

    // 验证输出不为零
    Tensor cpu_output = output.clone();
    cpu_output.to_cpu();
    const float *output_ptr = cpu_output.ptr<float>();

    bool all_zero = true;
    for (int i = 0; i < head_num * head_size; ++i)
    {
        if (std::abs(output_ptr[i]) > 1e-6f)
        {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "Output is all zero!";
}

// ==================== 主函数 ====================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
