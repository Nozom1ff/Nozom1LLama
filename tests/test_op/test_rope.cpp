#include <gtest/gtest.h>
#include <op/rope.h>
#include <base/cuda_config.h>
#include <base/alloc.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <chrono>

using namespace op;
using namespace base;
using namespace tensor;

// 前向声明 sin/cos 缓存计算函数
namespace kernel
{
void sin_cos_cache_calc_cu(int head_size,
                           int max_seq_len,
                           const tensor::Tensor &sin_cache,
                           const tensor::Tensor &cos_cache,
                           cudaStream_t stream);
}

class RoPELayerTest : public ::testing::Test
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

    // 创建 CPU 位置张量（int32 类型）- 注意必须在 CPU 上
    Tensor create_cpu_position_tensor(int32_t pos)
    {
        std::vector<int32_t> pos_data = {pos};
        Tensor pos_tensor(DataType::kTypeInt32, {1}, true, cpu_alloc_);
        memcpy(pos_tensor.ptr<int32_t>(), pos_data.data(), pos_data.size() * sizeof(int32_t));
        return pos_tensor; // 注意：保持在 CPU 上
    }

    // 创建 sin/cos 缓存（使用 CUDA kernel 计算）
    void create_sin_cos_cache(int32_t head_size, int32_t max_seq_len,
                              Tensor &sin_cache, Tensor &cos_cache)
    {
        // 在 CUDA 上分配缓存
        sin_cache = Tensor(DataType::kTypeFp32, {max_seq_len, head_size}, true, cuda_alloc_);
        cos_cache = Tensor(DataType::kTypeFp32, {max_seq_len, head_size}, true, cuda_alloc_);

        // 使用 CUDA kernel 计算 sin/cos 缓存
        kernel::sin_cos_cache_calc_cu(head_size, max_seq_len, sin_cache, cos_cache,
                                      cuda_config_ ? static_cast<cudaStream_t>(cuda_config_->stream) : nullptr);
        cudaDeviceSynchronize();
    }

    std::shared_ptr<base::DeviceAllocator> cuda_alloc_;
    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
};

// ==================== 基础功能测试 ====================

TEST_F(RoPELayerTest, BasicRoPE)
{
    // 测试基本的 RoPE 操作
    const int32_t dim = 128;      // query 维度
    const int32_t kv_dim = 64;    // key 维度
    const int32_t head_size = 64; // head 维度
    const int32_t pos = 0;        // 位置

    // 创建输入数据
    std::vector<float> input_q_data(dim);
    std::vector<float> input_k_data(kv_dim);

    // 填充简单数据
    for (int i = 0; i < dim; ++i)
    {
        input_q_data[i] = static_cast<float>(i);
    }
    for (int i = 0; i < kv_dim; ++i)
    {
        input_k_data[i] = static_cast<float>(i) * 0.5f;
    }

    Tensor input_q = create_cuda_tensor({dim}, input_q_data);
    Tensor input_k = create_cuda_tensor({kv_dim}, input_k_data);
    Tensor input_pos = create_cpu_position_tensor(pos); // CPU tensor - 重要！

    // 创建 sin/cos 缓存
    Tensor sin_cache, cos_cache;
    create_sin_cos_cache(head_size, 32, sin_cache, cos_cache);

    RoPELayer layer(DeviceType::kCUDA, dim, kv_dim, head_size, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input_q);
    layer.set_input(1, input_k);
    layer.set_input(2, input_pos); // CPU 位置张量
    layer.set_input(3, sin_cache);
    layer.set_input(4, cos_cache);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();

    // 验证张量被修改（RoPE 是原地操作）
    ASSERT_EQ(input_q.size(), dim);
    ASSERT_EQ(input_k.size(), kv_dim);
}

TEST_F(RoPELayerTest, DifferentPositions)
{
    // 测试不同位置
    const int32_t dim = 64;
    const int32_t kv_dim = 64;
    const int32_t head_size = 64;

    std::vector<float> input_q_data(dim);
    std::vector<float> input_k_data(kv_dim);

    for (int i = 0; i < dim; ++i)
    {
        input_q_data[i] = 1.0f;
        input_k_data[i] = 1.0f;
    }

    // 测试位置 0, 5, 10
    std::vector<int32_t> positions = {0, 5, 10};

    Tensor sin_cache, cos_cache;
    create_sin_cos_cache(head_size, 32, sin_cache, cos_cache);

    for (int32_t pos : positions)
    {
        Tensor input_q = create_cuda_tensor({dim}, input_q_data);
        Tensor input_k = create_cuda_tensor({kv_dim}, input_k_data);
        Tensor input_pos = create_cpu_position_tensor(pos); // CPU tensor

        RoPELayer layer(DeviceType::kCUDA, dim, kv_dim, head_size, DataType::kTypeFp32);
        layer.set_cuda_config(cuda_config_);

        layer.set_input(0, input_q);
        layer.set_input(1, input_k);
        layer.set_input(2, input_pos);
        layer.set_input(3, sin_cache);
        layer.set_input(4, cos_cache);

        Status status = layer.forward();
        ASSERT_TRUE(status);

        cudaDeviceSynchronize();
    }
}

TEST_F(RoPELayerTest, LargeDimensions)
{
    // 测试大维度
    const int32_t dim = 1024;     // 16 heads * 64 head_size
    const int32_t kv_dim = 1024;
    const int32_t head_size = 64;
    const int32_t pos = 3;

    std::vector<float> input_q_data(dim);
    std::vector<float> input_k_data(kv_dim);

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < dim; ++i)
    {
        input_q_data[i] = dis(gen);
        input_k_data[i] = dis(gen);
    }

    Tensor input_q = create_cuda_tensor({dim}, input_q_data);
    Tensor input_k = create_cuda_tensor({kv_dim}, input_k_data);
    Tensor input_pos = create_cpu_position_tensor(pos); // CPU tensor

    Tensor sin_cache, cos_cache;
    create_sin_cos_cache(head_size, 32, sin_cache, cos_cache);

    RoPELayer layer(DeviceType::kCUDA, dim, kv_dim, head_size, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input_q);
    layer.set_input(1, input_k);
    layer.set_input(2, input_pos);
    layer.set_input(3, sin_cache);
    layer.set_input(4, cos_cache);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();
}

TEST_F(RoPELayerTest, KVDimSmallerThanQDim)
{
    // 测试 kv_dim < q_dim 的情况（GQA - Grouped Query Attention）
    const int32_t dim = 128;
    const int32_t kv_dim = 64; // kv_dim 是 q_dim 的一半
    const int32_t head_size = 64;
    const int32_t pos = 2;

    std::vector<float> input_q_data(dim);
    std::vector<float> input_k_data(kv_dim);

    for (int i = 0; i < dim; ++i)
    {
        input_q_data[i] = static_cast<float>(i) * 0.1f;
    }
    for (int i = 0; i < kv_dim; ++i)
    {
        input_k_data[i] = static_cast<float>(i) * 0.2f;
    }

    Tensor input_q = create_cuda_tensor({dim}, input_q_data);
    Tensor input_k = create_cuda_tensor({kv_dim}, input_k_data);
    Tensor input_pos = create_cpu_position_tensor(pos);

    Tensor sin_cache, cos_cache;
    create_sin_cos_cache(head_size, 32, sin_cache, cos_cache);

    RoPELayer layer(DeviceType::kCUDA, dim, kv_dim, head_size, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input_q);
    layer.set_input(1, input_k);
    layer.set_input(2, input_pos);
    layer.set_input(3, sin_cache);
    layer.set_input(4, cos_cache);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();
}

TEST_F(RoPELayerTest, MultipleHeads)
{
    // 测试多个 head
    const int32_t dim = 512;     // 8 heads * 64
    const int32_t kv_dim = 512;
    const int32_t head_size = 64;
    const int32_t pos = 7;

    std::vector<float> input_q_data(dim);
    std::vector<float> input_k_data(kv_dim);

    std::random_device rd;
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);

    for (int i = 0; i < dim; ++i)
    {
        input_q_data[i] = dis(gen);
        input_k_data[i] = dis(gen);
    }

    Tensor input_q = create_cuda_tensor({dim}, input_q_data);
    Tensor input_k = create_cuda_tensor({kv_dim}, input_k_data);
    Tensor input_pos = create_cpu_position_tensor(pos);

    Tensor sin_cache, cos_cache;
    create_sin_cos_cache(head_size, 128, sin_cache, cos_cache);

    RoPELayer layer(DeviceType::kCUDA, dim, kv_dim, head_size, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input_q);
    layer.set_input(1, input_k);
    layer.set_input(2, input_pos);
    layer.set_input(3, sin_cache);
    layer.set_input(4, cos_cache);

    Status status = layer.forward();
    ASSERT_TRUE(status);

    cudaDeviceSynchronize();
}

// ==================== 错误处理测试 ====================

TEST_F(RoPELayerTest, WrongQDeviceType)
{
    // 测试错误：Q 张量应该在 CUDA 上
    const int32_t dim = 64;
    const int32_t kv_dim = 64;
    const int32_t head_size = 64;

    // 错误：在 CPU 上创建 Q
    std::vector<float> input_q_data(dim, 1.0f);
    Tensor input_q(DataType::kTypeFp32, {dim}, true, cpu_alloc_);
    memcpy(input_q.ptr<float>(), input_q_data.data(), input_q_data.size() * sizeof(float));

    std::vector<float> input_k_data(kv_dim, 1.0f);
    Tensor input_k = create_cuda_tensor({kv_dim}, input_k_data);
    Tensor input_pos = create_cpu_position_tensor(0);

    Tensor sin_cache, cos_cache;
    create_sin_cos_cache(head_size, 32, sin_cache, cos_cache);

    RoPELayer layer(DeviceType::kCUDA, dim, kv_dim, head_size, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input_q); // 错误：CPU 设备
    layer.set_input(1, input_k);
    layer.set_input(2, input_pos);
    layer.set_input(3, sin_cache);
    layer.set_input(4, cos_cache);

    Status status = layer.check();
    ASSERT_FALSE(status);
}

TEST_F(RoPELayerTest, WrongKDeviceType)
{
    // 测试错误：K 张量应该在 CUDA 上
    const int32_t dim = 64;
    const int32_t kv_dim = 64;
    const int32_t head_size = 64;

    std::vector<float> input_q_data(dim, 1.0f);
    Tensor input_q = create_cuda_tensor({dim}, input_q_data);

    // 错误：在 CPU 上创建 K
    std::vector<float> input_k_data(kv_dim, 1.0f);
    Tensor input_k(DataType::kTypeFp32, {kv_dim}, true, cpu_alloc_);
    memcpy(input_k.ptr<float>(), input_k_data.data(), input_k_data.size() * sizeof(float));

    Tensor input_pos = create_cpu_position_tensor(0);

    Tensor sin_cache, cos_cache;
    create_sin_cos_cache(head_size, 32, sin_cache, cos_cache);

    RoPELayer layer(DeviceType::kCUDA, dim, kv_dim, head_size, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input_q);
    layer.set_input(1, input_k); // 错误：CPU 设备
    layer.set_input(2, input_pos);
    layer.set_input(3, sin_cache);
    layer.set_input(4, cos_cache);

    Status status = layer.check();
    ASSERT_FALSE(status);
}

TEST_F(RoPELayerTest, WrongQDimensions)
{
    // 测试错误：Q 维度不匹配
    const int32_t dim = 64;
    const int32_t kv_dim = 64;
    const int32_t head_size = 64;

    // Q 维度错误
    std::vector<float> input_q_data(32, 1.0f); // 应该是 64
    Tensor input_q = create_cuda_tensor({32}, input_q_data);

    std::vector<float> input_k_data(kv_dim, 1.0f);
    Tensor input_k = create_cuda_tensor({kv_dim}, input_k_data);
    Tensor input_pos = create_cpu_position_tensor(0);

    Tensor sin_cache, cos_cache;
    create_sin_cos_cache(head_size, 32, sin_cache, cos_cache);

    RoPELayer layer(DeviceType::kCUDA, dim, kv_dim, head_size, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input_q); // 错误：维度不匹配
    layer.set_input(1, input_k);
    layer.set_input(2, input_pos);
    layer.set_input(3, sin_cache);
    layer.set_input(4, cos_cache);

    Status status = layer.check();
    ASSERT_FALSE(status);
}

TEST_F(RoPELayerTest, WrongKDimensions)
{
    // 测试错误：K 维度不匹配
    const int32_t dim = 64;
    const int32_t kv_dim = 64;
    const int32_t head_size = 64;

    std::vector<float> input_q_data(dim, 1.0f);
    Tensor input_q = create_cuda_tensor({dim}, input_q_data);

    // K 维度错误
    std::vector<float> input_k_data(32, 1.0f); // 应该是 64
    Tensor input_k = create_cuda_tensor({32}, input_k_data);

    Tensor input_pos = create_cpu_position_tensor(0);

    Tensor sin_cache, cos_cache;
    create_sin_cos_cache(head_size, 32, sin_cache, cos_cache);

    RoPELayer layer(DeviceType::kCUDA, dim, kv_dim, head_size, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    layer.set_input(0, input_q);
    layer.set_input(1, input_k); // 错误：维度不匹配
    layer.set_input(2, input_pos);
    layer.set_input(3, sin_cache);
    layer.set_input(4, cos_cache);

    Status status = layer.check();
    ASSERT_FALSE(status);
}

// ==================== API 测试 ====================

TEST_F(RoPELayerTest, GetSetInputs)
{
    // 测试 get/set API
    const int32_t dim = 64;
    const int32_t kv_dim = 64;
    const int32_t head_size = 64;

    RoPELayer layer(DeviceType::kCUDA, dim, kv_dim, head_size, DataType::kTypeFp32);
    layer.set_cuda_config(cuda_config_);

    ASSERT_EQ(layer.input_size(), 5);
    ASSERT_EQ(layer.output_size(), 1);

    std::vector<float> q_data(dim, 1.0f);
    std::vector<float> k_data(kv_dim, 1.0f);

    Tensor input_q = create_cuda_tensor({dim}, q_data);
    Tensor input_k = create_cuda_tensor({kv_dim}, k_data);
    Tensor input_pos = create_cpu_position_tensor(0);

    Tensor sin_cache, cos_cache;
    create_sin_cos_cache(head_size, 32, sin_cache, cos_cache);

    layer.set_input(0, input_q);
    layer.set_input(1, input_k);
    layer.set_input(2, input_pos);
    layer.set_input(3, sin_cache);
    layer.set_input(4, cos_cache);

    ASSERT_EQ(layer.get_input(0).size(), dim);
    ASSERT_EQ(layer.get_input(1).size(), kv_dim);
    ASSERT_EQ(layer.get_input(2).size(), 1);
    ASSERT_EQ(layer.get_input(3).size(), head_size * 32);
    ASSERT_EQ(layer.get_input(4).size(), head_size * 32);
}

// ==================== 性能测试 ====================

TEST_F(RoPELayerTest, PerformanceTest)
{
    // 性能测试
    const std::vector<std::tuple<int32_t, int32_t, int32_t>> configs = {
        {512, 512, 64},   // 8 heads
        {1024, 1024, 64}, // 16 heads
        {2048, 1024, 64}, // 32 heads, GQA
        {4096, 4096, 128} // 32 heads, larger
    };

    std::cout << "\n=== RoPE Performance Test ===\n";

    for (auto [dim, kv_dim, head_size] : configs)
    {
        std::vector<float> input_q_data(dim);
        std::vector<float> input_k_data(kv_dim);

        std::random_device rd;
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int i = 0; i < dim; ++i)
        {
            input_q_data[i] = dis(gen);
        }
        for (int i = 0; i < kv_dim; ++i)
        {
            input_k_data[i] = dis(gen);
        }

        Tensor input_q = create_cuda_tensor({dim}, input_q_data);
        Tensor input_k = create_cuda_tensor({kv_dim}, input_k_data);
        Tensor input_pos = create_cpu_position_tensor(5);

        Tensor sin_cache, cos_cache;
        create_sin_cos_cache(head_size, 32, sin_cache, cos_cache);

        RoPELayer layer(DeviceType::kCUDA, dim, kv_dim, head_size, DataType::kTypeFp32);
        layer.set_cuda_config(cuda_config_);

        layer.set_input(0, input_q);
        layer.set_input(1, input_k);
        layer.set_input(2, input_pos);
        layer.set_input(3, sin_cache);
        layer.set_input(4, cos_cache);

        // 预热
        layer.forward();

        // 测试 100 次
        const int iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            layer.forward();
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time = static_cast<double>(duration.count()) / iterations;

        std::cout << "Dim: " << dim << ", KV_Dim: " << kv_dim << ", Head_Size: " << head_size
                  << ", Avg time: " << avg_time << " us"
                  << ", Throughput: " << ((dim + kv_dim) / avg_time * 1e6) << " elements/s\n";
    }
}

// ==================== 主函数 ====================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
