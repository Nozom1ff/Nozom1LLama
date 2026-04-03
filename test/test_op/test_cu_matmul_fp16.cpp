// CUDA 头文件必须在最前面，使用 cuda_runtime.h 而不是 cuda_runtime_api.h
#include <cuda_runtime.h>

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <vector>

#include "../source/op/kernels/cuda/matmul_kernel.cuh"
#include "base/buffer.h"
#include "base/base.h"
#include "base/data_type.h"  // 在 cuda_runtime.h 之后包含

using namespace kernel;

// ========== WMMA FP16 Matmul 测试 ==========

/**
 * @brief CPU 端手动计算 FP32 矩阵乘法作为参考
 */
static void cpu_matmul_reference(const float* input, const float* weight, float* output, int M, int K) {
  // input: [M], weight: [K, M], output: [K]
  // output[i] = sum_j(input[j] * weight[i * M + j])
  for (int i = 0; i < K; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < M; ++j) {
      sum += input[j] * weight[i * M + j];
    }
    output[i] = sum;
  }
}

/**
 * @brief 基本 WMMA FP16 矩阵乘法测试
 * 测试输入 [M] 与权重 [K, M] 的矩阵乘法，输出 [K]
 */
TEST(test_matmul_fp16_wmma, basic_small) {
  // 检查 GPU 是否支持 FP16 Tensor Core
  if (!base::DataTypeConverter::supports_fp16_tensor_core()) {
    GTEST_SKIP() << "FP16 Tensor Core not supported on this device";
    return;
  }

  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  // 测试维度: input[M=16], weight[K=16, M=16], output[K=16]
  // 注意：WMMA 要求维度是 16 的倍数
  const int M = 16;
  const int K = 16;

  // 创建 CPU 端 FP32 张量
  std::vector<float> input_fp32(M);
  std::vector<float> weight_fp32(K * M);

  // 初始化测试数据
  for (int i = 0; i < M; ++i) {
    input_fp32[i] = 1.0f;  // 全 1
  }
  for (int i = 0; i < K * M; ++i) {
    weight_fp32[i] = static_cast<float>(i);  // 0, 1, 2, ..., K*M-1
  }

  // 转换为 FP16
  std::vector<base::float16_t> input_fp16 = base::DataTypeConverter::fp32_to_fp16(input_fp32);
  std::vector<base::float16_t> weight_fp16 = base::DataTypeConverter::fp32_to_fp16(weight_fp32);

  // 创建 FP16 张量
  tensor::Tensor input_fp16_tensor(base::DataType::kDataTypeFp16, M, false, alloc_cpu, input_fp16.data());
  tensor::Tensor weight_fp16_tensor(base::DataType::kDataTypeFp16, K, M, false, alloc_cpu, weight_fp16.data());

  // 拷贝到 GPU
  input_fp16_tensor.to_cuda(nullptr);
  weight_fp16_tensor.to_cuda(nullptr);

  // 创建 GPU 输出张量 (FP32)
  tensor::Tensor output_fp32(base::DataType::kDataTypeFp32, K, true, alloc_cu);

  // 执行 WMMA FP16 kernel
  CudaConfig config;
  config.stream = nullptr;
  matmul_kernel_cu_fp16(input_fp16_tensor, weight_fp16_tensor, output_fp32, &config);

  // 拷贝结果回 CPU
  output_fp32.to_cpu();

  // CPU 端计算参考结果
  std::vector<float> output_ref(K);
  cpu_matmul_reference(input_fp32.data(), weight_fp32.data(), output_ref.data(), M, K);

  // 验证结果 (允许一定的 FP16 精度误差)
  const float eps = 0.1f;  // FP16 精度容差
  for (int i = 0; i < K; ++i) {
    float gpu_result = output_fp32.index<float>(i);
    float ref_result = output_ref[i];
    float rel_error = std::abs(gpu_result - ref_result) / (std::abs(ref_result) + 1e-6f);

    EXPECT_LT(rel_error, eps) << "Mismatch at index " << i
                              << ": GPU=" << gpu_result << ", Ref=" << ref_result;
  }
}

/**
 * @brief 中等规模 WMMA FP16 矩阵乘法测试
 */
TEST(test_matmul_fp16_wmma, medium_size) {
  if (!base::DataTypeConverter::supports_fp16_tensor_core()) {
    GTEST_SKIP() << "FP16 Tensor Core not supported on this device";
    return;
  }

  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  // 测试维度: input[M=128], weight[K=64, M=128], output[K=64]
  const int M = 128;
  const int K = 64;

  // 创建数据
  std::vector<float> input_fp32(M);
  std::vector<float> weight_fp32(K * M);

  // 使用随机数据
  std::mt19937 gen(42);  // 固定种子以便复现
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for (int i = 0; i < M; ++i) {
    input_fp32[i] = dis(gen);
  }
  for (int i = 0; i < K * M; ++i) {
    weight_fp32[i] = dis(gen);
  }

  // 转换为 FP16
  std::vector<base::float16_t> input_fp16 = base::DataTypeConverter::fp32_to_fp16(input_fp32);
  std::vector<base::float16_t> weight_fp16 = base::DataTypeConverter::fp32_to_fp16(weight_fp32);

  // 创建 FP16 张量
  tensor::Tensor input_fp16_tensor(base::DataType::kDataTypeFp16, M, false, alloc_cpu, input_fp16.data());
  tensor::Tensor weight_fp16_tensor(base::DataType::kDataTypeFp16, K, M, false, alloc_cpu, weight_fp16.data());

  // 拷贝到 GPU
  input_fp16_tensor.to_cuda(nullptr);
  weight_fp16_tensor.to_cuda(nullptr);

  // 创建 GPU 输出张量 (FP32)
  tensor::Tensor output_fp32(base::DataType::kDataTypeFp32, K, true, alloc_cu);

  // 执行 WMMA FP16 kernel
  CudaConfig config;
  config.stream = nullptr;
  matmul_kernel_cu_fp16(input_fp16_tensor, weight_fp16_tensor, output_fp32, &config);

  // 拷贝结果回 CPU
  output_fp32.to_cpu();

  // CPU 端计算参考结果
  std::vector<float> output_ref(K);
  cpu_matmul_reference(input_fp32.data(), weight_fp32.data(), output_ref.data(), M, K);

  // 验证结果
  float max_rel_error = 0.0f;
  const float eps = 0.05f;  // FP16 相对精度容差

  for (int i = 0; i < K; ++i) {
    float gpu_result = output_fp32.index<float>(i);
    float ref_result = output_ref[i];
    float rel_error = std::abs(gpu_result - ref_result) / (std::abs(ref_result) + 1e-6f);
    max_rel_error = std::max(max_rel_error, rel_error);

    EXPECT_LT(rel_error, eps) << "Mismatch at index " << i
                              << ": GPU=" << gpu_result << ", Ref=" << ref_result;
  }

  LOG(INFO) << "Max relative error: " << max_rel_error;
}

/**
 * @brief 大规模 WMMA FP16 矩阵乘法测试
 * 测试接近实际 LLM 模型的维度
 */
TEST(test_matmul_fp16_wmma, large_size) {
  if (!base::DataTypeConverter::supports_fp16_tensor_core()) {
    GTEST_SKIP() << "FP16 Tensor Core not supported on this device";
    return;
  }

  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  // 测试维度: 模拟 LLM 层
  // input[M=1024], weight[K=1024, M=1024], output[K=1024]
  const int M = 1024;
  const int K = 1024;

  LOG(INFO) << "Testing WMMA FP16 matmul with M=" << M << ", K=" << K;

  // 创建数据
  std::vector<float> input_fp32(M);
  std::vector<float> weight_fp32(K * M);

  // 使用随机数据
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-0.1f, 0.1f);  // 小范围避免溢出

  for (int i = 0; i < M; ++i) {
    input_fp32[i] = dis(gen);
  }
  for (int i = 0; i < K * M; ++i) {
    weight_fp32[i] = dis(gen);
  }

  // 转换为 FP16
  std::vector<base::float16_t> input_fp16 = base::DataTypeConverter::fp32_to_fp16(input_fp32);
  std::vector<base::float16_t> weight_fp16 = base::DataTypeConverter::fp32_to_fp16(weight_fp32);

  // 创建 FP16 张量
  tensor::Tensor input_fp16_tensor(base::DataType::kDataTypeFp16, M, false, alloc_cpu, input_fp16.data());
  tensor::Tensor weight_fp16_tensor(base::DataType::kDataTypeFp16, K, M, false, alloc_cpu, weight_fp16.data());

  // 拷贝到 GPU
  input_fp16_tensor.to_cuda(nullptr);
  weight_fp16_tensor.to_cuda(nullptr);

  // 创建 GPU 输出张量 (FP32)
  tensor::Tensor output_fp32(base::DataType::kDataTypeFp32, K, true, alloc_cu);

  // 执行 WMMA FP16 kernel
  CudaConfig config;
  config.stream = nullptr;

  // 计时
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  matmul_kernel_cu_fp16(input_fp16_tensor, weight_fp16_tensor, output_fp32, &config);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // 拷贝结果回 CPU
  output_fp32.to_cpu();

  // 计算性能
  double gflops = (2.0 * M * K) / (milliseconds / 1000.0) / 1e9;
  LOG(INFO) << "WMMA FP16 matmul time: " << milliseconds << " ms";
  LOG(INFO) << "Performance: " << gflops << " GFLOPS";

  // 抽样验证几个点
  const int num_samples = 10;
  std::mt19937 sample_gen(42);
  std::uniform_int_distribution<int> sample_dis(0, K - 1);

  const float eps = 0.01f;
  for (int i = 0; i < num_samples; ++i) {
    int idx = sample_dis(sample_gen);

    // 手动计算参考结果
    float ref_result = 0.0f;
    for (int j = 0; j < M; ++j) {
      ref_result += input_fp32[j] * weight_fp32[idx * M + j];
    }

    float gpu_result = output_fp32.index<float>(idx);
    float rel_error = std::abs(gpu_result - ref_result) / (std::abs(ref_result) + 1e-6f);

    EXPECT_LT(rel_error, eps) << "Sample mismatch at index " << idx
                              << ": GPU=" << gpu_result << ", Ref=" << ref_result;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

/**
 * @brief 边界条件测试 - 极小维度
 */
TEST(test_matmul_fp16_wmma, edge_case_minimum) {
  if (!base::DataTypeConverter::supports_fp16_tensor_core()) {
    GTEST_SKIP() << "FP16 Tensor Core not supported on this device";
    return;
  }

  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  // WMMA 最小维度: 16
  const int M = 16;
  const int K = 16;

  std::vector<float> input_fp32(M);
  std::vector<float> weight_fp32(K * M);

  // 全零测试
  for (int i = 0; i < M; ++i) {
    input_fp32[i] = 0.0f;
  }
  for (int i = 0; i < K * M; ++i) {
    weight_fp32[i] = static_cast<float>(i);
  }

  std::vector<base::float16_t> input_fp16 = base::DataTypeConverter::fp32_to_fp16(input_fp32);
  std::vector<base::float16_t> weight_fp16 = base::DataTypeConverter::fp32_to_fp16(weight_fp32);

  tensor::Tensor input_fp16_tensor(base::DataType::kDataTypeFp16, M, false, alloc_cpu, input_fp16.data());
  tensor::Tensor weight_fp16_tensor(base::DataType::kDataTypeFp16, K, M, false, alloc_cpu, weight_fp16.data());

  input_fp16_tensor.to_cuda(nullptr);
  weight_fp16_tensor.to_cuda(nullptr);

  tensor::Tensor output_fp32(base::DataType::kDataTypeFp32, K, true, alloc_cu);

  CudaConfig config;
  config.stream = nullptr;
  matmul_kernel_cu_fp16(input_fp16_tensor, weight_fp16_tensor, output_fp32, &config);

  output_fp32.to_cpu();

  // 全零输入应该得到全零输出
  for (int i = 0; i < K; ++i) {
    EXPECT_NEAR(output_fp32.index<float>(i), 0.0f, 1e-5f);
  }
}

/**
 * @brief 使用 CUDA Stream 的测试
 */
TEST(test_matmul_fp16_wmma, with_stream) {
  if (!base::DataTypeConverter::supports_fp16_tensor_core()) {
    GTEST_SKIP() << "FP16 Tensor Core not supported on this device";
    return;
  }

  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  const int M = 128;
  const int K = 128;

  std::vector<float> input_fp32(M);
  std::vector<float> weight_fp32(K * M);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for (int i = 0; i < M; ++i) {
    input_fp32[i] = dis(gen);
  }
  for (int i = 0; i < K * M; ++i) {
    weight_fp32[i] = dis(gen);
  }

  std::vector<base::float16_t> input_fp16 = base::DataTypeConverter::fp32_to_fp16(input_fp32);
  std::vector<base::float16_t> weight_fp16 = base::DataTypeConverter::fp32_to_fp16(weight_fp32);

  tensor::Tensor input_fp16_tensor(base::DataType::kDataTypeFp16, M, false, alloc_cpu, input_fp16.data());
  tensor::Tensor weight_fp16_tensor(base::DataType::kDataTypeFp16, K, M, false, alloc_cpu, weight_fp16.data());

  input_fp16_tensor.to_cuda(nullptr);
  weight_fp16_tensor.to_cuda(nullptr);

  tensor::Tensor output_fp32(base::DataType::kDataTypeFp32, K, true, alloc_cu);

  // 创建 CUDA stream
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  CudaConfig config;
  config.stream = stream;
  matmul_kernel_cu_fp16(input_fp16_tensor, weight_fp16_tensor, output_fp32, &config);

  // 等待 stream 完成
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  output_fp32.to_cpu();

  // 计算参考结果
  std::vector<float> output_ref(K);
  cpu_matmul_reference(input_fp32.data(), weight_fp32.data(), output_ref.data(), M, K);

  // 验证结果
  const float eps = 0.05f;
  for (int i = 0; i < K; ++i) {
    float gpu_result = output_fp32.index<float>(i);
    float ref_result = output_ref[i];
    float rel_error = std::abs(gpu_result - ref_result) / (std::abs(ref_result) + 1e-6f);
    EXPECT_LT(rel_error, eps);
  }

  cudaStreamDestroy(stream);
}

/**
 * @brief 非对齐维度测试
 * 测试 M 不是 16 的倍数的情况
 */
TEST(test_matmul_fp16_wmma, non_aligned_size) {
  if (!base::DataTypeConverter::supports_fp16_tensor_core()) {
    GTEST_SKIP() << "FP16 Tensor Core not supported on this device";
    return;
  }

  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  // 非对齐维度
  const int M = 100;  // 不是 16 的倍数
  const int K = 64;

  std::vector<float> input_fp32(M);
  std::vector<float> weight_fp32(K * M);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

  for (int i = 0; i < M; ++i) {
    input_fp32[i] = dis(gen);
  }
  for (int i = 0; i < K * M; ++i) {
    weight_fp32[i] = dis(gen);
  }

  std::vector<base::float16_t> input_fp16 = base::DataTypeConverter::fp32_to_fp16(input_fp32);
  std::vector<base::float16_t> weight_fp16 = base::DataTypeConverter::fp32_to_fp16(weight_fp32);

  tensor::Tensor input_fp16_tensor(base::DataType::kDataTypeFp16, M, false, alloc_cpu, input_fp16.data());
  tensor::Tensor weight_fp16_tensor(base::DataType::kDataTypeFp16, K, M, false, alloc_cpu, weight_fp16.data());

  input_fp16_tensor.to_cuda(nullptr);
  weight_fp16_tensor.to_cuda(nullptr);

  tensor::Tensor output_fp32(base::DataType::kDataTypeFp32, K, true, alloc_cu);

  CudaConfig config;
  config.stream = nullptr;
  matmul_kernel_cu_fp16(input_fp16_tensor, weight_fp16_tensor, output_fp32, &config);

  output_fp32.to_cpu();

  // 计算参考结果
  std::vector<float> output_ref(K);
  cpu_matmul_reference(input_fp32.data(), weight_fp32.data(), output_ref.data(), M, K);

  // 验证结果
  const float eps = 0.05f;
  for (int i = 0; i < K; ++i) {
    float gpu_result = output_fp32.index<float>(i);
    float ref_result = output_ref[i];
    float rel_error = std::abs(gpu_result - ref_result) / (std::abs(ref_result) + 1e-6f);
    EXPECT_LT(rel_error, eps) << "Mismatch at index " << i
                              << ": GPU=" << gpu_result << ", Ref=" << ref_result;
  }
}
