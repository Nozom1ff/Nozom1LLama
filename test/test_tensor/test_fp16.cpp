#include <gtest/gtest.h>
#include <cuda_runtime.h>  // 先包含 CUDA 头文件
#include <base/base.h>
#include <base/data_type.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

using namespace base;

// FP16 数据类型基础测试
TEST(FP16Test, DataTypeSize) {
  EXPECT_EQ(DataTypeSize(DataType::kDataTypeFp32), 4);
  EXPECT_EQ(DataTypeSize(DataType::kDataTypeInt8), 1);
  EXPECT_EQ(DataTypeSize(DataType::kDataTypeInt32), 4);
  EXPECT_EQ(DataTypeSize(DataType::kDataTypeFp16), 2);  // 新增
}

// 单值转换测试
TEST(FP16Test, SingleValueConversion) {
  // FP32 → FP16 → FP32
  float original = 3.14159f;
  float16_t fp16 = fp32_to_fp16(original);
  float converted = fp16_to_fp32(fp16);

  EXPECT_NEAR(original, converted, 0.001f);

  // INT8 → FP16 (带 scale)
  int8_t int8_val = 64;
  float scale = 0.05f;
  fp16 = int8_to_fp16(int8_val, scale);
  float expected = static_cast<float>(int8_val) * scale;
  converted = fp16_to_fp32(fp16);

  EXPECT_NEAR(expected, converted, 0.001f);
}

// 向量转换测试
TEST(FP16Test, VectorConversion) {
  const size_t size = 1000;

  // 生成测试数据
  std::vector<float> fp32_data(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for (auto& val : fp32_data) {
    val = dis(gen);
  }

  // FP32 → FP16 → FP32
  std::vector<float16_t> fp16_data = DataTypeConverter::fp32_to_fp16(fp32_data);
  std::vector<float> fp32_converted = DataTypeConverter::fp16_to_fp32(fp16_data);

  // 验证精度
  float max_error = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    float error = std::abs(fp32_data[i] - fp32_converted[i]);
    max_error = std::max(max_error, error);
  }

  LOG(INFO) << "FP32→FP16→FP32 max error: " << max_error;
  EXPECT_LT(max_error, 0.01f);
}

// INT8 → FP16 转换测试
TEST(FP16Test, INT8ToFP16Conversion) {
  const size_t size = 1024;
  const int group_size = 128;

  // 生成 INT8 权重
  std::vector<int8_t> int8_weights(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(-128, 127);

  for (auto& w : int8_weights) {
    w = static_cast<int8_t>(dis(gen));
  }

  // 生成 scales
  std::vector<float> scales(size / group_size);
  std::uniform_real_distribution<float> scale_dis(0.01f, 0.1f);
  for (auto& s : scales) {
    s = scale_dis(gen);
  }

  // INT8 → FP16
  std::vector<float16_t> fp16_weights =
    DataTypeConverter::int8_to_fp16(int8_weights, scales, group_size);

  // 验证精度
  float max_error = DataTypeConverter::verify_int8_to_fp16_precision(
    int8_weights.data(), scales.data(), group_size,
    fp16_weights.data(), size);

  LOG(INFO) << "INT8→FP16 max error: " << max_error;
  EXPECT_LT(max_error, 0.01f);
}

// GPU 功能检测测试
TEST(FP16Test, GPUCapabilities) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device available";
    return;
  }

  bool fp16_supported = DataTypeConverter::supports_fp16();
  LOG(INFO) << "FP16 supported: " << (fp16_supported ? "Yes" : "No");

  bool tensor_core_supported = DataTypeConverter::supports_fp16_tensor_core();
  LOG(INFO) << "FP16 Tensor Core supported: " << (tensor_core_supported ? "Yes" : "No");

  std::string arch = DataTypeConverter::get_gpu_architecture();
  LOG(INFO) << "GPU: " << arch;
}

// GPU 端转换测试 (需要 CUDA)
TEST(FP16Test, GPUConversion) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device available";
    return;
  }

  if (!DataTypeConverter::supports_fp16()) {
    GTEST_SKIP() << "FP16 not supported on this device";
    return;
  }

  const size_t size = 1024 * 1024;

  // 生成测试数据
  std::vector<float> fp32_host(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for (auto& val : fp32_host) {
    val = dis(gen);
  }

  // 分配 GPU 内存
  float* fp32_device = nullptr;
  float16_t* fp16_device = nullptr;
  float* fp32_recon_device = nullptr;

  ASSERT_EQ(cudaMalloc(&fp32_device, size * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&fp16_device, size * sizeof(float16_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&fp32_recon_device, size * sizeof(float)), cudaSuccess);

  // 拷贝到 GPU
  ASSERT_EQ(cudaMemcpy(fp32_device, fp32_host.data(),
                      size * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

  // GPU 端转换
  DataTypeConverter::fp32_to_fp16_gpu(fp32_device, fp16_device, size);
  DataTypeConverter::fp16_to_fp32_gpu(fp16_device, fp32_recon_device, size);

  // 拷贝回主机
  std::vector<float> fp32_recon_host(size);
  ASSERT_EQ(cudaMemcpy(fp32_recon_host.data(), fp32_recon_device,
                      size * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

  // 验证精度
  float max_error = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    float error = std::abs(fp32_host[i] - fp32_recon_host[i]);
    max_error = std::max(max_error, error);
  }

  LOG(INFO) << "GPU FP32→FP16→FP32 max error: " << max_error;
  EXPECT_LT(max_error, 0.01f);

  // 清理
  cudaFree(fp32_device);
  cudaFree(fp16_device);
  cudaFree(fp32_recon_device);
}

// INT8 → FP16 GPU 转换测试
TEST(FP16Test, INT8ToFP16_GPU) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device available";
    return;
  }

  if (!DataTypeConverter::supports_fp16()) {
    GTEST_SKIP() << "FP16 not supported on this device";
    return;
  }

  const size_t size = 1024 * 512;
  const int group_size = 128;

  // 生成测试数据
  std::vector<int8_t> int8_host(size);
  std::vector<float> scales_host(size / group_size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(-128, 127);
  std::uniform_real_distribution<float> scale_dis(0.01f, 0.1f);

  for (auto& w : int8_host) {
    w = static_cast<int8_t>(dis(gen));
  }
  for (auto& s : scales_host) {
    s = scale_dis(gen);
  }

  // 分配 GPU 内存
  int8_t* int8_device = nullptr;
  float* scales_device = nullptr;
  float16_t* fp16_device = nullptr;

  ASSERT_EQ(cudaMalloc(&int8_device, size * sizeof(int8_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&scales_device, (size / group_size) * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&fp16_device, size * sizeof(float16_t)), cudaSuccess);

  // 拷贝到 GPU
  ASSERT_EQ(cudaMemcpy(int8_device, int8_host.data(),
                      size * sizeof(int8_t), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(scales_device, scales_host.data(),
                      (size / group_size) * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

  // GPU 端转换
  DataTypeConverter::int8_to_fp16_gpu(
    int8_device, scales_device, group_size, size, fp16_device);

  // 拷贝回主机验证
  std::vector<float16_t> fp16_host(size);
  ASSERT_EQ(cudaMemcpy(fp16_host.data(), fp16_device,
                      size * sizeof(float16_t), cudaMemcpyDeviceToHost), cudaSuccess);

  // 验证精度
  float max_error = DataTypeConverter::verify_int8_to_fp16_precision(
    int8_host.data(), scales_host.data(), group_size,
    fp16_host.data(), size);

  LOG(INFO) << "GPU INT8→FP16 max error: " << max_error;
  EXPECT_LT(max_error, 0.01f);

  // 清理
  cudaFree(int8_device);
  cudaFree(scales_device);
  cudaFree(fp16_device);
}

// 性能基准测试
TEST(FP16Test, PerformanceBenchmark) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device available";
    return;
  }

  const size_t size = 10 * 1024 * 1024;  // 10M 元素

  // 生成测试数据
  std::vector<float> fp32_data(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for (auto& val : fp32_data) {
    val = dis(gen);
  }

  // CPU 转换性能
  auto cpu_start = std::chrono::high_resolution_clock::now();
  auto fp16_data = DataTypeConverter::fp32_to_fp16(fp32_data);
  auto cpu_end = std::chrono::high_resolution_clock::now();

  double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
  double cpu_gb_s = (size * (sizeof(float) + sizeof(float16_t))) / (cpu_ms / 1000.0) / (1024 * 1024 * 1024);

  LOG(INFO) << "CPU FP32→FP16: " << cpu_ms << " ms, " << cpu_gb_s << " GB/s";

  if (DataTypeConverter::supports_fp16()) {
    // GPU 转换性能
    float* fp32_device = nullptr;
    float16_t* fp16_device = nullptr;

    ASSERT_EQ(cudaMalloc(&fp32_device, size * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&fp16_device, size * sizeof(float16_t)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(fp32_device, fp32_data.data(),
                        size * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    cudaDeviceSynchronize();
    auto gpu_start = std::chrono::high_resolution_clock::now();

    DataTypeConverter::fp32_to_fp16_gpu(fp32_device, fp16_device, size);

    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();

    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    double gpu_gb_s = (size * (sizeof(float) + sizeof(float16_t))) / (gpu_ms / 1000.0) / (1024 * 1024 * 1024);

    LOG(INFO) << "GPU FP32→FP16: " << gpu_ms << " ms, " << gpu_gb_s << " GB/s";
    LOG(INFO) << "Speedup: " << (cpu_ms / gpu_ms) << "x";

    EXPECT_LT(gpu_ms, cpu_ms);

    cudaFree(fp32_device);
    cudaFree(fp16_device);
  }
}
