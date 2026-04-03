#include "base/data_type.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUDA FP16 转换实现
// 这些函数必须在 .cu 文件中定义，因为使用了 CUDA kernel

namespace kernel {

// GPU: FP32 → FP16 kernel
__global__ void fp32_to_fp16_kernel(const float* __restrict__ src,
                                    unsigned short* __restrict__ dst,
                                    size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = __float2half(src[idx]);
  }
}

// GPU: FP16 → FP32 kernel
__global__ void fp16_to_fp32_kernel(const unsigned short* __restrict__ src,
                                    float* __restrict__ dst,
                                    size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = __half2float(src[idx]);
  }
}

// GPU: INT8 → FP16 kernel
__global__ void int8_to_fp16_kernel(const int8_t* __restrict__ int8_weights,
                                    const float* __restrict__ scales,
                                    int group_size,
                                    size_t size,
                                    unsigned short* __restrict__ dst) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int group_idx = static_cast<int>(idx) / group_size;
    float dequantized = static_cast<float>(int8_weights[idx]) * scales[group_idx];
    dst[idx] = __float2half(dequantized);
  }
}

}  // namespace kernel

namespace base {

void DataTypeConverter::fp32_to_fp16_gpu(const float* src_device,
                                        unsigned short* dst_device,
                                        size_t size,
                                        cudaStream_t stream) {
  if (!src_device || !dst_device) {
    throw std::runtime_error("src_device or dst_device is null");
  }

  const int threads = 256;
  const int blocks = (size + threads - 1) / threads;

  if (stream) {
    kernel::fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
        src_device, dst_device, size);
  } else {
    kernel::fp32_to_fp16_kernel<<<blocks, threads>>>(
        src_device, dst_device, size);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in fp32_to_fp16_gpu: " +
                            std::string(cudaGetErrorString(err)));
  }
}

void DataTypeConverter::fp16_to_fp32_gpu(const unsigned short* src_device,
                                        float* dst_device,
                                        size_t size,
                                        cudaStream_t stream) {
  if (!src_device || !dst_device) {
    throw std::runtime_error("src_device or dst_device is null");
  }

  const int threads = 256;
  const int blocks = (size + threads - 1) / threads;

  if (stream) {
    kernel::fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
        src_device, dst_device, size);
  } else {
    kernel::fp16_to_fp32_kernel<<<blocks, threads>>>(
        src_device, dst_device, size);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in fp16_to_fp32_gpu: " +
                            std::string(cudaGetErrorString(err)));
  }
}

void DataTypeConverter::int8_to_fp16_gpu(const int8_t* int8_weights_device,
                                        const float* scales_device,
                                        int group_size,
                                        size_t size,
                                        unsigned short* dst_device,
                                        cudaStream_t stream) {
  if (!int8_weights_device || !scales_device || !dst_device) {
    throw std::runtime_error("input pointer is null");
  }

  const int threads = 256;
  const int blocks = (size + threads - 1) / threads;

  if (stream) {
    kernel::int8_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
        int8_weights_device, scales_device, group_size, size, dst_device);
  } else {
    kernel::int8_to_fp16_kernel<<<blocks, threads>>>(
        int8_weights_device, scales_device, group_size, size, dst_device);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in int8_to_fp16_gpu: " +
                            std::string(cudaGetErrorString(err)));
  }
}

// ========== GPU 能力检测实现 ==========

bool DataTypeConverter::supports_fp16() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    return false;
  }

  int device = 0;
  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return false;
  }

  // FP16 支持需要计算能力 sm_53+ (Maxwell) 或更高
  // 但原生 FP16 计算建议 sm_60+ (Pascal)
  return (prop.major >= 5 && prop.minor >= 3) || (prop.major >= 6);
}

bool DataTypeConverter::supports_fp16_tensor_core() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    return false;
  }

  int device = 0;
  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return false;
  }

  // Tensor Core 支持需要计算能力 sm_70+ (Volta) 或更高
  // sm_70: Volta (V100)
  // sm_75: Turing (RTX 20 系列, T4)
  // sm_80: Ampere (A100, RTX 30 系列)
  // sm_86: Ampere (RTX 30 系列, A40)
  // sm_89: Ada Lovelace (RTX 40 系列, L40)
  // sm_90: Hopper (H100)
  return prop.major >= 7;
}

std::string DataTypeConverter::get_gpu_architecture() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    return "CPU only (no CUDA)";
  }

  int device = 0;
  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return "Unknown GPU";
  }

  // 返回 GPU 名称和计算能力
  char buffer[256];
  snprintf(buffer, sizeof(buffer), "%s (sm_%d%d)",
           prop.name, prop.major, prop.minor);
  return std::string(buffer);
}

}  // namespace base
