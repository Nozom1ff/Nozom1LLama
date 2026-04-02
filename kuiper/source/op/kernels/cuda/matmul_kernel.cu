#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    int row_offset = p * M;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

#pragma unroll
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      sdata[tid] += part_sum;
    }

    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  // CHECK_EQ(M % packet_size, 0);

  CHECK_EQ(M, input.get_dim(0));
  if (config && config->stream) {
    matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                              const_cast<float*>(output.ptr<float>()), M, K);
  }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);
  CHECK_EQ(M, input.get_dim(0));
  if (config->stream) {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                  scale.ptr<float>(), group_size,
                                                  const_cast<float*>(output.ptr<float>()), M, K);
  }
}

// ========== WMMA FP16 Kernel 实现 ==========

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda::wmma;

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void matmul_wmma_fp16_kernel(
    const half* __restrict__ A,   // [M, K] 输入矩阵 (FP16)
    const half* __restrict__ B,   // [K, N] 权重矩阵 (FP16, 转置后)
    float* __restrict__ C,        // [M, N] 输出矩阵 (FP32)
    int M, int N, int K) {

  // WMMA 配置: 16x16x16
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;

  // Warp 索引 (每个 warp 处理一个 16x16 的分块)
  const int warp_m = (blockIdx.x * blockDim.y + threadIdx.y) / 32;
  const int warp_n = (blockIdx.y * blockDim.x + threadIdx.x) / 32;

  // 边界检查
  if (warp_m * WMMA_M >= M || warp_n * WMMA_N >= N) {
    return;
  }

  // WMMA fragments
  fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
  fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
  fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  // 共享内存分块
  __shared__ half shmem_A[TILE_M][TILE_K + 8];  // +8 避免bank conflict
  __shared__ half shmem_B[TILE_K][TILE_N + 8];

  // 初始化累加器为 0
  fill_fragment(c_frag, 0.0f);

  // 分块矩阵乘法
  for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
    // 加载 A 的当前块到共享内存
    #pragma unroll
    for (int i = threadIdx.y; i < TILE_M; i += blockDim.y) {
      #pragma unroll
      for (int j = threadIdx.x; j < TILE_K; j += blockDim.x) {
        int row = warp_m * WMMA_M + i;
        int col = k_tile + j;
        if (row < M && col < K) {
          shmem_A[i][j] = A[row * K + col];
        } else {
          shmem_A[i][j] = __float2half(0.0f);
        }
      }
    }

    // 加载 B 的当前块到共享内存
    #pragma unroll
    for (int i = threadIdx.y; i < TILE_K; i += blockDim.y) {
      #pragma unroll
      for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) {
        int row = k_tile + i;
        int col = warp_n * WMMA_N + j;
        if (row < K && col < N) {
          shmem_B[i][j] = B[row * N + col];
        } else {
          shmem_B[i][j] = __float2half(0.0f);
        }
      }
    }

    __syncthreads();

    // WMMA 矩阵乘累加
    #pragma unroll
    for (int ki = 0; ki < TILE_K; ki += WMMA_K) {
      // 从共享内存加载到 fragment
      load_matrix_sync(a_frag, &shmem_A[0][ki], TILE_K + 8);
      load_matrix_sync(b_frag, &shmem_B[ki][0], TILE_N + 8);

      // 执行矩阵乘累加: C = A * B + C
      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __syncthreads();
  }

  // 存储结果到全局内存
  store_matrix_sync(&C[warp_m * WMMA_M * N + warp_n * WMMA_N], c_frag, N, mem_row_major);
}

void matmul_kernel_cu_fp16(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(input.data_type() == base::DataType::kDataTypeFp16);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(weight.data_type() == base::DataType::kDataTypeFp16);

  CHECK(output.is_empty() == false && output.dims_size() == 1);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(output.data_type() == base::DataType::kDataTypeFp32);

  // 获取矩阵维度
  // input: [M] 或 [batch, M]
  // weight: [K, M] (注意：存储是转置的)
  // output: [K]

  const int32_t M = input.get_dim(input.dims_size() - 1);  // input 的最后一维
  const int32_t K = weight.get_dim(0);                      // weight 的行数 (输出维度)
  CHECK_EQ(M, weight.get_dim(1));                           // weight 的列数必须等于 input 维度

  // WMMA tile 大小
  constexpr int TILE_M = 64;
  constexpr int TILE_N = 64;
  constexpr int TILE_K = 64;

  // Block 和 Grid 配置
  // 每个 block 包含多个 warp，每个 warp 处理一个 16x16 的 WMMA 分块
  dim3 block(32, 4);  // 128 threads = 4 warps
  dim3 grid((K + TILE_M - 1) / TILE_M, (1 + TILE_N - 1) / TILE_N);

  cudaStream_t stream = config ? config->stream : nullptr;

  if (stream) {
    matmul_wmma_fp16_kernel<TILE_M, TILE_N, TILE_K><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(input.ptr<base::float16_t>()),
        reinterpret_cast<const half*>(weight.ptr<base::float16_t>()),
        const_cast<float*>(output.ptr<float>()), static_cast<int>(1), static_cast<int>(K), static_cast<int>(M));
  } else {
    matmul_wmma_fp16_kernel<TILE_M, TILE_N, TILE_K><<<grid, block>>>(
        reinterpret_cast<const half*>(input.ptr<base::float16_t>()),
        reinterpret_cast<const half*>(weight.ptr<base::float16_t>()),
        const_cast<float*>(output.ptr<float>()), static_cast<int>(1), static_cast<int>(K), static_cast<int>(M));
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(FATAL) << "CUDA error in matmul_kernel_cu_fp16: "
               << cudaGetErrorString(err);
  }
}
}  // namespace kernel