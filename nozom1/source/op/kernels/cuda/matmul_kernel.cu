#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interfaces.h"
#include "matmul_kernel.cuh"

// [] 优化gev
namespace kernel
{
/**
 * @brief gemv
 * @param M input 一维张量的长度
 * @param K output 长度
 * @param weight [K,M]
 * @note 现在一个block处理一条长维为M的张量，总共K个BLOCK，所有线程都在处理M内的数据
 */
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float *input, const float *weight, float *output, int M, int K)
{
    __shared__ float sdata[THREAD_PER_BLOCK];
    int tid = threadIdx.x;

    int start_row = blockIdx.x * ROW_PER_BLOCK;  // NOTE 其实没有用到 可能他想一个block处理多个连续列
    int end_row   = start_row + ROW_PER_BLOCK;
    if (start_row >= K)
    {
        return;
    }
    const int pack_num = M / 4;
    const int pack_off = 4 * pack_num;
#pragma unroll
    for (int p = start_row; p < end_row; ++p)
    {
        sdata[tid]                = 0;
        int row_offset            = p * M;
        float4 *input_float4_ptr  = (float4 *)input;
        float4 *weight_float4_ptr = (float4 *)(weight + row_offset);

#pragma unroll
        for (int i = tid; i < pack_num; i += blockDim.x)
        {
            float4 in      = *(input_float4_ptr + i);
            float4 wei     = *(weight_float4_ptr + i);
            float part_sum = in.x * wei.x + in.y * wei.y + in.z * wei.z + in.w * wei.w;
            sdata[tid] += part_sum;
        }
        for (int i = pack_off + tid; i < M; i += blockDim.x)
        {
            sdata[tid] += input[i] * weight[row_offset + i];
        }
        __syncthreads();

        using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;
        float part_sum = BlockReduce(temp).Sum(sdata[tid]);
        __syncthreads();
        if (tid == 0)
        {
            output[p] = part_sum;
        }
        __syncthreads();  // NOTE 兼容 ROW_PER_BLOCK>1
    }
}

// NOTE 这里有个scale ，同时weight变为int8
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float *input,
                                          const int8_t *weight,
                                          const float *scales,
                                          const int32_t group_size,
                                          float *output,
                                          int M,
                                          int K)
{
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    int start_row = blockIdx.x * ROW_PER_BLOCK;
    int end_row   = start_row + ROW_PER_BLOCK;
    if (start_row >= K)
    {
        return;
    }
    for (int p = start_row; p < end_row; ++p)
    {
        sdata[tid] = 0;
        for (int i = tid; i < M; i += THREAD_PER_BLOCK)
        {
            const int weight_idx = p * M + i;
            const int group_idx  = weight_idx / group_size;
            sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
        }
        __syncthreads();

        using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp;
        float part_sum = BlockReduce(temp).Sum(sdata[tid]);
        __syncthreads();

        if (tid == 0)
        {
            output[p] = part_sum;
        }
        __syncthreads();
    }
}

void matmul_kernel_cu(const tensor::Tensor &input,
                      const tensor::Tensor &weight,
                      const tensor::Tensor &output,
                      const float scale,
                      const CudaConfig *config)
{
    CHECK(input.is_empty() == false && input.dims_size() <= 2);
    CHECK(input.device_type() == base::DeviceType::kCUDA);

    CHECK(weight.is_empty() == false && weight.dims_size() == 2);
    CHECK(weight.device_type() == base::DeviceType::kCUDA);
    const int32_t K = weight.get_dim(0);  // row
    const int32_t M = weight.get_dim(1);  // col

    // CHECK_EQ(M % packet_size, 0);

    CHECK_EQ(M, input.get_dim(0));
    if (config && config->stream)
    {
        matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, config->stream>>>(
            input.ptr<float>(), weight.ptr<float>(), const_cast<float *>(output.ptr<float>()), M, K);
    }
    else
    {
        matmul_kernel_cu_fp32<128, 1>
            <<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(), const_cast<float *>(output.ptr<float>()), M, K);
    }
}


}  // namespace kernel