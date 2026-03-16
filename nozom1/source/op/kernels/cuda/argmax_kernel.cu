#include "argmax_kernel.cuh"
#include "base/alloc.h"

namespace kernel
{
__forceinline__ __device__ void warp_reduce_argmax(float &val, size_t &ptr)
{
    float tmp_val;
    size_t tmp_ptr;
    int mask = __ballot_sync(0xFFFFFFFF, true);
    for (int k = (warpSize >> 1); k > 0; k >>= 1)
    {
        tmp_val = __shfl_down_sync(mask, val, k, warpSize);
        tmp_ptr = __shfl_down_sync(mask, ptr, k, warpSize);
        if (val < tmp_val)
        {
            val = tmp_val;
            ptr = tmp_ptr;
        }
        else if (tmp_val == val && tmp_ptr < ptr)
        {
            ptr = tmp_ptr;
        }
    }
}

__forceinline__ __device__ void block_reduce_argmax(float &val, size_t &ptr, float *shared_value, size_t *shared_ptr)
{
    // NOTE 传入指针以让device函数访问共享内存
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    warp_reduce_argmax(val, ptr);
    __syncthreads();
    if (lane_id == 0)
    {
        shared_value[warp_id] = val;
        shared_ptr[warp_id]   = ptr;
    }
    __syncthreads();
    if (threadIdx.x < blockDim.x / warpSize)  // < 256/32
    {
        val = shared_value[threadIdx.x];
        ptr = shared_ptr[threadIdx.x];
    }
    else
    {
        val = 0;
        ptr = SIZE_MAX;
    }
    // NOTE 线程 A 写共享内存 → 线程 B 读该共享内存 → 必须加 __syncthreads()； 这里不需要
    if (warp_id == 0)
    {
        warp_reduce_argmax(val, ptr);
    }
}

// [x] 用多个block而不是一个 所以需要一个额外的数组存储
// DONE
__global__ void argmax_kernel_fp32_block(const float *input_ptr, size_t size, float *temp_max_val, size_t *temp_max_ptr)
{
    __shared__ size_t shared_max_ptr[32];
    __shared__ float shared_max_value[32];
    int tid        = threadIdx.x;
    int bid        = blockIdx.x;
    int global_tid = threadIdx.x + blockDim.x * blockIdx.x;

    // 初始化为无效值
    size_t max_index = SIZE_MAX;
    float max_value  = -INFINITY;

    // Grid-stride 循环处理所有元素
    for (size_t i = global_tid; i < size; i += gridDim.x * blockDim.x)
    {
        if (input_ptr[i] > max_value)
        {
            max_index = i;
            max_value = input_ptr[i];
        }
        else if (input_ptr[i] == max_value && i < max_index)
        {
            max_index = i;
        }
    }
    block_reduce_argmax(max_value, max_index, shared_max_value, shared_max_ptr);

    // 每个 block 的 thread 0 写入局部结果到 temp 数组
    if (tid == 0)
    {
        temp_max_val[bid] = max_value;
        temp_max_ptr[bid] = max_index;
    }
}

// NOTE 这里还是得用一个blcok
__global__ void argmax_kernel_fp32(const float *temp_max_val,
                                   const size_t *temp_max_ptr,
                                   size_t num_blocks,
                                   size_t *output_idx)
{
    __shared__ size_t shared_max_ptr[32];
    __shared__ float shared_max_value[32];
    uint32_t tid = threadIdx.x;

    // 初始化：读取对应 block 的局部 argmax
    size_t max_index = SIZE_MAX;
    float max_value  = -INFINITY;

    // 修复核心：用 grid-stride 循环遍历所有 temp 数组元素（步长=blockDim.x）
    for (size_t i = tid; i < num_blocks; i += blockDim.x)
    {
        // 跳过无效的 block 结果（比如 block 处理的元素数为 0）
        if (temp_max_ptr[i] == SIZE_MAX)
        {
            continue;
        }

        float curr_val  = temp_max_val[i];
        size_t curr_ptr = temp_max_ptr[i];
        if (curr_val > max_value)
        {
            max_value = curr_val;
            max_index = curr_ptr;
        }
        else if (curr_val == max_value && curr_ptr < max_index)
        {
            max_index = curr_ptr;
        }
    }
    // block 内规约：得到全局 argmax
    block_reduce_argmax(max_value, max_index, shared_max_value, shared_max_ptr);

    // thread 0 写入最终结果
    if (tid == 0)
    {
        // 如果没有找到有效值（所有 block 都是 SIZE_MAX），返回 0 作为后备
        *output_idx = (max_index == SIZE_MAX) ? 0 : max_index;
    }
}

size_t argmax_kernel_cu(const float *input_ptr, size_t size, void *stream)
{
    std::shared_ptr<base::DeviceAllocator> alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

    size_t *index           = static_cast<size_t *>(alloc_cu->allocate(sizeof(size_t)));
    const int max_grid_size = 65535;  // NOTE 限制避免temp过大
    int grid_size           = min((int)((size + 512 - 1) / 512), max_grid_size);

    float *temp_val  = static_cast<float *>(alloc_cu->allocate(sizeof(float) * grid_size));
    size_t *temp_ptr = static_cast<size_t *>(alloc_cu->allocate(sizeof(size_t) * (grid_size)));

    // 初始化 temp 数组（使用 CUDA memset）
    cudaMemset(temp_val, 0, sizeof(float) * grid_size);
    cudaMemset(temp_ptr, 0xFF, sizeof(size_t) * grid_size);  // 设置为 SIZE_MAX

    size_t output_index = 0;

    if (!stream)
    {
        argmax_kernel_fp32_block<<<grid_size, 512>>>(input_ptr, size, temp_val, temp_ptr);
        argmax_kernel_fp32<<<1, 512>>>(temp_val, temp_ptr, grid_size, index);
        cudaMemcpy(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost);
    }
    else
    {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        argmax_kernel_fp32_block<<<grid_size, 512, 0, stream_>>>(input_ptr, size, temp_val, temp_ptr);
        argmax_kernel_fp32<<<1, 512, 0, stream_>>>(temp_val, temp_ptr, grid_size, index);
        cudaMemcpyAsync(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    }

    return output_index;
}



}  // namespace kernel