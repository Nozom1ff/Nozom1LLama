#include "add_kernel.cuh"

namespace kernel
{
// TODO 向量化访存
// NOTE 优化 单线程多元素，grid loop 是对的！
__global__ void add_kernel_cu_fp32(size_t N, const float *a, const float *b, float *c)
{
    // 1. 计算当前线程的基础全局索引
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 2. grid-stride 循环：步长=总线程数（gridDim.x * blockDim.x）
    //    自动覆盖所有元素，无需关心 N 是否为线程数的整数倍
    for (size_t i = global_idx; i < N; i += gridDim.x * blockDim.x)
    {
        // 3. 边界检查（双重保险，循环条件已保证 i<N，但建议保留）
        if (i < N)
        {
            c[i] = a[i] + b[i];
        }
    }
}

// TODO fp16支持
void add_kernel_cu(const tensor::Tensor &input1,
                   const tensor::Tensor &input2,
                   const tensor::Tensor &output,
                   void *stream)
{
    CHECK_EQ(input1.is_empty(), false);
    CHECK_EQ(input2.is_empty(), false);
    CHECK_EQ(output.is_empty(), false);
    size_t size = input1.size();
    CHECK_EQ(size, input2.size());
    CHECK_EQ(size, output.size());
    int32_t thread_num = 256;
    int32_t block_num  = (size + thread_num - 1) / thread_num;
    if (stream)
    {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        add_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
            size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float *>(output.ptr<float>()));
    }
}

}  // namespace kernel