#include <tensor/tensor.h>
#include "swiglu_kernel.cuh"
#define MAX_EXP_F32 88.3762626647949f  // 32位float指数最大
#define MIN_EXP_F32 -88.3762626647949f

namespace kernel
{
__global__ void swiglu_kernel_cu_fp32(int size, const float *A, const float *B, float *C)
{
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
    {
        return;
    }
    extern __shared__ float shared_mem[];
    float *smemA = shared_mem;
    float *smemB = shared_mem + blockDim.x;  // 总长度为 2*blockDim.x
    smemA[tid]   = A[idx];
    smemB[tid]   = B[idx];
    __syncthreads();
    float value = fminf(fmaxf(smemA[tid], MIN_EXP_F32), MAX_EXP_F32);
    value       = 1.0f / (1.0f + exp(-value));
    smemA[tid] *= value;
    C[idx] = smemA[tid] * smemB[tid];
}

void swiglu_kernel_cu(const tensor::Tensor &input1,
                      const tensor::Tensor &input2,
                      const tensor::Tensor &output,
                      void *stream)
{
    CHECK_EQ(input1.is_empty(), false);
    CHECK(input1.device_type() == base::DeviceType::kCUDA);

    CHECK_EQ(input2.is_empty(), false);
    CHECK(input2.device_type() == base::DeviceType::kCUDA);

    CHECK_EQ(output.is_empty(), false);
    CHECK(output.device_type() == base::DeviceType::kCUDA);
    int size           = static_cast<int32_t>(input1.size());
    int threads        = 128;
    int blocks         = (size + threads - 1) / threads;
    const size_t shmem = threads * sizeof(float) * 2;
    if (!stream)
    {
        swiglu_kernel_cu_fp32<<<blocks, threads, shmem>>>(
            size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float *>(output.ptr<float>()));
    }
    else
    {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        swiglu_kernel_cu_fp32<<<blocks, threads, shmem, stream_>>>(
            size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float *>(output.ptr<float>()));
    }
}
}  // namespace kernel