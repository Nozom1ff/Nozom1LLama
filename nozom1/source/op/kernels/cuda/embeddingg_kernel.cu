#include "emb_kernel.cuh"

// NOTE 为什么不需要手动free input_cu
/**
 *   input_cu 是栈上分配的局部对象，而非通过 new 分配的堆对象。
  自动内存管理链
  当 emb_kernel_cu 函数返回时：
  1. input_cu 析构函数被自动调用（C++ RAII 机制）
  2. Tensor 内部使用 shared_ptr<Buffer> 管理 buffer：
  // tensor.h 中
  std::shared_ptr<base::Buffer> buffer_;
  3. Buffer 析构函数自动释放内存（buffer.cpp:15-22）：
  Buffer::~Buffer() {
    if (!use_external_) {
      if (ptr_ && allocator_) {
        allocator_->release(ptr_);  // 自动调用 allocator 释放
        ptr_ = nullptr;
      }
    }
  }
-------------------------------------------------------------
  to_cuda() 的内存转换
  在 tensor.cpp:104-119 的 to_cuda() 中：
  - 第112行创建新的 CUDA buffer
  - 第115行 this->buffer_ = cu_buffer; 替换旧 buffer
  - 原来的 CPU buffer 引用计数减少，如无人引用则自动释放
 */

namespace kernel
{

__global__ void emb_kernel_cu_fp32(int32_t vocab_size,
                                   int32_t token_num,
                                   int32_t weight_dim,
                                   const int32_t *input_ptr,
                                   const float *weight_ptr,
                                   float *output_ptr)
{
    int32_t token_idx = blockIdx.x;
    if (token_idx >= token_num)
    {
        return;
    }
    int32_t token = input_ptr[token_idx];
    if (token >= vocab_size)
    {
        return;
    }

    float *output_ptr_start       = output_ptr + token_idx * weight_dim;
    const float *weight_ptr_start = weight_ptr + token * weight_dim;
    // TODO 向量化访存 & __half
    for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x)
    {
        output_ptr_start[i] = weight_ptr_start[i];
    }
}

// BUG input_cu 没有考虑一开始input就在cuda上的情况
void emb_kernel_cu(const tensor::Tensor &input,
                   const tensor::Tensor &weight,
                   const tensor::Tensor &output,
                   int32_t vocab_size,
                   void *stream)
{
    tensor::Tensor input_cu;
    const tensor::Tensor *input_ptr;
    if (input.device_type() != base::DeviceType::kCUDA)
    {
        input_cu = input.clone();
        input_cu.to_cuda();
        input_ptr = &input_cu;
    }
    else
    {
        input_ptr = &input;
    }
    const int32_t input_num  = static_cast<int32_t>(input.size());
    const int32_t weight_dim = weight.get_dim(1);
    CHECK(weight.device_type() == output.device_type());
    CHECK(output.device_type() == base::DeviceType::kCUDA);

    constexpr int32_t max_seq_len = 512;
    constexpr int32_t thread_num  = 256;

    const int32_t *in_ptr = input_ptr->ptr<int32_t>();
    const float *wei_ptr  = weight.ptr<float>();
    float *out_ptr        = const_cast<float *>(output.ptr<float>());
    if (stream)
    {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        emb_kernel_cu_fp32<<<max_seq_len, thread_num, 0, stream_>>>(
            vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
    }
    else
    {
        emb_kernel_cu_fp32<<<max_seq_len, thread_num>>>(vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
    }
}
}  // namespace kernel