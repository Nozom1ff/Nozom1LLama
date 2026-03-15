#ifndef __NOZOM1_CUDA_EMBEDDING_KERNEL_H__
#define __NOZOM1_CUDA_EMBEDDING_KERNEL_H__

#include "tensor/tensor.h"

namespace kernel
{
void emb_kernel_cu(const tensor::Tensor &input,
                   const tensor::Tensor &weight,
                   const tensor::Tensor &output,
                   int32_t vocab_size,
                   void *stream);
}  // namespace kernel

#endif  // __NOZOM1_CUDA_EMBEDDING_KERNEL_H__
