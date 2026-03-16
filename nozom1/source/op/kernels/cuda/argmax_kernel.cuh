#ifndef __ARGMAX_KERNEL_CUH__
#define __ARGMAX_KERNEL_CUH__

namespace kernel
{
size_t argmax_kernel_cu(const float *input_ptr, size_t size, void *stream);
}
#endif  // ARGMAX_KERNEL_CUH
