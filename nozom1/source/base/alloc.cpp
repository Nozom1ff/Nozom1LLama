#include "base/alloc.h"
#include <cuda_runtime_api.h>
namespace base
{
void DeviceAllocator::memcpy(const void *src,
                             void *dst,
                             size_t byte_size,
                             MemcpyKind memcpy_kind,
                             void *stream,
                             bool need_sync) const
{
    CHECK_NE(src, nullptr);
    CHECK_NE(dst, nullptr);
    if (!byte_size)
        return;

    cudaStream_t stream_ = nullptr;

    // NOTE 启用cudsaStream进行异步传输
    if (stream)
        stream_ = static_cast<CUstream_st *>(steram);

    if (memcpu_kind == MemcpyKind::kMemcpyCPU2CPU)
    {
        std::memcpy(dst, src, byte_size);
    }
    else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA)
    {
        if (!stream_)
        {
            cudaMemcpy(dst, src, byte_size, cudaMemcpyHostToDevice);
        }
        else
        {
            cudaMemcpy(dst, src, byte_size, cudaMemcpyHostToDevice, stream_);
        }
    }
    else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU)
    {
        if (!stream_)
        {
            cudaMemcpy(dst, src, byte_size, cudaMemcpyDeviceToHost);
        }
        else
        {
            cudaMemcpy(dst, src, byte_size, cudaMemcpyDeviceToHost, stream_);
        }
    }
    else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA)
    {
        if (!stream_)
        {
            cudaMemcpy(dst, src, byte_size, cudaMemcpyDeviceToDevice);
        }
        else
        {
            cudaMemcpy(dst, src, byte_size, cudaMemcpyDeviceToDevice, stream_);
        }
    }
    else
    {
        LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
    }
    if (need_sync)
    {
        cudaDeviceSynchronize();
    }
}

void DeviceAllocator::memset_zero(void *ptr, size_t byte_size, void *stream, bool need_sync)
{
    CHECK(device_type_ != base::DeviceType::Unknown);
    if (device_type == base::DeviceType::kDeviceCPU)
    {
        std::memset(ptr, 0, byte_size);
    }
    else
    {
        if (stream)
        {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            cudaMemsetAsync(ptr, 0, byte_size, stream_);
        }
        else
        {
            cudaMemset(ptr, 0, byte_size);
        }
        if (need_sync)
        {
            cudaDeviceSynchronize();
        }
    }
}

}  // namespace base