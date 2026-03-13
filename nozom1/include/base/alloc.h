#ifndef __NOZOM1_INCLUDE_BASE_ALLOC_H__
#define __NOZOM1_INCLUDE_BASE_ALLOC_H__
#include <map>
#include <memory>
#include "base.h"
namespace base
{
enum class MemcpyKind
{
    kMemcpyCPU2CPU   = 0,
    kMemcpyCPU2CUDA  = 1,
    kMemcpyCUDA2CPU  = 2,
    kMemcpyCUDA2CUDA = 3,
};

class DeviceAllocator
{
public:
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

    virtual DeviceType deviceType() { return device_type_; }

    virtual void release(void *ptr) const = 0;

    virtual void *allocate(size_t byte_sie) const = 0;

    virtual void memcpy(const void *src,
                        void *dst,
                        size_t byte_size,
                        MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU,
                        void *stream           = nullptr,
                        bool need_sync         = false) const;

    virtual void memset_zero(void *ptr, size_t byte_size, void *stream, bool need_sync = false);

private:
    DeviceType device_type_ = DeviceType::kUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator
{
public:
    explicit CPUDeviceAllocator();
    void *allocate(size_t byte_size) const override;

    void release(void *ptr) const override;
};

struct CudaMemoryBuffer
{
    void *data;
    size_t byte_size;
    bool busy;
    CudaMemoryBuffer() = default;
    CudaMemoryBuffer(void *data, size_t byte_size, bool busy) : data(data), byte_size(byte_size), busy(busy) {}
};

class CUDADeviceAllocator : public DeviceAllocator
{
public:
    explicit CUDADeviceAllocator();

    void *allocate(size_t byte_size) const override;

    void release(void *ptr) const override;

private:
    // 辅助函数
    void *AllocateLargeBuffer(size_t byte_size, int device_id) const;
    void *AllocateSmallBuffer(size_t byte_size, int device_id) const;
    void CleanupIdleBuffers(int device_id) const;

    // NOTE 被const修饰的函数可以修改mutable变量修饰的变量
    mutable std::map<int, size_t> no_busy_cnt_;
    mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
    mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};

class CPUDeviceAllocatorFactory
{
public:
    static std::shared_ptr<CPUDeviceAllocator> get_instance()
    {
        if (instance == nullptr)
        {
            instance = std::make_shared<CPUDeviceAllocator>();
        }
        return instance;
    }

private:
    static std::shared_ptr<CPUDeviceAllocator> instance;
};

class CUDADeviceAllocatorFactory
{
public:
    static std::shared_ptr<CUDADeviceAllocator> get_instance()
    {
        if (instance == nullptr)
        {
            instance = std::make_shared<CUDADeviceAllocator>();
        }
        return instance;
    }

private:
    static std::shared_ptr<CUDADeviceAllocator> instance;
};

}  // namespace base
#endif