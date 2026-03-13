#ifndef __NOZOM1_INCLUDE_BASE_BUFFER_H__
#define __NOZOM1_INCLUDE_BASE_BUFFER_H__
#include <memory>
#include "base/alloc.h"

namespace base
{
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer>
{

private:
    size_t byte_size        = 0;
    void *ptr               = nullptr;
    bool use_externel_      = false;
    DeviceType device_type_ = DeviceType::kCPU;
    std::shared_ptr<DeviceAllocator> allocator_;

public:
    explicit Buffer() = default;

    explicit Buffer(size_t byte_size,
                    std::shared_ptr<DeviceAllocator> allocator = nullptr,
                    void *ptr                                  = nullptr,
                    bool use_external                          = false);
    virtual ~Buffer();

    bool allocate();

    void copy_from(const Buffer &buffer) const;

    void copy_from(const Buffer *buffer) const;

    void *ptr();

    size_t byte_size() const;

    std::shared_ptr<DeviceAllocator> allocator() const;

    DeviceType device_type() const;

    void set_device_type(DeviceType device_type);

    std::shared_ptr<Buffer> get_shared_from_this();

    bool is_external() const;
};
}  // namespace base
#endif