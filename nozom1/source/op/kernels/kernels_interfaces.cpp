#include <base/base.h>
#include "cuda/add_kernel.cuh"
#include "kernels_interfaces.h"

namespace kernel
{
AddKernel get_add_kernel(base::DeviceType device_type)
{
    if (device_type == base::DeviceType::kCPU)
    {
        LOG(FATAL) << "CPU Operator is not supported!";
        return nullptr;
    }
    else if (device_type == base::DeviceType::kCUDA)
    {
        return add_kernel_cu;
    }
    else
    {
        LOG(FATAL) << "Unknown device type for get a add kernel.";
        return nullptr;
    }
}
}  // namespace kernel