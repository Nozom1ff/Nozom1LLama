#include "kernels_interfaces.h"
#include <base/base.h>
#include "cuda/add_kernel.cuh"
#include "cuda/embedding_kernel.cuh"
#include "cuda/swiglu_kernel.cuh"
#include "cuda/rmsnorm_kernel.cuh"

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

EmbeddingKernel get_emb_kernel(base::DeviceType device_type)
{
    if (device_type == base::DeviceType::kCPU)
    {
        LOG(FATAL) << "CPU Operator is not supported!";
        return nullptr;
    }
    else if (device_type == base::DeviceType::kCUDA)
    {
        return emb_kernel_cu;
    }
    else
    {
        LOG(FATAL) << "Unknown device type for get an embedding kernel.";
        return nullptr;
    }
}

SwigluKernel get_swiglu_kernel(base::DeviceType device_type, void *stream)
{
    if (device_type == base::DeviceType::kCPU)
    {
        LOG(FATAL) << "CPU Operator is not supported!";
        return nullptr;
    }
    else if (device_type == base::DeviceType::kCUDA)
    {
        return swiglu_kernel_cu;
    }
    else
    {
        LOG(FATAL) << "Unknown device type for get a swiglu kernel.";
        return nullptr;
    }
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type)
{
    if (device_type == base::DeviceType::kCPU)
    {
        LOG(FATAL) << "CPU Operator is not supported!";
        return nullptr;
    }
    else if (device_type == base::DeviceType::kCUDA)
    {
        return rmsnorm_kernel_cu;
    }
    else
    {
        LOG(FATAL) << "Unknown device type for get a rmsnorm kernel.";
        return nullptr;
    }
}

RMSNormKernelDim get_rmsnorm_dim_kernel(base::DeviceType device_type)
{
    if (device_type == base::DeviceType::kCPU)
    {
        LOG(FATAL) << "CPU Operator is not supported!";
        return nullptr;
    }
    else if (device_type == base::DeviceType::kCUDA)
    {
        return rmsnorm_kernel_cu_dim;
    }
    else
    {
        LOG(FATAL) << "Unknown device type for get a rmsnorm dim kernel.";
        return nullptr;
    }
}
}  // namespace kernel