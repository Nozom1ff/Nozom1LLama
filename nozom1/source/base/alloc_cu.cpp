#include <cuda_runtime_api.h>
#include "base/alloc.h"
namespace base
{

// 常量定义
namespace {
constexpr size_t kSmallBufferThreshold = 1024 * 1024;         // 1MB
constexpr size_t kMaxWasteSize          = 1 * 1024 * 1024;     // 1MB
constexpr size_t kCleanupThreshold       = 1024 * 1024 * 1024;  // 1GB
constexpr size_t kMB                    = 1024 * 1024;

struct BufferSearchResult
{
    void *data;
    size_t index;
    bool found;
};

/**
 * @brief cuda统一状态检查
 */
void *CudaMallocChecked(size_t byte_size, int device_id)
{
    void *ptr       = nullptr;
    cudaError_t err = cudaMalloc(&ptr, byte_size);
    if (err != cudaSuccess)
    {
        LOG(ERROR) << "CUDA error: Failed to allocate " << (byte_size / kMB) << " MB on device " << device_id
                   << ". Error: " << cudaGetErrorString(err);
        return nullptr;
    }
    return ptr;
}

void CudaFreeChecked(void *ptr, int device_id)
{
    if (!ptr)
        return;
    cudaSetDevice(device_id);
    cudaError_t err = cudaFree(ptr);
    CHECK(err == cudaSuccess) << "CUDA error: Failed to free memory on device " << device_id << ": "
                              << cudaGetErrorString(err);
}

/**
 * @brief 查找空闲缓冲区
 * @param buffers 可以是big 也可以是普通
 * @param use_best_fit: true=Best-Fit(大对象) false=First-Fit(小对象)
 */
BufferSearchResult FindFreeBuffer(std::vector<CudaMemoryBuffer> &buffers, size_t required_size, bool use_best_fit = false)
{
    BufferSearchResult result{nullptr, size_t(-1), false};
    for (size_t i = 0; i < buffers.size(); ++i)
    {
        auto &buffer = buffers[i];
        // NOTE 检查大小是否满足 同时不能浪费
        if (buffer.byte_size >= required_size && !buffer.busy)
        {
            size_t waste = buffer.byte_size - required_size;
            if (waste > kMaxWasteSize)
                continue;

            // 选择策略
            if (!result.found)
            {
                result = {buffer.data, i, true};
            }
            else if (use_best_fit)  // NOTE 找到了但还接着找
            {
                // Best-Fit: 选择最小的
                if (buffer.byte_size < buffers[result.index].byte_size)
                {
                    result = {buffer.data, i, true};
                }
            }
            else
            {
                // First-Fit: 找到第一个就停止
                result = {buffer.data, i, true};
                break;
            }
        }
    }
    return result;
}

/**
 * @brief 分配新缓冲区并加入池
 */
void *AllocateAndAddToPool(std::vector<CudaMemoryBuffer> &buffers, size_t byte_size, int device_id)
{
    void *ptr = CudaMallocChecked(byte_size, device_id);
    if (ptr)
    {
        buffers.emplace_back(ptr, byte_size, true);
    }
    return ptr;
}

int GetCurrentDeviceId()
{
    int device_id   = -1;
    cudaError_t err = cudaGetDevice(&device_id);
    CHECK(err == cudaSuccess) << "Failed to get current CUDA device";
    return device_id;
}

}  // anonymous namespace

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kCUDA) {}

void *CUDADeviceAllocator::allocate(size_t byte_size) const
{
    if (byte_size == 0)
    {
        return nullptr;
    }
    const int device_id = GetCurrentDeviceId();
    if (byte_size > kSmallBufferThreshold)
    {
        return AllocateLargeBuffer(byte_size, device_id);
    }
    else
    {
        return AllocateSmallBuffer(byte_size, device_id);
    }
}
void *CUDADeviceAllocator::AllocateLargeBuffer(size_t byte_size, int device_id) const
{
    auto &big_buffers = big_buffers_map_[device_id];

    // Best-Fit 查找
    BufferSearchResult result = FindFreeBuffer(big_buffers, byte_size, true);
    if (result.found)
    {
        big_buffers[result.index].busy = true;
        return big_buffers[result.index].data;
    }

    // 没找到，分配新块
    return AllocateAndAddToPool(big_buffers, byte_size, device_id);
}
void *CUDADeviceAllocator::AllocateSmallBuffer(size_t byte_size, int device_id) const
{
    auto &cuda_buffers = cuda_buffers_map_[device_id];

    // First-Fit 查找
    BufferSearchResult result = FindFreeBuffer(cuda_buffers, byte_size, false);
    if (result.found)
    {
        cuda_buffers[result.index].busy = true;
        no_busy_cnt_[device_id] -= cuda_buffers[result.index].byte_size;
        return cuda_buffers[result.index].data;
    }

    // 没找到，分配新块
    return AllocateAndAddToPool(cuda_buffers, byte_size, device_id);
}

void CUDADeviceAllocator::release(void *ptr) const
{
    if (!ptr)
    {
        return;
    }

    if (cuda_buffers_map_.empty())
    {
        return;
    }

    // 清理空闲内存（延迟释放）
    for (auto &it : cuda_buffers_map_)
    {
        if (no_busy_cnt_[it.first] > kCleanupThreshold)
        {
            CleanupIdleBuffers(it.first);
        }
    }

    // 尝试归还到池
    for (auto &it : cuda_buffers_map_)
    {
        auto &cuda_buffers = it.second;
        for (size_t i = 0; i < cuda_buffers.size(); ++i)
        {
            if (cuda_buffers[i].data == ptr)
            {
                no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
                cuda_buffers[i].busy = false;
                return;
            }
        }

        // 检查大缓冲区池
        auto &big_buffers = big_buffers_map_[it.first];
        for (size_t i = 0; i < big_buffers.size(); ++i)
        {
            if (big_buffers[i].data == ptr)
            {
                big_buffers[i].busy = false;
                return;
            }
        }
    }

    // 不在池中，直接释放
    int device_id = GetCurrentDeviceId();
    CudaFreeChecked(ptr, device_id);
}

void CUDADeviceAllocator::CleanupIdleBuffers(int device_id) const
{
    auto &cuda_buffers = cuda_buffers_map_[device_id];
    std::vector<CudaMemoryBuffer> active_buffers;

    cudaSetDevice(device_id);
    for (size_t i = 0; i < cuda_buffers.size(); ++i)
    {
        if (cuda_buffers[i].busy)
        {
            active_buffers.push_back(cuda_buffers[i]);
        }
        else
        {
            CudaFreeChecked(cuda_buffers[i].data, device_id);
        }
    }

    cuda_buffers.clear();
    cuda_buffers            = std::move(active_buffers);
    no_busy_cnt_[device_id] = 0;
}

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

}  // namespace base
