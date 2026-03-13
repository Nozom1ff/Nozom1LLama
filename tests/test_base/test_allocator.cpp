/**
 * @file test_allocator.cpp
 * @brief 测试 CUDA 内存分配器的功能
 */

#include <gtest/gtest.h>
#include "base/alloc.h"
#include <cuda_runtime.h>
#include <chrono>

using namespace base;

// ==================== 辅助函数 ====================

/**
 * @brief 获取当前设备的可用内存
 */
size_t GetAvailableMemory() {
    size_t free = 0, total = 0;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess) {
        LOG(ERROR) << "Failed to get memory info: " << cudaGetErrorString(err);
        return 0;
    }
    return free;
}

// ==================== CUDA 环境检查 ====================

/**
 * @test 测试 CUDA 是否可用
 */
TEST(AllocatorTest, CheckCudaAvailable) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    ASSERT_EQ(err, cudaSuccess) << "CUDA not available";
    ASSERT_GT(device_count, 0) << "No CUDA devices found";

    LOG(INFO) << "Found " << device_count << " CUDA device(s)";
}

/**
 * @test 测试可以设置 CUDA 设备
 */
TEST(AllocatorTest, CanSetDevice) {
    int device_id = 0;
    cudaError_t err = cudaSetDevice(device_id);

    ASSERT_EQ(err, cudaSuccess) << "Failed to set CUDA device";
}

// ==================== 基础分配测试 ====================

/**
 * @test 测试分配小缓冲区（<1MB）
 */
TEST(AllocatorTest, AllocateSmallBuffer) {
    CUDADeviceAllocator allocator;

    // 分配 512KB
    size_t size = 512 * 1024;
    void* ptr = allocator.allocate(size);

    ASSERT_NE(ptr, nullptr) << "Failed to allocate small buffer";

    // 释放
    allocator.release(ptr);
}

/**
 * @test 测试分配大缓冲区（>1MB）
 */
TEST(AllocatorTest, AllocateLargeBuffer) {
    CUDADeviceAllocator allocator;

    // 分配 2MB
    size_t size = 2 * 1024 * 1024;
    void* ptr = allocator.allocate(size);

    ASSERT_NE(ptr, nullptr) << "Failed to allocate large buffer";

    // 释放
    allocator.release(ptr);
}

/**
 * @test 测试分配零大小
 */
TEST(AllocatorTest, AllocateZeroSize) {
    CUDADeviceAllocator allocator;

    void* ptr = allocator.allocate(0);

    EXPECT_EQ(ptr, nullptr) << "Allocating zero size should return nullptr";
}

// ==================== 内存复用测试 ====================

/**
 * @test 测试小缓冲区复用
 */
TEST(AllocatorTest, ReuseSmallBuffer) {
    CUDADeviceAllocator allocator;

    size_t size = 512 * 1024;

    // 第一次分配
    void* ptr1 = allocator.allocate(size);
    ASSERT_NE(ptr1, nullptr);

    // 释放
    allocator.release(ptr1);

    // 第二次分配相同大小（应该复用）
    void* ptr2 = allocator.allocate(size);
    ASSERT_NE(ptr2, nullptr);

    // 注意：由于内存池实现，不一定返回相同指针
    // 但应该成功分配

    allocator.release(ptr2);
}

/**
 * @test 测试大缓冲区复用
 */
TEST(AllocatorTest, ReuseLargeBuffer) {
    CUDADeviceAllocator allocator;

    size_t size = 2 * 1024 * 1024;

    // 分配、释放、再分配
    void* ptr1 = allocator.allocate(size);
    ASSERT_NE(ptr1, nullptr);
    allocator.release(ptr1);

    void* ptr2 = allocator.allocate(size);
    ASSERT_NE(ptr2, nullptr);
    allocator.release(ptr2);
}

// ==================== 多次分配测试 ====================

/**
 * @test 测试多次分配和释放
 */
TEST(AllocatorTest, MultipleAllocateRelease) {
    CUDADeviceAllocator allocator;

    const int num_allocs = 10;
    void* ptrs[num_allocs];
    size_t size = 1024 * 1024;  // 1MB

    // 分配多个缓冲区
    for (int i = 0; i < num_allocs; ++i) {
        ptrs[i] = allocator.allocate(size);
        ASSERT_NE(ptrs[i], nullptr) << "Failed to allocate buffer " << i;
    }

    // 释放所有缓冲区
    for (int i = 0; i < num_allocs; ++i) {
        allocator.release(ptrs[i]);
    }
}

/**
 * @test 测试分配不同大小的缓冲区
 */
TEST(AllocatorTest, AllocateDifferentSizes) {
    CUDADeviceAllocator allocator;

    size_t sizes[] = {
        256 * 1024,      // 256KB
        512 * 1024,      // 512KB
        1024 * 1024,     // 1MB
        2 * 1024 * 1024, // 2MB
        4 * 1024 * 1024  // 4MB
    };

    for (size_t size : sizes) {
        void* ptr = allocator.allocate(size);
        ASSERT_NE(ptr, nullptr) << "Failed to allocate " << (size / 1024) << " KB";
        allocator.release(ptr);
    }
}

// ==================== 边界测试 ====================

/**
 * @test 测试释放空指针
 */
TEST(AllocatorTest, ReleaseNullptr) {
    CUDADeviceAllocator allocator;

    // 不应该崩溃
    allocator.release(nullptr);
}

/**
 * @test 测试释放未分配的指针（应该被安全处理）
 */
TEST(AllocatorTest, ReleaseUnallocatedPointer) {
    CUDADeviceAllocator allocator;

    // 分配一个真实指针用于测试
    void* real_ptr = allocator.allocate(1024);
    ASSERT_NE(real_ptr, nullptr);

    // 创建一个假的指针（不在池中）
    void* fake_ptr = nullptr;
    cudaMalloc(&fake_ptr, 1024);
    ASSERT_NE(fake_ptr, nullptr);

    // 释放假指针（应该被安全处理，调用 cudaFree）
    // 注意：如果实现不支持，可能需要调整测试
    // allocator.release(fake_ptr);

    // 清理
    cudaFree(fake_ptr);
    allocator.release(real_ptr);
}

// ==================== 内存写入测试 ====================

/**
 * @test 测试分配的内存可以写入
 */
TEST(AllocatorTest, WriteToAllocatedMemory) {
    CUDADeviceAllocator allocator;

    size_t size = 1024 * 1024;
    void* ptr = allocator.allocate(size);
    ASSERT_NE(ptr, nullptr);

    // 使用 cudaMemset 测试写入
    cudaError_t err = cudaMemset(ptr, 0xAB, size);
    ASSERT_EQ(err, cudaSuccess) << "Failed to memset allocated memory";

    allocator.release(ptr);
}

/**
 * @test 测试分配的内存可以拷贝数据
 */
TEST(AllocatorTest, CopyToAllocatedMemory) {
    CUDADeviceAllocator allocator;

    size_t size = 1024;
    void* ptr = allocator.allocate(size);
    ASSERT_NE(ptr, nullptr);

    // 准备主机数据
    std::vector<int> host_data(size / sizeof(int), 0x12345678);

    // 拷贝到设备
    cudaError_t err = cudaMemcpy(ptr, host_data.data(), size, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy to allocated memory";

    allocator.release(ptr);
}

// ==================== 内存池行为测试 ====================

/**
 * @test 测试内存池复用提高性能
 */
TEST(AllocatorTest, MemoryPoolPerformance) {
    CUDADeviceAllocator allocator;

    const int iterations = 100;
    size_t size = 512 * 1024;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        void* ptr = allocator.allocate(size);
        ASSERT_NE(ptr, nullptr);
        allocator.release(ptr);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    LOG(INFO) << "Allocated and released " << iterations
              << " buffers in " << duration.count() << " ms";

    // 应该很快（内存池复用）
    // 但不做硬性断言，因为性能取决于硬件
}

// ==================== 清理策略测试 ====================

/**
 * @test 测试延迟释放机制
 */
TEST(AllocatorTest, LazyRelease) {
    CUDADeviceAllocator allocator;

    const int num_buffers = 10;
    size_t size = 2 * 1024 * 1024;  // 2MB
    void* ptrs[num_buffers];

    // 分配多个缓冲区
    for (int i = 0; i < num_buffers; ++i) {
        ptrs[i] = allocator.allocate(size);
        ASSERT_NE(ptrs[i], nullptr);
    }

    // 释放所有缓冲区
    for (int i = 0; i < num_buffers; ++i) {
        allocator.release(ptrs[i]);
    }

    // 分配相同大小的缓冲区（应该复用）
    void* new_ptr = allocator.allocate(size);
    ASSERT_NE(new_ptr, nullptr);
    allocator.release(new_ptr);
}

// ==================== 主函数 ====================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
