#include <sys/types.h>
#include <iostream>
#include <torch/extension.h>

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

// SHMEM头文件
#include "shmem/host/init/aclshmem_init.h"
#include "shmem/host/mem/aclshmem_mem.h"

extern "C" {
using c10_npu::NPUCachingAllocator::DeviceStats;
static bool shmem_allocator_initialized = false;
static bool shmem_allocator_active = false;

// SHMEM分配器的malloc函数
void* shmem_malloc(ssize_t size, int device, aclrtStream stream)
{
    if (!shmem_allocator_initialized) {
        // 初始化SHMEM分配器
        aclshmemx_init_attr_t attr = {0};
        attr.version = 1;
        attr.my_rank = 0;  // 单进程场景
        attr.n_ranks = 1;  // 单进程场景
        attr.local_mem_size = 1024 * 1024 * 1024; // 1GB共享内存
        
        int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
        if (ret != 0) {
            std::cerr << "SHMEM allocator initialization failed with code: " << ret << std::endl;
            // 回退到标准ACL分配
            void *fallback_ptr;
            aclrtMallocAlign32(&fallback_ptr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
            return fallback_ptr;
        }
        shmem_allocator_initialized = true;
        std::cout << "SHMEM allocator initialized successfully" << std::endl;
    }
    
    // 使用SHMEM分配内存
    void *ptr = aclshmem_malloc(size);
    if (ptr == nullptr) {
        std::cerr << "SHMEM malloc failed for size: " << size << std::endl;
        // 回退到标准ACL分配
        void *fallback_ptr;
        aclrtMallocAlign32(&fallback_ptr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
        return fallback_ptr;
    }
    
    shmem_allocator_active = true;
    std::cout << "SHMEM alloc ptr = " << ptr << ", size = " << size << std::endl;
    return ptr;
}

// SHMEM分配器的free函数
void shmem_free(void* ptr, ssize_t size, int device, aclrtStream stream)
{
    if (shmem_allocator_active && aclshmem_ptr_valid(ptr)) {
        std::cout << "SHMEM free ptr = " << ptr << ", size = " << size << std::endl;
        aclshmem_free(ptr);
    } else {
        std::cout << "Standard ACL free ptr = " << ptr << ", size = " << size << std::endl;
        aclrtFree(ptr);
    }
}

// 检查是否使用了SHMEM分配器
bool check_shmem_allocator_used()
{
    return shmem_allocator_active;
}

// 获取设备统计信息
DeviceStats shmem_get_device_stats(int device)
{
    DeviceStats stats = {0};
    // 这里可以添加SHMEM特定的统计信息
    // 暂时返回空统计
    return stats;
}

// 重置峰值状态
void shmem_reset_peak_status(int device)
{
    std::cout << "SHMEM reset peak status for device " << device << std::endl;
    // 这里可以添加SHMEM特定的重置逻辑
}

// 初始化函数
void shmem_init_allocator()
{
    std::cout << "Initializing SHMEM pluggable allocator" << std::endl;
}

// 清理函数
void shmem_cleanup_allocator()
{
    if (shmem_allocator_initialized) {
        aclshmem_finalize();
        shmem_allocator_initialized = false;
        std::cout << "SHMEM allocator finalized" << std::endl;
    }
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shmem_malloc", &shmem_malloc, "SHMEM memory allocation function");
    m.def("shmem_free", &shmem_free, "SHMEM memory deallocation function");
    m.def("check_shmem_allocator_used", &check_shmem_allocator_used, "Check if SHMEM allocator was used");
    m.def("shmem_get_device_stats", &shmem_get_device_stats, "Get SHMEM device statistics");
    m.def("shmem_reset_peak_status", &shmem_reset_peak_status, "Reset SHMEM peak status");
    m.def("shmem_init_allocator", &shmem_init_allocator, "Initialize SHMEM allocator");
    m.def("shmem_cleanup_allocator", &shmem_cleanup_allocator, "Cleanup SHMEM allocator");
}