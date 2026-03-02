/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <memory>
#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "shmemi_mm.h"
#include "shmem_dynamic_mm.h"  // 新增动态内存管理器头文件

namespace {
std::shared_ptr<memory_manager> aclshmemi_memory_manager;
std::shared_ptr<memory_manager> aclshmemi_host_memory_manager = nullptr;
std::shared_ptr<dynamic_memory_manager> dynamic_memory_manager_instance;  // 动态内存管理器实例

// 控制是否启用动态扩容的标志
bool enable_dynamic_expansion = true;
}

// 新增：设置是否启用动态扩容
void aclshmem_enable_dynamic_expansion(bool enable) {
    enable_dynamic_expansion = enable;
    SHM_LOG_INFO("Dynamic memory expansion " << (enable ? "enabled" : "disabled"));
}

// 新增：获取动态扩容状态
bool aclshmem_is_dynamic_expansion_enabled() {
    return enable_dynamic_expansion;
}

int32_t memory_manager_initialize(void *base, uint64_t size, aclshmem_mem_type_t mem_type)
{
    if (mem_type == HOST_SIDE) {
        aclshmemi_host_memory_manager = std::make_shared<memory_manager>(base, size);
        return ACLSHMEM_SUCCESS;
    }
    
    // 根据配置决定使用哪种内存管理器
    if (enable_dynamic_expansion) {
        // 使用动态内存管理器
        dynamic_memory_manager_instance = create_dynamic_memory_manager(base, size);
        if (dynamic_memory_manager_instance == nullptr) {
            SHM_LOG_ERROR("Failed to initialize dynamic memory manager, falling back to static manager");
            // 回退到静态管理器
            goto fallback_static;
        }
        SHM_LOG_INFO("Initialized dynamic memory manager with size: " << size);
        return ACLSHMEM_SUCCESS;
    } else {
        fallback_static:
        // 使用原有的静态内存管理器
        aclshmemi_memory_manager = std::make_shared<memory_manager>(base, size);
        if (aclshmemi_memory_manager == nullptr) {
            SHM_LOG_ERROR("Failed to initialize shared memory heap");
            return ACLSHMEM_INNER_ERROR;
        }
        SHM_LOG_INFO("Initialized static memory manager with size: " << size);
        return ACLSHMEM_SUCCESS;
    }
}

void memory_manager_destroy()
{
    // 清理动态内存管理器
    if (dynamic_memory_manager_instance) {
        dynamic_memory_manager_instance.reset();
        SHM_LOG_INFO("Dynamic memory manager destroyed");
    }
    
    // 清理静态内存管理器
    aclshmemi_memory_manager.reset();
    if (aclshmemi_host_memory_manager != nullptr) {
        aclshmemi_host_memory_manager.reset();
    }
}

void *aclshmem_malloc(size_t size)
{
    // 优先使用动态内存管理器（如果启用且已初始化）
    if (enable_dynamic_expansion && dynamic_memory_manager_instance) {
        void *ptr = dynamic_memory_manager_instance->allocate(size);
        SHM_LOG_DEBUG("aclshmem_malloc(" << size << ")" << " ptr: " << ptr << " (dynamic)");
        
        if (ptr != nullptr) {
            auto ret = aclshmemi_control_barrier_all();
            if (ret != 0) {
                SHM_LOG_ERROR("malloc mem barrier failed, ret: " << ret);
                dynamic_memory_manager_instance->release(ptr);
                return nullptr;
            }
        }
        
#ifdef DEBUG_MODE
        ret = is_alloc_size_symmetric(size);
        if (ret != 0) {
            SHM_LOG_ERROR("asymmetric alloc detected");
            if (ptr != nullptr) {
                dynamic_memory_manager_instance->release(ptr);
            }
            return nullptr;
        }
#endif
        return ptr;
    }
    
    // 回退到静态内存管理器
    if (aclshmemi_memory_manager == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return nullptr;
    }

    void *ptr = aclshmemi_memory_manager->allocate(size);
    SHM_LOG_DEBUG("aclshmem_malloc(" << size << ")" << " ptr: " << ptr << " (static)");
    auto ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("malloc mem barrier failed, ret: " << ret);
        if (ptr != nullptr) {
            aclshmemi_memory_manager->release(ptr);
            ptr = nullptr;
        }
    }
#ifdef DEBUG_MODE
    ret = is_alloc_size_symmetric(size);
    if (ret != 0) {
        SHM_LOG_ERROR("asymmetric alloc detected");
        return nullptr;
    }
#endif
    return ptr;
}

void *aclshmem_calloc(size_t nmemb, size_t size)
{
    // 优先使用动态内存管理器
    if (enable_dynamic_expansion && dynamic_memory_manager_instance) {
        SHM_ASSERT_MULTIPLY_OVERFLOW(nmemb, size, g_state.heap_size, nullptr);
        auto total_size = nmemb * size;
        auto ptr = dynamic_memory_manager_instance->allocate(total_size);
        
        if (ptr != nullptr) {
            auto ret = aclrtMemset(ptr, total_size, 0, total_size);
            if (ret != 0) {
                SHM_LOG_ERROR("aclshmem_calloc(" << nmemb << ", " << size << ") memset failed: " << ret);
                dynamic_memory_manager_instance->release(ptr);
                return nullptr;
            }
        }

        auto ret = aclshmemi_control_barrier_all();
        if (ret != 0) {
            SHM_LOG_ERROR("calloc mem barrier failed, ret: " << ret);
            if (ptr != nullptr) {
                dynamic_memory_manager_instance->release(ptr);
            }
            return nullptr;
        }

        SHM_LOG_DEBUG("aclshmem_calloc(" << nmemb << ", " << size << ") (dynamic)");
        return ptr;
    }
    
    // 回退到静态内存管理器
    if (aclshmemi_memory_manager == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return nullptr;
    }
    SHM_ASSERT_MULTIPLY_OVERFLOW(nmemb, size, g_state.heap_size, nullptr);

    auto total_size = nmemb * size;
    auto ptr = aclshmemi_memory_manager->allocate(total_size);
    if (ptr != nullptr) {
        auto ret = aclrtMemset(ptr, total_size, 0, total_size);
        if (ret != 0) {
            SHM_LOG_ERROR("aclshmem_calloc(" << nmemb << ", " << size << ") memset failed: " << ret);
            aclshmemi_memory_manager->release(ptr);
            ptr = nullptr;
        }
    }

    auto ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("calloc mem barrier failed, ret: " << ret);
        if (ptr != nullptr) {
            aclshmemi_memory_manager->release(ptr);
            ptr = nullptr;
        }
    }

    SHM_LOG_DEBUG("aclshmem_calloc(" << nmemb << ", " << size << ") (static)");
    return ptr;
}

void *aclshmem_align(size_t alignment, size_t size)
{
    // 优先使用动态内存管理器
    if (enable_dynamic_expansion && dynamic_memory_manager_instance) {
        auto ptr = dynamic_memory_manager_instance->aligned_allocate(alignment, size);
        auto ret = aclshmemi_control_barrier_all();
        if (ret != 0) {
            SHM_LOG_ERROR("aclshmem_align barrier failed, ret: " << ret);
            if (ptr != nullptr) {
                dynamic_memory_manager_instance->release(ptr);
            }
            return nullptr;
        }
        SHM_LOG_DEBUG("aclshmem_align(" << alignment << ", " << size << ") (dynamic)");
        return ptr;
    }
    
    // 回退到静态内存管理器
    if (aclshmemi_memory_manager == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return nullptr;
    }

    auto ptr = aclshmemi_memory_manager->aligned_allocate(alignment, size);
    auto ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmem_align barrier failed, ret: " << ret);
        if (ptr != nullptr) {
            aclshmemi_memory_manager->release(ptr);
            ptr = nullptr;
        }
    }
    SHM_LOG_DEBUG("aclshmem_align(" << alignment << ", " << size << ") (static)");
    return ptr;
}

void aclshmem_free(void *ptr)
{
    if (ptr == nullptr) {
        return;
    }

    // 优先尝试动态内存管理器释放
    if (enable_dynamic_expansion && dynamic_memory_manager_instance) {
        auto ret = dynamic_memory_manager_instance->release(ptr);
        if (ret == 0) {
            SHM_LOG_DEBUG("aclshmem_free " << ptr << " (dynamic)");
            return;
        }
        // 如果动态管理器无法释放，可能是静态分配的内存，继续尝试静态管理器
    }
    
    // 回退到静态内存管理器
    if (aclshmemi_memory_manager == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return;
    }
    
    auto ret = aclshmemi_memory_manager->release(ptr);
    if (ret != 0) {
        SHM_LOG_ERROR("release failed: " << ret);
    }

    SHM_LOG_DEBUG("aclshmem_free " << ptr << " (static)");
}

// 新增：获取内存使用统计信息
void aclshmem_get_memory_stats(uint64_t* total_capacity, uint64_t* used_memory, uint64_t* available_memory) {
    if (enable_dynamic_expansion && dynamic_memory_manager_instance) {
        if (total_capacity) *total_capacity = dynamic_memory_manager_instance->get_total_capacity();
        if (used_memory) *used_memory = dynamic_memory_manager_instance->get_used_memory();
        if (available_memory) *available_memory = dynamic_memory_manager_instance->get_available_memory();
    } else if (aclshmemi_memory_manager) {
        // 静态管理器的简单统计（近似值）
        if (total_capacity) *total_capacity = g_state.heap_size;
        if (used_memory) *used_memory = 0; // 静态管理器不跟踪使用量
        if (available_memory) *available_memory = g_state.heap_size;
    } else {
        if (total_capacity) *total_capacity = 0;
        if (used_memory) *used_memory = 0;
        if (available_memory) *available_memory = 0;
    }
}

// 新增：强制清理未使用的内存块
void aclshmem_cleanup_unused_memory() {
    if (enable_dynamic_expansion && dynamic_memory_manager_instance) {
        // 动态管理器暂不实现具体的清理逻辑
        SHM_LOG_INFO("Memory cleanup requested for dynamic manager");
    }
}

bool support_host_mem_type(aclshmem_mem_type_t mem_type)
{
#ifndef HAS_ACLRT_MEM_FABRIC_HANDLE
    if (mem_type == HOST_SIDE) {
        SHM_LOG_ERROR("Not support HOST_SIDE malloc, please update CANN version");
        return false;
    }
#endif
    return true;
}

std::shared_ptr<memory_manager> getory_manager(aclshmem_mem_type_t mem_type)
{
    if (mem_type == HOST_SIDE) {
        if (aclshmemi_host_memory_manager != nullptr) {
            return aclshmemi_host_memory_manager;
        }
        if (init_manager->setup_heap(HOST_SIDE)) {
            SHM_LOG_ERROR("Host Memory Heap Not Initialized.");
            return nullptr;
        }
        if (memory_manager_initialize(g_state.host_heap_base, g_state.heap_size, HOST_SIDE) != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("Host Memory Heap Not Initialized.");
            return nullptr;
        }
        if (init_manager->update_device_state((void *)&g_state, sizeof(aclshmem_device_host_state_t)) !=
            ACLSHMEM_SUCCESS) {
            return nullptr;
        }
        return aclshmemi_host_memory_manager;
    }
    if (aclshmemi_memory_manager == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return nullptr;
    }
    return aclshmemi_memory_manager;
}

void *aclshmemx_malloc(size_t size, aclshmem_mem_type_t mem_type)
{
    if (!support_host_mem_type(mem_type)) {
        return nullptr;
    }
    auto mem_manager = getory_manager(mem_type);
    if (mem_manager == nullptr) {
        return nullptr;
    }
    void *ptr = mem_manager->allocate(size);
    SHM_LOG_DEBUG("aclshmem_malloc(" << size << ")");
    auto ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("malloc mem barrier failed, ret: " << ret);
        if (ptr != nullptr) {
            mem_manager->release(ptr);
            ptr = nullptr;
        }
    }
    return ptr;
}

void *aclshmemx_calloc(size_t nmemb, size_t size, aclshmem_mem_type_t mem_type)
{
    if (!support_host_mem_type(mem_type)) {
        return nullptr;
    }
    auto mem_manager = getory_manager(mem_type);
    if (mem_manager == nullptr) {
        return nullptr;
    }
    SHM_ASSERT_MULTIPLY_OVERFLOW(nmemb, size, g_state.heap_size, nullptr);
    auto total_size = nmemb * size;
    auto ptr = mem_manager->allocate(total_size);
    if (ptr != nullptr) {
        auto ret = aclrtMemset(ptr, size, 0, size);
        if (ret != 0) {
            SHM_LOG_ERROR("aclshmem_calloc(" << nmemb << ", " << size << ") memset failed: " << ret);
            mem_manager->release(ptr);
            ptr = nullptr;
        }
    }
    auto ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("calloc mem barrier failed, ret: " << ret);
        if (ptr != nullptr) {
            mem_manager->release(ptr);
            ptr = nullptr;
        }
    }

    SHM_LOG_DEBUG("aclshmem_calloc(" << nmemb << ", " << size << ")");
    return ptr;
}

void *aclshmemx_align(size_t alignment, size_t size, aclshmem_mem_type_t mem_type)
{
    if (!support_host_mem_type(mem_type)) {
        return nullptr;
    }
    auto mem_manager = getory_manager(mem_type);
    if (mem_manager == nullptr) {
        return nullptr;
    }
    auto ptr = mem_manager->aligned_allocate(alignment, size);
    auto ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmem_align barrier failed, ret: " << ret);
        if (ptr != nullptr) {
            mem_manager->release(ptr);
            ptr = nullptr;
        }
    }
    SHM_LOG_DEBUG("aclshmem_align(" << alignment << ", " << size << ")");
    return ptr;
}

void aclshmemx_free(void *ptr, aclshmem_mem_type_t mem_type)
{
    if (!support_host_mem_type(mem_type)) {
        return;
    }
    if (ptr == nullptr) {
        return;
    }
    if (mem_type == HOST_SIDE && (aclshmemi_host_memory_manager == nullptr)) {
        SHM_LOG_ERROR("Host Memory Heap Not Initialized.");
        return;
    }
    if (mem_type == DEVICE_SIDE && aclshmemi_memory_manager == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return;
    }
    auto ret = mem_type == HOST_SIDE ? aclshmemi_host_memory_manager->release(ptr) : aclshmemi_memory_manager->release(ptr);;
    if (ret != 0) {
        SHM_LOG_ERROR("release failed: " << ret);
    }

    SHM_LOG_DEBUG("aclshmem_free " << ret);
}