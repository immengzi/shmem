/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_MM_H
#define SHMEMI_MM_H

#include <pthread.h>
#include <cstdint>
#include <map>
#include <set>

#include "host/shmem_host_def.h"

// 基础内存管理接口
int32_t memory_manager_initialize(void *base, uint64_t size, aclshmem_mem_type_t mem_type = DEVICE_SIDE);
void memory_manager_destroy();

// 动态扩容控制接口
void aclshmem_enable_dynamic_expansion(bool enable);
bool aclshmem_is_dynamic_expansion_enabled();

// 核心内存分配接口
void *aclshmem_malloc(size_t size);
void *aclshmem_calloc(size_t nmemb, size_t size);
void *aclshmem_align(size_t alignment, size_t size);
void aclshmem_free(void *ptr);

// 内存统计和管理接口
void aclshmem_get_memory_stats(uint64_t* total_capacity, uint64_t* used_memory, uint64_t* available_memory);
void aclshmem_cleanup_unused_memory();

// 扩展内存分配接口（支持不同内存类型）
void *aclshmemx_malloc(size_t size, aclshmem_mem_type_t mem_type);
void *aclshmemx_calloc(size_t nmemb, size_t size, aclshmem_mem_type_t mem_type);
void *aclshmemx_align(size_t alignment, size_t size, aclshmem_mem_type_t mem_type);
void aclshmemx_free(void *ptr, aclshmem_mem_type_t mem_type);

// 内存有效性检查
bool aclshmem_ptr_valid(void *ptr);

#endif  // ACLSHMEMI_MM_H