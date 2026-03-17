/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_HOST_DYNAMIC_MEM_H
#define SHMEM_HOST_DYNAMIC_MEM_H

#include <cstdint>

// 动态扩容控制接口
void aclshmem_enable_dynamic_expansion(bool enable);
bool aclshmem_is_dynamic_expansion_enabled();

// 内存统计接口
void aclshmem_get_memory_stats(uint64_t* total_capacity, uint64_t* used_memory, uint64_t* available_memory);
void aclshmem_cleanup_unused_memory();

#endif // SHMEM_HOST_DYNAMIC_MEM_H