/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DYNAMIC_MM_H
#define SHMEM_DYNAMIC_MM_H

#include <pthread.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <vector>
#include <memory>

#include "host/shmem_host_def.h"
#include "utils/shmemi_host_types.h"

// 动态内存块信息结构
struct dynamic_memory_block {
    void* base_addr;           // 内存块基地址
    uint64_t size;             // 内存块大小
    uint64_t used_size;        // 当前活跃分配的字节总量（仅用于统计）
    uint64_t high_water;       // bump指针：下一次分配的起始偏移（只增不减）
    bool is_external;          // 是否为外部CANN分配的内存

    // 外部块空闲槽位列表：按 offset 有序排列的 {offset, size} 对。
    // 使用有序 vector（lower_bound 插入 + 原地邻居合并），避免 std::map 每节点
    // 的堆分配开销——推理热路径中 alloc/free 非常频繁，map 的 new/delete 会
    // 累积成系统性延迟。N 通常 < 10，vector 的缓存局部性远优于 map。
    // 仅用于 is_external == true 的块。
    std::vector<std::pair<uint64_t, uint64_t>> free_slots;

    dynamic_memory_block(void* addr, uint64_t block_size, bool external = false)
        : base_addr(addr), size(block_size), used_size(0), high_water(0),
          is_external(external) {}
};

// 动态扩容内存管理器
class dynamic_memory_manager {
public:
    dynamic_memory_manager(void *base, uint64_t initial_size) noexcept;
    ~dynamic_memory_manager() noexcept;

public:
    // 基本内存分配接口
    void *allocate(uint64_t size) noexcept;
    void *aligned_allocate(uint64_t alignment, uint64_t size) noexcept;
    bool change_size(void *address, uint64_t size) noexcept;
    int32_t release(void *address) noexcept;
    bool allocated_size(void *address, uint64_t &size) const noexcept;
    
    // 动态扩容相关接口
    bool expand_pool(uint64_t required_size) noexcept;
    uint64_t get_total_capacity() const noexcept;
    uint64_t get_used_memory() const noexcept;
    uint64_t get_available_memory() const noexcept;
    void cleanup_unused_blocks() noexcept;

private:
    static uint64_t allocated_size_align_up(uint64_t input_size) noexcept;
    static bool alignment_matches(const memory_range &mr, uint64_t alignment, uint64_t size,
                                  uint64_t &head_skip) noexcept;
    // 动态扩容辅助函数
    dynamic_memory_block* find_suitable_block(uint64_t size) noexcept;
    void* allocate_from_block(dynamic_memory_block* block, uint64_t size) noexcept;
    void update_block_statistics(dynamic_memory_block* block, int64_t size_delta) noexcept;

private:
    // 初始内存池
    uint8_t *const initial_base_;
    const uint64_t initial_size_;
    
    // 动态内存块分配记录：将 block 指针与大小信息合并在一张 map 中，
    // 避免原来 address_to_block_map_ + external_alloc_info_map_ 的双重查找开销。
    // 每次动态块 alloc/free 只需一次 O(log N) 查找（原来是两次）。
    struct DynAllocInfo {
        dynamic_memory_block* block;
        uint64_t data_size;    // 实际分配的数据大小
        uint64_t total_size;   // 含对齐 padding 的总大小（用于 free_slots 归还）
        uint64_t block_offset; // 内存块内起始偏移（含 pre-padding，used by release()）
    };
    std::vector<std::unique_ptr<dynamic_memory_block>> memory_blocks_;
    std::map<void*, DynAllocInfo> address_to_block_map_;
    
    // 统计信息
    uint64_t total_allocated_;
    uint64_t total_capacity_;
    
    // 同步锁
    mutable pthread_spinlock_t spinlock_{};
    
    // 内存管理数据结构（针对初始内存池）
    std::map<uint64_t, uint64_t> address_idle_tree_;
    std::map<uint64_t, uint64_t> address_used_tree_;
    std::set<memory_range, range_size_first_comparator> size_idle_tree_;
};

// 动态内存管理器工厂函数
std::shared_ptr<dynamic_memory_manager> create_dynamic_memory_manager(void *base, uint64_t size);

#endif  // SHMEM_DYNAMIC_MM_H