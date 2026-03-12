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
#include <algorithm>
#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "shmem_dynamic_mm.h"

// 内存扩容策略常量
constexpr uint64_t MIN_EXPANSION_SIZE = 512 * 1024 * 1024;  // 512MB最小扩容（减少扩容次数）
constexpr double EXPANSION_FACTOR = 1.5;                    // 1.5倍扩容因子
constexpr uint64_t MAX_BLOCK_SIZE = 8ULL * 1024 * 1024 * 1024;  // 8GB最大单块
// 设备内存预留：expand_pool 保留给驱动/CANN 内部使用的最小余量
constexpr uint64_t DEVICE_MEM_HEADROOM = 512 * 1024 * 1024;  // 512MB保留

dynamic_memory_manager::dynamic_memory_manager(void *base, uint64_t initial_size) noexcept 
    : initial_base_{reinterpret_cast<uint8_t *>(base)}, 
      initial_size_{initial_size},
      total_allocated_(0),
      total_capacity_(initial_size) {
    
    pthread_spin_init(&spinlock_, 0);
    
    // 初始化初始内存池的数据结构
    address_idle_tree_[0] = initial_size_;
    size_idle_tree_.insert({0, initial_size_});
    
    // 创建初始内存块
    memory_blocks_.push_back(std::make_unique<dynamic_memory_block>(base, initial_size_, false));
    
    SHM_LOG_INFO("Dynamic memory manager initialized with initial size: " << initial_size_ << " bytes");
}

dynamic_memory_manager::~dynamic_memory_manager() noexcept {
    pthread_spin_destroy(&spinlock_);
    
    // 释放所有外部分配的内存块
    for (auto& block : memory_blocks_) {
        if (block->is_external && block->base_addr != nullptr) {
            aclrtFree(block->base_addr);
            SHM_LOG_DEBUG("Freed external memory block: " << block->base_addr);
        }
    }
    
    memory_blocks_.clear();
    address_to_block_map_.clear();
    
    SHM_LOG_INFO("Dynamic memory manager destroyed");
}

void *dynamic_memory_manager::allocate(uint64_t size) noexcept {
    if (size == 0) {
        SHM_LOG_ERROR("Cannot allocate zero size memory");
        return nullptr;
    }

    auto aligned_size = allocated_size_align_up(size);
    
    pthread_spin_lock(&spinlock_);
    
    // 首先尝试在现有内存块中分配
    void *ptr = nullptr;
    
    // 1. 尝试在初始内存池中分配
    memory_range anchor{0, aligned_size};
    auto size_pos = size_idle_tree_.lower_bound(anchor);
    if (size_pos != size_idle_tree_.end()) {
        auto target_offset = size_pos->offset;
        auto target_size = size_pos->size;
        auto addr_pos = address_idle_tree_.find(target_offset);
        
        if (addr_pos != address_idle_tree_.end()) {
            size_idle_tree_.erase(size_pos);
            address_idle_tree_.erase(addr_pos);
            address_used_tree_.emplace(target_offset, aligned_size);
            
            if (target_size > aligned_size) {
                memory_range left{target_offset + aligned_size, target_size - aligned_size};
                address_idle_tree_.emplace(left.offset, left.size);
                size_idle_tree_.emplace(left);
            }
            
            ptr = initial_base_ + target_offset;
            update_block_statistics(memory_blocks_[0].get(), aligned_size);
            pthread_spin_unlock(&spinlock_);
            
            SHM_LOG_DEBUG("Allocated " << size << " bytes from initial pool at " << ptr);
            return ptr;
        }
    }
    
    // 2. 在动态内存块中寻找合适的块
    dynamic_memory_block* suitable_block = find_suitable_block(aligned_size);
    if (suitable_block != nullptr) {
        ptr = allocate_from_block(suitable_block, aligned_size);
        if (ptr != nullptr) {
            pthread_spin_unlock(&spinlock_);
            SHM_LOG_DEBUG("Allocated " << size << " bytes from dynamic block at " << ptr);
            return ptr;
        }
    }
    
    // 3. 如果找不到合适的块，尝试扩容
    // 注意：expand_pool 内部会先解锁，分配内存后再加锁更新元数据
    if (!expand_pool(aligned_size)) {
        pthread_spin_unlock(&spinlock_);  // 必须先解锁再返回
        SHM_LOG_ERROR("Failed to expand memory pool for size: " << size);
        return nullptr;
    }
    
    // 扩容后再次尝试分配
    suitable_block = find_suitable_block(aligned_size);
    if (suitable_block != nullptr) {
        ptr = allocate_from_block(suitable_block, aligned_size);
    }
    
    pthread_spin_unlock(&spinlock_);
    
    if (ptr != nullptr) {
        SHM_LOG_DEBUG("Allocated " << size << " bytes from expanded pool at " << ptr);
    } else {
        SHM_LOG_ERROR("Failed to allocate " << size << " bytes even after expansion");
    }
    
    return ptr;
}

void *dynamic_memory_manager::aligned_allocate(uint64_t alignment, uint64_t size) noexcept {
    if (size == 0 || alignment == 0) {
        SHM_LOG_ERROR("Invalid input, align=" << alignment << ", size=" << size);
        return nullptr;
    }

    if ((alignment & (alignment - 1UL)) != 0) {
        SHM_LOG_ERROR("Alignment should be power of 2, but real " << alignment);
        return nullptr;
    }

    auto aligned_size = allocated_size_align_up(size);
    uint64_t head_skip = 0;
    memory_range anchor{0, aligned_size};

    pthread_spin_lock(&spinlock_);
    
    // 在初始内存池中查找对齐分配
    auto size_pos = size_idle_tree_.lower_bound(anchor);
    while (size_pos != size_idle_tree_.end() && 
           !alignment_matches(*size_pos, alignment, aligned_size, head_skip)) {
        ++size_pos;
    }

    if (size_pos != size_idle_tree_.end()) {
        auto target_offset = size_pos->offset;
        auto target_size = size_pos->size;
        memory_range result_range{size_pos->offset + head_skip, aligned_size};
        size_idle_tree_.erase(size_pos);

        if (head_skip > 0) {
            size_idle_tree_.emplace(memory_range{target_offset, head_skip});
            address_idle_tree_.emplace(target_offset, head_skip);
        }

        if (head_skip + aligned_size < target_size) {
            memory_range leftMR{target_offset + head_skip + aligned_size, 
                               target_size - head_skip - aligned_size};
            size_idle_tree_.emplace(leftMR);
            address_idle_tree_.emplace(leftMR.offset, leftMR.size);
        }

        address_used_tree_.emplace(result_range.offset, result_range.size);
        update_block_statistics(memory_blocks_[0].get(), aligned_size);
        pthread_spin_unlock(&spinlock_);

        void* result = initial_base_ + result_range.offset;
        SHM_LOG_DEBUG("Aligned allocated " << size << " bytes (alignment " << alignment 
                      << ") from initial pool at " << result);
        return result;
    }
    
    // 在动态块中查找对齐分配（使用 high_water 作为 bump 指针，避免与释放后的分配重叠）
    for (auto& block : memory_blocks_) {
        if (!block->is_external) continue;

        uint8_t* block_start = static_cast<uint8_t*>(block->base_addr);

        // 先尝试从空闲槽位 map 中找到合适的对齐槽
        auto& slots = block->free_slots;
        bool slot_found = false;
        for (auto it = slots.begin(); it != slots.end(); ++it) {
            uint8_t* slot_ptr = block_start + it->first;
            uint8_t* aligned_start = reinterpret_cast<uint8_t*>(
                (reinterpret_cast<uintptr_t>(slot_ptr) + alignment - 1) & ~(alignment - 1));
            uint64_t padding_size = static_cast<uint64_t>(aligned_start - slot_ptr);
            uint64_t total_size = padding_size + aligned_size;
            if (it->second >= total_size) {
                uint64_t remaining = it->second - total_size;
                uint64_t slot_offset = it->first;
                it = slots.erase(it);
                if (remaining > 0) {
                    slots[slot_offset + total_size] = remaining;
                }
                address_to_block_map_[aligned_start] = block.get();
                external_alloc_info_map_[aligned_start] = {aligned_size, total_size};
                block->used_size += aligned_size;
                update_block_statistics(block.get(), aligned_size);
                pthread_spin_unlock(&spinlock_);
                SHM_LOG_DEBUG("Aligned allocated " << size << " bytes (slot, padding: "
                              << padding_size << ") from dynamic block at " << (void*)aligned_start);
                return aligned_start;
            }
        }
        if (slot_found) continue;

        // 从 bump 区域分配（high_water 只增不减，避免重叠）
        uint8_t* bump_ptr = block_start + block->high_water;
        uint8_t* aligned_start = reinterpret_cast<uint8_t*>(
            (reinterpret_cast<uintptr_t>(bump_ptr) + alignment - 1) & ~(alignment - 1));
        uint64_t padding_size = static_cast<uint64_t>(aligned_start - bump_ptr);
        uint64_t total_size = padding_size + aligned_size;

        if (block->high_water + total_size <= block->size) {
            block->high_water += total_size;
            block->used_size += aligned_size;
            address_to_block_map_[aligned_start] = block.get();
            external_alloc_info_map_[aligned_start] = {aligned_size, total_size};
            update_block_statistics(block.get(), aligned_size);
            pthread_spin_unlock(&spinlock_);
            SHM_LOG_DEBUG("Aligned allocated " << size << " bytes (bump, padding: "
                          << padding_size << ") from dynamic block at " << (void*)aligned_start);
            return aligned_start;
        }
    }
    
    pthread_spin_unlock(&spinlock_);
    SHM_LOG_ERROR("Cannot allocate aligned memory with size: " << size << ", alignment: " << alignment);
    return nullptr;
}

int32_t dynamic_memory_manager::release(void *address) noexcept {
    if (address == nullptr) {
        return 0;
    }

    auto u8a = reinterpret_cast<uint8_t *>(address);
    
    pthread_spin_lock(&spinlock_);
    
    // 1. 首先检查是否在动态内存块中（优先检查，避免地址范围重叠误判）
    auto it = address_to_block_map_.find(address);
    if (it != address_to_block_map_.end()) {
        dynamic_memory_block* block = it->second;

        // 获取准确的分配大小进行统计修正
        auto info_it = external_alloc_info_map_.find(address);
        uint64_t data_size = allocated_size_align_up(1);  // 默认使用对齐后的最小大小
        uint64_t total_size = data_size;

        if (info_it != external_alloc_info_map_.end()) {
            data_size = info_it->second.first;
            total_size = info_it->second.second;
            external_alloc_info_map_.erase(info_it);
        }

        address_to_block_map_.erase(it);

        // 更新活跃分配统计（used_size 只用于统计，不作为 bump 指针）
        if (block->used_size >= total_size) {
            block->used_size -= total_size;
        } else {
            block->used_size = 0;  // 防止下溢
        }

        // 将归还的区域插入有序空闲槽位 map，O(log n) 插入并与相邻槽合并，减少碎片
        // high_water 保持不变，只有 bump 区域全空时才回退
        uint64_t freed_offset = static_cast<uint64_t>(
            static_cast<uint8_t*>(address) -
            static_cast<uint8_t*>(block->base_addr));

        auto& slots = block->free_slots;
        uint64_t merged_offset = freed_offset;
        uint64_t merged_size   = total_size;

        // 向后合并：freed 末尾紧接下一块起始
        auto next_it = slots.lower_bound(freed_offset);
        if (next_it != slots.end() &&
            merged_offset + merged_size == next_it->first) {
            merged_size += next_it->second;
            next_it = slots.erase(next_it);
        }
        // 向前合并：上一块末尾紧接 freed 起始
        if (next_it != slots.begin()) {
            auto prev_it = std::prev(next_it);
            if (prev_it->first + prev_it->second == merged_offset) {
                merged_offset = prev_it->first;
                merged_size  += prev_it->second;
                slots.erase(prev_it);
            }
        }
        // 若合并后的空闲区紧贴 high_water，将 high_water 回退以释放 bump 空间
        if (merged_offset + merged_size == block->high_water) {
            block->high_water = merged_offset;
            // 不写入 slots，因为该区间已归还给 bump 区
        } else {
            slots[merged_offset] = merged_size;
        }

        update_block_statistics(block, -static_cast<int64_t>(data_size));

        pthread_spin_unlock(&spinlock_);
        SHM_LOG_DEBUG("Released memory at " << address << " (data: " << data_size
                      << ", total: " << total_size << ") from dynamic block, "
                      << "free_slots=" << block->free_slots.size()
                      << " high_water=" << block->high_water);
        return 0;
    }
    
    // 2. 检查是否在初始内存池中（使用精确的偏移量查找）
    if (u8a >= initial_base_ && u8a < initial_base_ + initial_size_) {
        auto offset = u8a - initial_base_;
        auto pos = address_used_tree_.find(offset);
        if (pos != address_used_tree_.end()) {
            auto size = pos->second;
            uint64_t final_offset = static_cast<uint64_t>(offset);
            uint64_t final_size = size;
            address_used_tree_.erase(pos);

            // 合并空闲块
            auto prev_addr_pos = address_idle_tree_.lower_bound(offset);
            if (prev_addr_pos != address_idle_tree_.begin()) {
                --prev_addr_pos;
                if (prev_addr_pos != address_idle_tree_.end() &&
                    prev_addr_pos->first + prev_addr_pos->second == static_cast<uint64_t>(offset)) {
                    final_offset = prev_addr_pos->first;
                    final_size += prev_addr_pos->second;
                    auto prev_addr_range = *prev_addr_pos;
                    address_idle_tree_.erase(prev_addr_pos);
                    size_idle_tree_.erase(memory_range{prev_addr_range.first, prev_addr_range.second});
                }
            }

            auto next_addr_pos = address_idle_tree_.find(offset + size);
            if (next_addr_pos != address_idle_tree_.end()) {
                uint64_t next_addr = next_addr_pos->first;
                uint64_t next_size = next_addr_pos->second;
                final_size += next_size;
                address_idle_tree_.erase(next_addr_pos);
                size_idle_tree_.erase(memory_range{next_addr, next_size});
            }
            
            address_idle_tree_.emplace(final_offset, final_size);
            size_idle_tree_.emplace(memory_range{final_offset, final_size});
            update_block_statistics(memory_blocks_[0].get(), -static_cast<int64_t>(size));
            
            pthread_spin_unlock(&spinlock_);
            SHM_LOG_DEBUG("Released memory at " << address << " from initial pool");
            return 0;
        }
    }
    
    // 地址未找到
    pthread_spin_unlock(&spinlock_);
    SHM_LOG_ERROR("Release invalid address " << address);
    return -1;
}

bool dynamic_memory_manager::expand_pool(uint64_t required_size) noexcept {
    // 注意：此函数假设调用者已持有 spinlock_
    // 先解锁，避免长时间持有锁进行内存分配
    pthread_spin_unlock(&spinlock_);

    // ── 边界检查：查询设备剩余显存，防止总分配量超出物理 HBM ──────────────────
    // Bug 修复：原实现对 aclrtMalloc 扩容块没有上界约束，
    // 在 gpu_memory_utilization 较高时会触发 OOM 或分配超过 64GB 的错误。
    constexpr uint64_t ALIGNMENT = 256 * 1024; // 256KB对齐

    uint64_t free_mem = 0, total_mem = 0;
    aclError mem_info_ret = aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem);
    if (mem_info_ret != ACL_SUCCESS) {
        // 无法查询时退保守：只按 required_size 申请，不过度预留
        SHM_LOG_WARN("aclrtGetMemInfo failed (ret=" << mem_info_ret
                     << "), falling back to conservative expansion");
        free_mem = required_size + DEVICE_MEM_HEADROOM;  // 假设刚好够用
    }

    // 留出 DEVICE_MEM_HEADROOM 给驱动/CANN 内部使用
    if (free_mem <= DEVICE_MEM_HEADROOM) {
        SHM_LOG_ERROR("Insufficient device memory for expansion: free=" << free_mem
                      << " headroom=" << DEVICE_MEM_HEADROOM);
        pthread_spin_lock(&spinlock_);
        return false;
    }
    uint64_t available = free_mem - DEVICE_MEM_HEADROOM;

    // required_size 本身就超出可用显存，直接报错
    if (required_size > available) {
        SHM_LOG_ERROR("Cannot satisfy required_size=" << required_size
                      << " with available=" << available << " (free=" << free_mem << ")");
        pthread_spin_lock(&spinlock_);
        return false;
    }

    // ── 计算期望扩容大小 ──────────────────────────────────────────────────────
    // 至少分配 required_size，同时用 EXPANSION_FACTOR 预留余量减少下次扩容频率；
    // 上限取 MAX_BLOCK_SIZE 和 available 中的较小值，确保不超出物理 HBM。
    uint64_t expansion_size = std::max(required_size, MIN_EXPANSION_SIZE);
    expansion_size = static_cast<uint64_t>(expansion_size * EXPANSION_FACTOR);
    expansion_size = std::min(expansion_size, MAX_BLOCK_SIZE);   // 单块不超过 MAX_BLOCK_SIZE
    expansion_size = std::min(expansion_size, available);        // 不超出设备剩余显存
    // 对齐到 256KB
    expansion_size = (expansion_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    // 对齐后仍需满足 required_size（边界情况：required_size 接近 available）
    if (expansion_size < required_size) {
        expansion_size = (required_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }

    SHM_LOG_INFO("Attempting to expand memory pool by " << expansion_size
                 << " bytes (free=" << free_mem << ", available=" << available << ")");

    // ── 分级尝试：优先分配期望大小，失败则逐步折半直到 required_size ──────────
    void* new_block_addr = nullptr;
    uint64_t try_size = expansion_size;
    uint64_t min_acceptable = (required_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

    while (try_size >= min_acceptable) {
        new_block_addr = nullptr;
        aclError ret = aclrtMalloc(&new_block_addr, try_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret == ACL_SUCCESS && new_block_addr != nullptr) {
            expansion_size = try_size;
            break;
        }
        SHM_LOG_WARN("aclrtMalloc(" << try_size << ") failed (ret=" << ret
                     << "), retrying with smaller size");
        if (try_size == min_acceptable) {
            break;  // 已是最小可接受大小，放弃
        }
        // 折半，但不低于 min_acceptable
        try_size = std::max(try_size / 2, min_acceptable);
        try_size = (try_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }

    if (new_block_addr == nullptr) {
        SHM_LOG_ERROR("Failed to expand memory pool for required_size=" << required_size
                      << " after all retries");
        pthread_spin_lock(&spinlock_);
        return false;
    }

    // 内存分配成功，重新加锁更新元数据
    pthread_spin_lock(&spinlock_);

    // 创建新的内存块管理结构
    auto new_block = std::make_unique<dynamic_memory_block>(new_block_addr, expansion_size, true);
    memory_blocks_.push_back(std::move(new_block));

    total_capacity_ += expansion_size;

    SHM_LOG_INFO("Successfully expanded memory pool. New block: " << new_block_addr
                 << ", size: " << expansion_size << ", total capacity: " << total_capacity_);

    return true;
}

uint64_t dynamic_memory_manager::get_total_capacity() const noexcept {
    return total_capacity_;
}

uint64_t dynamic_memory_manager::get_used_memory() const noexcept {
    return total_allocated_;
}

uint64_t dynamic_memory_manager::get_available_memory() const noexcept {
    return total_capacity_ - total_allocated_;
}

void dynamic_memory_manager::cleanup_unused_blocks() noexcept {
    pthread_spin_lock(&spinlock_);
    
    uint64_t freed_count = 0;
    uint64_t freed_size = 0;
    
    // 清理完全空闲的外部内存块（used_size == 0 且没有活跃分配）
    for (auto it = memory_blocks_.begin(); it != memory_blocks_.end();) {
        auto& block = *it;
        if (block->is_external && block->used_size == 0 && block->free_slots.empty()) {
            // 检查是否还有未释放的分配（双重检查）
            bool has_active_alloc = false;
            for (const auto& pair : address_to_block_map_) {
                if (pair.second == block.get()) {
                    has_active_alloc = true;
                    break;
                }
            }
            
            if (has_active_alloc) {
                SHM_LOG_WARN("Block at " << block->base_addr << " has used_size=0 but active allocations exist");
                ++it;
                continue;
            }
            
            // 释放 CANN 内存
            if (block->base_addr != nullptr) {
                aclrtFree(block->base_addr);
                freed_size += block->size;
                freed_count++;
                SHM_LOG_INFO("Freed unused external memory block: " << block->base_addr 
                             << " (size: " << block->size << ")");
            }
            
            // 更新总容量
            total_capacity_ -= block->size;
            
            // 从 vector 中移除
            it = memory_blocks_.erase(it);
        } else {
            ++it;
        }
    }
    
    pthread_spin_unlock(&spinlock_);
    
    if (freed_count > 0) {
        SHM_LOG_INFO("Memory cleanup completed. Freed " << freed_count << " blocks (" 
                     << freed_size << " bytes). Current capacity: " << total_capacity_ 
                     << ", allocated: " << total_allocated_);
    } else {
        SHM_LOG_DEBUG("Memory cleanup completed. No blocks freed. Current capacity: " 
                      << total_capacity_ << ", allocated: " << total_allocated_);
    }
}

// 私有辅助函数实现
dynamic_memory_block* dynamic_memory_manager::find_suitable_block(uint64_t size) noexcept {
    // 优先从空闲槽位中寻找合适的外部块（best-fit 复用已归还的段）
    for (auto& block : memory_blocks_) {
        if (!block->is_external) continue;
        // 检查空闲槽位（map 按 offset 有序，遍历 value 即大小）
        for (const auto& kv : block->free_slots) {
            if (kv.second >= size) {
                return block.get();
            }
        }
        // 检查 bump 区域剩余空间
        if ((block->size - block->high_water) >= size) {
            return block.get();
        }
    }
    return nullptr;
}

void* dynamic_memory_manager::allocate_from_block(dynamic_memory_block* block, uint64_t size) noexcept {
    if (!block->is_external) {
        return nullptr; // 初始内存池不应该通过此函数分配
    }

    uint8_t* block_start = static_cast<uint8_t*>(block->base_addr);
    void* allocated_addr = nullptr;

    // 优先复用空闲槽位（best-fit：在 map 中找能满足需求的最小槽）
    // map 按 offset 有序；通过线性扫描找最小满足槽，通常槽数量很少
    auto best_it = block->free_slots.end();
    uint64_t best_size = UINT64_MAX;
    for (auto it = block->free_slots.begin(); it != block->free_slots.end(); ++it) {
        if (it->second >= size && it->second < best_size) {
            best_size = it->second;
            best_it = it;
        }
    }

    if (best_it != block->free_slots.end()) {
        uint64_t slot_offset = best_it->first;
        uint64_t slot_size   = best_it->second;
        block->free_slots.erase(best_it);
        // 如果槽比请求大，将剩余部分放回空闲 map
        if (slot_size > size) {
            block->free_slots[slot_offset + size] = slot_size - size;
        }
        allocated_addr = block_start + slot_offset;
        SHM_LOG_DEBUG("Reusing free slot at offset " << slot_offset
                      << " (slot=" << slot_size << " req=" << size << ")");
    } else {
        // 没有合适的槽位，从 bump 区域分配
        if (block->size - block->high_water < size) {
            // 防御：不应该发生（调用前已通过 find_suitable_block 检查）
            SHM_LOG_ERROR("allocate_from_block: bump area exhausted unexpectedly");
            return nullptr;
        }
        allocated_addr = block_start + block->high_water;
        block->high_water += size;
    }

    block->used_size += size;
    address_to_block_map_[allocated_addr] = block;
    // 记录分配信息：<数据大小, 总大小（无padding时等于数据大小）>
    external_alloc_info_map_[allocated_addr] = {size, size};
    update_block_statistics(block, size);

    return allocated_addr;
}

void dynamic_memory_manager::update_block_statistics(dynamic_memory_block* block, int64_t size_delta) noexcept {
    // 检查下溢：当 size_delta 为负且绝对值大于当前值时
    if (size_delta < 0 && static_cast<uint64_t>(-size_delta) > total_allocated_) {
        SHM_LOG_WARN("Memory allocation underflow detected: delta=" << size_delta 
                     << ", current=" << total_allocated_);
        total_allocated_ = 0;
    } else {
        total_allocated_ += size_delta;
    }
    
    // 检查上溢：使用量超过总容量
    if (total_allocated_ > total_capacity_) {
        SHM_LOG_WARN("Memory usage exceeds capacity: " << total_allocated_ << " > " << total_capacity_);
    }
}

// 静态工具函数实现
uint64_t dynamic_memory_manager::allocated_size_align_up(uint64_t input_size) noexcept {
    constexpr uint64_t align_size = 16UL;
    constexpr uint64_t align_size_mask = ~(align_size - 1UL);
    return (input_size + align_size - 1UL) & align_size_mask;
}

bool dynamic_memory_manager::alignment_matches(const memory_range &mr, uint64_t alignment, uint64_t size,
                                               uint64_t &head_skip) noexcept {
    if (mr.size < size) {
        return false;
    }

    if ((mr.offset & (alignment - 1UL)) == 0UL) {
        head_skip = 0;
        return true;
    }

    auto aligned_offset = ((mr.offset + alignment - 1UL) & (~(alignment - 1UL)));
    head_skip = aligned_offset - mr.offset;
    return mr.size >= size + head_skip;
}

// 工厂函数
std::shared_ptr<dynamic_memory_manager> create_dynamic_memory_manager(void *base, uint64_t size) {
    return std::make_shared<dynamic_memory_manager>(base, size);
}

// 为了兼容现有接口，保留原有函数但使用动态管理器
namespace {
std::shared_ptr<dynamic_memory_manager> dynamic_memory_manager_instance;
}

int32_t dynamic_memory_manager_initialize(void *base, uint64_t size, aclshmem_mem_type_t mem_type = DEVICE_SIDE) {
    if (mem_type == HOST_SIDE) {
        SHM_LOG_ERROR("Dynamic memory manager doesn't support HOST_SIDE currently");
        return ACLSHMEM_INNER_ERROR;
    }
    
    dynamic_memory_manager_instance = create_dynamic_memory_manager(base, size);
    if (dynamic_memory_manager_instance == nullptr) {
        SHM_LOG_ERROR("Failed to initialize dynamic memory manager");
        return ACLSHMEM_INNER_ERROR;
    }
    
    SHM_LOG_INFO("Dynamic memory manager initialized successfully");
    return ACLSHMEM_SUCCESS;
}

void dynamic_memory_manager_destroy() {
    dynamic_memory_manager_instance.reset();
    SHM_LOG_INFO("Dynamic memory manager destroyed");
}

void *dynamic_aclshmem_malloc(size_t size) {
    if (dynamic_memory_manager_instance == nullptr) {
        SHM_LOG_ERROR("Dynamic memory manager not initialized");
        return nullptr;
    }
    
    void *ptr = dynamic_memory_manager_instance->allocate(size);
    SHM_LOG_DEBUG("dynamic_aclshmem_malloc(" << size << ") = " << ptr);
    return ptr;
}

void dynamic_aclshmem_free(void *ptr) {
    if (dynamic_memory_manager_instance == nullptr || ptr == nullptr) {
        return;
    }
    
    dynamic_memory_manager_instance->release(ptr);
    SHM_LOG_DEBUG("dynamic_aclshmem_free(" << ptr << ")");
}