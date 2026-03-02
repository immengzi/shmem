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
constexpr uint64_t MIN_EXPANSION_SIZE = 256 * 1024 * 1024;  // 256MB最小扩容
constexpr double EXPANSION_FACTOR = 1.5;                    // 1.5倍扩容因子
constexpr uint64_t MAX_BLOCK_SIZE = 4ULL * 1024 * 1024 * 1024;  // 4GB最大单块

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
    
    // 在动态块中查找对齐分配
    for (auto& block : memory_blocks_) {
        if (block->is_external) {
            // 对于外部内存块，使用CANN的对齐分配
            void* aligned_ptr = nullptr;
            uint64_t actual_size = aligned_size + alignment - 1;
            
            // 这里简化处理，实际应该更精确地管理对齐内存
            if (block->size - block->used_size >= actual_size) {
                // 简化的对齐分配逻辑
                uint8_t* block_start = static_cast<uint8_t*>(block->base_addr);
                uint8_t* aligned_start = reinterpret_cast<uint8_t*>(
                    (reinterpret_cast<uintptr_t>(block_start + block->used_size) + alignment - 1) & 
                    ~(alignment - 1));
                
                if (aligned_start + aligned_size <= block_start + block->size) {
                    aligned_ptr = aligned_start;
                    uint64_t padding_size = aligned_start - (block_start + block->used_size);
                    block->used_size += padding_size + aligned_size;
                    address_to_block_map_[aligned_ptr] = block.get();
                    external_alloc_size_map_[aligned_ptr] = aligned_size;  // 记录实际分配大小
                    update_block_statistics(block.get(), aligned_size);
                    pthread_spin_unlock(&spinlock_);
                    
                    SHM_LOG_DEBUG("Aligned allocated " << size << " bytes (padding: " << padding_size 
                                  << ") from dynamic block at " << aligned_ptr);
                    return aligned_ptr;
                }
            }
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
    
    // 检查是否在初始内存池中
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
    
    // 检查是否在动态内存块中
    auto it = address_to_block_map_.find(address);
    if (it != address_to_block_map_.end()) {
        dynamic_memory_block* block = it->second;
        
        // 获取准确的分配大小进行统计修正
        auto size_it = external_alloc_size_map_.find(address);
        uint64_t alloc_size = (size_it != external_alloc_size_map_.end()) ? 
                              size_it->second : allocated_size_align_up(1);
        
        address_to_block_map_.erase(it);
        if (size_it != external_alloc_size_map_.end()) {
            external_alloc_size_map_.erase(size_it);
        }
        
        // 更新块的使用统计
        if (block->used_size >= alloc_size) {
            block->used_size -= alloc_size;
        } else {
            block->used_size = 0;  // 防止下溢
        }
        
        update_block_statistics(block, -static_cast<int64_t>(alloc_size));
        
        pthread_spin_unlock(&spinlock_);
        SHM_LOG_DEBUG("Released memory at " << address << " (size: " << alloc_size << ") from dynamic block");
        return 0;
    }
    
    pthread_spin_unlock(&spinlock_);
    SHM_LOG_ERROR("Release invalid address " << address);
    return -1;
}

bool dynamic_memory_manager::expand_pool(uint64_t required_size) noexcept {
    // 注意：此函数假设调用者已持有 spinlock_
    // 先解锁，避免长时间持有锁进行内存分配
    pthread_spin_unlock(&spinlock_);
    
    // 计算需要的扩容大小
    uint64_t expansion_size = std::max(required_size, MIN_EXPANSION_SIZE);
    expansion_size = static_cast<uint64_t>(expansion_size * EXPANSION_FACTOR);
    expansion_size = std::min(expansion_size, MAX_BLOCK_SIZE);
    
    // 确保对齐
    constexpr uint64_t ALIGNMENT = 256 * 1024; // 256KB对齐
    expansion_size = (expansion_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
    SHM_LOG_INFO("Attempting to expand memory pool by " << expansion_size << " bytes");
    
    // 使用CANN接口分配新的内存块（此时不持有锁，允许其他线程并行）
    void* new_block_addr = nullptr;
    aclError ret = aclrtMalloc(&new_block_addr, expansion_size, ACL_MEM_MALLOC_HUGE_FIRST);
    
    if (ret != ACL_SUCCESS || new_block_addr == nullptr) {
        SHM_LOG_ERROR("Failed to allocate external memory block of size " << expansion_size 
                      << ", error code: " << ret);
        // 重新加锁，保持锁状态一致性
        pthread_spin_lock(&spinlock_);
        return false;
    }
    
    // 内存分配成功，重新加锁更新元数据
    pthread_spin_lock(&spinlock_);
    
    // 创建新的内存块管理结构
    auto new_block = std::make_unique<dynamic_memory_block>(new_block_addr, expansion_size, true);
    dynamic_memory_block* block_ptr = new_block.get();
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
    
    // 清理完全空闲的外部内存块（used_size == 0）
    for (auto it = memory_blocks_.begin(); it != memory_blocks_.end();) {
        auto& block = *it;
        if (block->is_external && block->used_size == 0) {
            // 释放 CANN 内存
            if (block->base_addr != nullptr) {
                aclrtFree(block->base_addr);
                SHM_LOG_INFO("Freed unused external memory block: " << block->base_addr 
                             << " (size: " << block->size << ")");
            }
            
            // 更新总容量
            total_capacity_ -= block->size;
            
            // 从 vector 中移除（vector 不支持 erase，需要 swap-pop）
            it = memory_blocks_.erase(it);
        } else {
            ++it;
        }
    }
    
    pthread_spin_unlock(&spinlock_);
    
    SHM_LOG_INFO("Memory cleanup completed. Current capacity: " << total_capacity_ 
                 << ", allocated: " << total_allocated_);
}

// 私有辅助函数实现
dynamic_memory_block* dynamic_memory_manager::find_suitable_block(uint64_t size) noexcept {
    // 查找有足够空间的外部内存块
    for (auto& block : memory_blocks_) {
        if (block->is_external && (block->size - block->used_size) >= size) {
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
    void* allocated_addr = block_start + block->used_size;
    
    block->used_size += size;
    address_to_block_map_[allocated_addr] = block;
    external_alloc_size_map_[allocated_addr] = size;  // 记录分配大小用于准确统计
    update_block_statistics(block, size);
    
    return allocated_addr;
}

void dynamic_memory_manager::update_block_statistics(dynamic_memory_block* block, int64_t size_delta) noexcept {
    total_allocated_ += size_delta;
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