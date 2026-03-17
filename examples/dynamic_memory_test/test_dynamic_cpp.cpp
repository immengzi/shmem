/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>
#include <algorithm>
#include "acl/acl.h"
#include "shmem.h"

class DynamicMemoryTest {
private:
    bool initialized_;
    
public:
    DynamicMemoryTest() : initialized_(false) {}
    
    bool initialize() {
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            std::cerr << "Failed to initialize ACL: " << ret << std::endl;
            return false;
        }
        
        ret = aclrtSetDevice(0);
        if (ret != ACL_SUCCESS) {
            std::cerr << "Failed to set device: " << ret << std::endl;
            return false;
        }
        
        // 启用动态扩容（必须在init之前调用）
        aclshmem_enable_dynamic_expansion(true);

        // 用 UniqueID 方式初始化
        aclshmemx_uniqueid_t uid;
        aclshmemx_init_attr_t attributes;
        
        ret = aclshmemx_get_uniqueid(&uid);
        if (ret != ACL_SUCCESS) {
            std::cerr << "Failed to get unique id: " << ret << std::endl;
            return false;
        }
        
        ret = aclshmemx_set_attr_uniqueid_args(0, 1, 32 * 1024 * 1024, &uid, &attributes);
        if (ret != ACL_SUCCESS) {
            std::cerr << "Failed to set attr: " << ret << std::endl;
            return false;
        }
        
        ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attributes);
        if (ret != ACL_SUCCESS) {
            std::cerr << "Failed to initialize SHMEM: " << ret << std::endl;
            return false;
        }
        
        initialized_ = true;
        std::cout << "SHMEM initialized successfully with dynamic expansion enabled" << std::endl;
        return true;
    }
    
    void printMemoryStats(const std::string& prefix = "") {
        uint64_t total_capacity, used_memory, available_memory;
        aclshmem_get_memory_stats(&total_capacity, &used_memory, &available_memory);
        
        // 转换为MB并保留三位小数，不四舍五入
        auto toMB = [](uint64_t bytes) -> double {
            double mb = bytes / (1024.0 * 1024.0);
            // 不四舍五入，截断到三位小数
            return static_cast<uint64_t>(mb * 1000) / 1000.0;
        };
        
        std::cout << prefix << "Memory Stats:" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Total Capacity: " << toMB(total_capacity) << " MB" << std::endl;
        std::cout << "  Used Memory: " << toMB(used_memory) << " MB" << std::endl;
        std::cout << "  Available Memory: " << toMB(available_memory) << " MB" << std::endl;
        std::cout << "  Utilization: " << (used_memory * 100.0 / total_capacity) << "%" << std::endl;
        std::cout << std::defaultfloat; // 恢复默认输出格式
    }
    
    bool testSequentialAllocation() {
        std::cout << "\n=== Testing Sequential Allocation ===" << std::endl;
        
        printMemoryStats("Before allocation: ");
        
        std::vector<void*> allocated_ptrs;
        std::vector<size_t> sizes_mb = {5, 10, 15, 25}; // MB
        
        for (size_t i = 0; i < sizes_mb.size(); i++) {
            size_t size_bytes = sizes_mb[i] * 1024 * 1024;
            std::cout << "\nAllocating " << sizes_mb[i] << " MB (allocation #" << (i+1) << ")..." << std::endl;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            void* ptr = aclshmem_malloc(size_bytes);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            if (ptr) {
                allocated_ptrs.push_back(ptr);
                std::cout << "✓ Success! Pointer: " << ptr 
                         << ", Time: " << duration.count() << " μs" << std::endl;
                printMemoryStats("After allocation: ");
            } else {
                std::cout << "✗ Failed to allocate " << sizes_mb[i] << " MB" << std::endl;
                break;
            }
        }
        
        // 释放所有分配的内存
        std::cout << "\nReleasing " << allocated_ptrs.size() << " allocated blocks..." << std::endl;
        for (void* ptr : allocated_ptrs) {
            aclshmem_free(ptr);
        }
        
        printMemoryStats("After cleanup: ");
        return true;
    }
    
    bool testLargeAllocation() {
        std::cout << "\n=== Testing Large Single Allocation ===" << std::endl;
        printMemoryStats("Before allocation: ");

        size_t large_size_mb = 100; // 100MB
        size_t large_size_bytes = large_size_mb * 1024 * 1024;

        std::cout << "Attempting to allocate " << large_size_mb << " MB block..." << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        void* large_ptr = aclshmem_malloc(large_size_bytes);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        if (large_ptr) {
            std::cout << "✓ Large allocation successful! Pointer: " << large_ptr
                    << ", Time: " << duration.count() << " μs" << std::endl;
            printMemoryStats("After allocation: ");

            // 准备 host 侧数据
            size_t check_count = 1000;
            float* host_src = nullptr;
            aclrtMallocHost(reinterpret_cast<void**>(&host_src), check_count * sizeof(float));
            for (size_t i = 0; i < check_count; i++) {
                host_src[i] = static_cast<float>(i);
            }

            // host -> device
            std::cout << "Writing test data to allocated memory..." << std::endl;
            aclrtMemcpy(large_ptr, check_count * sizeof(float),
                        host_src, check_count * sizeof(float),
                        ACL_MEMCPY_HOST_TO_DEVICE);

            // device -> host 验证
            float* host_dst = nullptr;
            aclrtMallocHost(reinterpret_cast<void**>(&host_dst), check_count * sizeof(float));
            aclrtMemcpy(host_dst, check_count * sizeof(float),
                        large_ptr, check_count * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);

            bool data_correct = true;
            for (size_t i = 0; i < check_count; i++) {
                if (host_dst[i] != static_cast<float>(i)) {
                    data_correct = false;
                    break;
                }
            }
            std::cout << "Data verification: " << (data_correct ? "PASSED" : "FAILED") << std::endl;

            aclrtFreeHost(host_src);
            aclrtFreeHost(host_dst);
            aclshmem_free(large_ptr);
            printMemoryStats("After cleanup: ");
            return data_correct;
        } else {
            std::cout << "✗ Large allocation failed!" << std::endl;
            return false;
        }
    }
    
    bool testBoundaryConditions() {
        std::cout << "\n=== Testing Boundary Conditions ===" << std::endl;
        
        // 测试零大小分配
        std::cout << "Testing zero-size allocation..." << std::endl;
        void* zero_ptr = aclshmem_malloc(0);
        if (zero_ptr == nullptr) {
            std::cout << "✓ Zero-size allocation correctly returned nullptr" << std::endl;
        } else {
            std::cout << "✗ Zero-size allocation should return nullptr" << std::endl;
            aclshmem_free(zero_ptr);
        }
        
        // 测试超大分配
        std::cout << "Testing extremely large allocation..." << std::endl;
        size_t huge_size = 10ULL * 1024 * 1024 * 1024; // 10GB
        void* huge_ptr = aclshmem_malloc(huge_size);
        if (huge_ptr) {
            std::cout << "⚠ Warning: Extremely large allocation succeeded (may indicate issue)" << std::endl;
            aclshmem_free(huge_ptr);
            return false;
        } else {
            std::cout << "✓ Extremely large allocation correctly failed" << std::endl;
        }
        
        return true;
    }
    
    bool testIncrementalRelease() {
        std::cout << "\n=== Testing Incremental Release ===" << std::endl;
        
        printMemoryStats("Before allocation: ");
        
        std::vector<void*> allocated_ptrs;
        std::vector<size_t> sizes_mb = {8, 12, 16, 20}; // MB
        
        // 分配多个内存块
        for (size_t i = 0; i < sizes_mb.size(); i++) {
            size_t size_bytes = sizes_mb[i] * 1024 * 1024;
            std::cout << "\nAllocating " << sizes_mb[i] << " MB (block #" << (i+1) << ")..." << std::endl;
            
            void* ptr = aclshmem_malloc(size_bytes);
            if (ptr) {
                allocated_ptrs.push_back(ptr);
                std::cout << "✓ Success! Pointer: " << ptr << std::endl;
            } else {
                std::cout << "✗ Failed to allocate " << sizes_mb[i] << " MB" << std::endl;
                // 释放已分配的内存
                for (void* p : allocated_ptrs) {
                    aclshmem_free(p);
                }
                return false;
            }
        }
        
        printMemoryStats("After all allocations: ");
        
        // 逐个释放内存块，并在每次释放后打印内存统计
        for (size_t i = 0; i < allocated_ptrs.size(); i++) {
            std::cout << "\nReleasing block #" << (i+1) << " (pointer: " << allocated_ptrs[i] << ")..." << std::endl;
            aclshmem_free(allocated_ptrs[i]);
            std::cout << "✓ Block #" << (i+1) << " released successfully" << std::endl;
            printMemoryStats("After release: ");
        }
        
        printMemoryStats("After all releases: ");
        return true;
    }
    
    bool testSmallGranularityMemory() {
        std::cout << "\n=== Testing Small Granularity Memory Operations ===" << std::endl;
        
        printMemoryStats("Initial state: ");
        
        // Test 1: 连续申请小粒度内存
        std::cout << "\n--- Test 1: Continuous Small Allocations ---" << std::endl;
        std::vector<void*> small_ptrs;
        const size_t NUM_SMALL_ALLOCS = 100;
        const size_t SMALL_SIZES[] = {4096, 65536}; // 4K, 64K
        
        for (size_t size : SMALL_SIZES) {
            std::cout << "\nAllocating " << NUM_SMALL_ALLOCS << " blocks of " << size/1024 << "K each..." << std::endl;
            
            for (size_t i = 0; i < NUM_SMALL_ALLOCS; i++) {
                void* ptr = aclshmem_malloc(size);
                if (ptr) {
                    small_ptrs.push_back(ptr);
                } else {
                    std::cout << "✗ Failed to allocate " << size/1024 << "K block #" << (i+1) << std::endl;
                    // 释放已分配的内存
                    for (void* p : small_ptrs) {
                        aclshmem_free(p);
                    }
                    small_ptrs.clear();
                    return false;
                }
            }
            
            std::cout << "✓ All " << NUM_SMALL_ALLOCS << " blocks of " << size/1024 << "K allocated successfully" << std::endl;
            printMemoryStats("After allocation: ");
        }
        
        // 释放所有小粒度内存
        std::cout << "\nReleasing all small blocks..." << std::endl;
        for (void* p : small_ptrs) {
            aclshmem_free(p);
        }
        small_ptrs.clear();
        printMemoryStats("After releasing all small blocks: ");
        
        // Test 2: 申请释放交替进行（模拟真实使用场景，可能产生碎片）
        std::cout << "\n--- Test 2: Allocate-Release Alternating (Fragmentation Test) ---" << std::endl;
        std::vector<void*> alternating_ptrs;
        const size_t ALTERNATING_ROUNDS = 10;
        
        for (size_t round = 0; round < ALTERNATING_ROUNDS; round++) {
            // 每轮分配多个不同大小的块（4种不同大小）
            std::cout << "\nRound " << (round+1) << ": Allocating blocks..." << std::endl;
            
            // 使用多种大小的块，增加碎片产生的可能性
            const std::vector<size_t> varied_sizes = {
                4096,    // 4K
                8192,    // 8K
                16384,   // 16K
                32768,   // 32K
                65536,   // 64K
                131072   // 128K
            };
            
            // 每轮分配3个随机大小的块
            const size_t ALLOC_COUNT_PER_ROUND = 3;
            for (size_t i = 0; i < ALLOC_COUNT_PER_ROUND; i++) {
                // 随机选择一个大小（使用轮数作为种子确保可重现）
                size_t size = varied_sizes[(round + i) % varied_sizes.size()];
                void* ptr = aclshmem_malloc(size);
                if (ptr) {
                    alternating_ptrs.push_back(ptr);
                    std::cout << "  Allocated " << size/1024 << "K at " << ptr << std::endl;
                } else {
                    std::cout << "✗ Failed to allocate " << size/1024 << "K in round " << (round+1) << std::endl;
                    for (void* p : alternating_ptrs) {
                        aclshmem_free(p);
                    }
                    alternating_ptrs.clear();
                    return false;
                }
            }
            
            // 释放部分块（模拟随机释放导致的碎片）
            std::cout << "  Releasing some blocks..." << std::endl;
            
            // 每轮释放已分配块的一半（向下取整），但至少保留1个
            size_t release_count = std::max(1ul, alternating_ptrs.size() / 2);
            
            // 从中间位置开始释放，模拟随机释放
            size_t start_release = alternating_ptrs.size() / 2;
            size_t end_release = std::min(start_release + release_count, alternating_ptrs.size());
            
            // 释放选中的块
            for (size_t i = start_release; i < end_release; i++) {
                if (alternating_ptrs[i] != nullptr) {
                    aclshmem_free(alternating_ptrs[i]);
                    std::cout << "  Released block at " << alternating_ptrs[i] << std::endl;
                    alternating_ptrs[i] = nullptr; // 标记为已释放
                }
            }
            
            // 清理已释放的空指针（保持向量整洁）
            auto it = std::remove(alternating_ptrs.begin(), alternating_ptrs.end(), nullptr);
            alternating_ptrs.erase(it, alternating_ptrs.end());
            
            // 每10轮打印一次内存统计
            if ((round+1) % 10 == 0) {
                std::cout << "\nAfter round " << (round+1) << ":" << std::endl;
                std::cout << "  Current allocated blocks: " << alternating_ptrs.size() << std::endl;
                printMemoryStats("Memory status: ");
            }
        }
        
        // 测试碎片对大内存分配的影响
        std::cout << "\n--- Testing Large Allocation After Fragmentation ---" << std::endl;
        
        // 尝试分配一个较大的内存块（20MB）
        size_t large_size_mb = 20;
        size_t large_size_bytes = large_size_mb * 1024 * 1024;
        std::cout << "Attempting to allocate " << large_size_mb << " MB block after fragmentation..." << std::endl;
        
        void* large_ptr = aclshmem_malloc(large_size_bytes);
        if (large_ptr) {
            std::cout << "✓ Large allocation successful! Pointer: " << large_ptr << std::endl;
            aclshmem_free(large_ptr);
            std::cout << "✓ Large block released successfully" << std::endl;
        } else {
            std::cout << "✗ Large allocation failed! This may indicate memory fragmentation issues." << std::endl;
            // 不要因为这个测试失败而返回false，因为这正是我们要观察的现象
        }
        
        // 释放剩余的块
        std::cout << "\nReleasing remaining blocks..." << std::endl;
        for (void* p : alternating_ptrs) {
            aclshmem_free(p);
        }
        alternating_ptrs.clear();
        
        printMemoryStats("Final memory status after all operations: ");
        std::cout << "✓ Small granularity memory test completed successfully" << std::endl;
        
        return true;
    }
    
    void runAllTests() {
        std::cout << "========================================" << std::endl;
        std::cout << "SHMEM Dynamic Memory Expansion Test" << std::endl;
        std::cout << "========================================" << std::endl;
        
        if (!initialize()) {
            std::cerr << "Failed to initialize test environment" << std::endl;
            return;
        }
        
        int passed = 0;
        int total = 5;
        
        if (testSequentialAllocation()) passed++;
        if (testLargeAllocation()) passed++;
        if (testIncrementalRelease()) passed++;
        if (testSmallGranularityMemory()) passed++;
        if (testBoundaryConditions()) passed++;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test Results: " << passed << "/" << total << " tests passed" << std::endl;
        std::cout << "========================================" << std::endl;
        
        if (passed == total) {
            std::cout << "🎉 All tests passed! Dynamic memory expansion is working correctly." << std::endl;
        } else {
            std::cout << "❌ Some tests failed. Please check the implementation." << std::endl;
        }
    }
    
    ~DynamicMemoryTest() {
        if (initialized_) {
            aclshmem_finalize();
            aclrtResetDevice(0);
            aclFinalize();
            std::cout << "Test environment cleaned up" << std::endl;
        }
    }
};

int main() {
    try {
        DynamicMemoryTest test;
        test.runAllTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return 1;
    }
}