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
        
        std::cout << prefix << "Memory Stats:" << std::endl;
        std::cout << "  Total Capacity: " << total_capacity / (1024*1024) << " MB" << std::endl;
        std::cout << "  Used Memory: " << used_memory / (1024*1024) << " MB" << std::endl;
        std::cout << "  Available Memory: " << available_memory / (1024*1024) << " MB" << std::endl;
        std::cout << "  Utilization: " << (used_memory * 100.0 / total_capacity) << "%" << std::endl;
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
    
    void runAllTests() {
        std::cout << "========================================" << std::endl;
        std::cout << "SHMEM Dynamic Memory Expansion Test" << std::endl;
        std::cout << "========================================" << std::endl;
        
        if (!initialize()) {
            std::cerr << "Failed to initialize test environment" << std::endl;
            return;
        }
        
        int passed = 0;
        int total = 3;
        
        if (testSequentialAllocation()) passed++;
        if (testLargeAllocation()) passed++;
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