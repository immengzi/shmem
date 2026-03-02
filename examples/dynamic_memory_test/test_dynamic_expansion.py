#!/usr/bin/env python3
# coding=utf-8
"""
SHMEM动态内存扩容功能测试脚本

该脚本用于测试SHMEM内存池的动态扩容能力，
验证当内存不足时能否自动调用CANN接口进行扩容。
"""

import os
import sys
import ctypes
import torch
import torch_npu
import numpy as np
import time
from typing import List, Tuple

# 设置环境变量
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"

class DynamicMemoryTester:
    def __init__(self):
        self.shmem_lib = None
        self.initialized = False
        
    def load_shmem_library(self):
        """加载SHMEM动态库"""
        try:
            # 尝试不同的可能路径
            lib_paths = [
                "./build/libshmem.so",
                "./build/libshmem.dylib",
                "/usr/local/lib/libshmem.so",
                "/usr/lib/libshmem.so"
            ]
            
            for lib_path in lib_paths:
                if os.path.exists(lib_path):
                    self.shmem_lib = ctypes.CDLL(lib_path)
                    print(f"Successfully loaded SHMEM library from: {lib_path}")
                    return True
                    
            print("Warning: Could not find SHMEM library, using standard allocation")
            return False
            
        except Exception as e:
            print(f"Failed to load SHMEM library: {e}")
            return False
    
    def setup_shmem_allocator(self):
        """设置SHMEM分配器"""
        if not self.shmem_lib:
            return False
            
        try:
            # 初始化SHMEM（小内存池，便于测试扩容）
            init_attr = self.shmem_lib.aclshmemx_init_attr_t()
            init_attr.version = 1
            init_attr.my_rank = 0
            init_attr.n_ranks = 1
            init_attr.local_mem_size = 64 * 1024 * 1024  # 64MB初始内存池
            
            # 启用动态扩容
            self.shmem_lib.aclshmem_enable_dynamic_expansion(ctypes.c_bool(True))
            
            ret = self.shmem_lib.aclshmemx_init_attr(0, ctypes.byref(init_attr))
            if ret == 0:
                self.initialized = True
                print("SHMEM allocator initialized with dynamic expansion enabled")
                return True
            else:
                print(f"SHMEM initialization failed with code: {ret}")
                return False
                
        except Exception as e:
            print(f"Error setting up SHMEM allocator: {e}")
            return False
    
    def get_memory_stats(self) -> Tuple[int, int, int]:
        """获取内存使用统计"""
        if not self.shmem_lib:
            return (0, 0, 0)
            
        try:
            total_capacity = ctypes.c_uint64()
            used_memory = ctypes.c_uint64()
            available_memory = ctypes.c_uint64()
            
            self.shmem_lib.aclshmem_get_memory_stats(
                ctypes.byref(total_capacity),
                ctypes.byref(used_memory),
                ctypes.byref(available_memory)
            )
            
            return (total_capacity.value, used_memory.value, available_memory.value)
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return (0, 0, 0)
    
    def print_memory_stats(self, prefix: str = ""):
        """打印内存统计信息"""
        total, used, available = self.get_memory_stats()
        print(f"{prefix}Memory Stats - Total: {total/1024/1024:.2f}MB, "
              f"Used: {used/1024/1024:.2f}MB, Available: {available/1024/1024:.2f}MB")
    
    def test_sequential_allocation(self, sizes_mb: List[float]):
        """测试顺序内存分配"""
        print(f"\n=== 测试顺序内存分配 (sizes: {sizes_mb} MB) ===")
        
        self.print_memory_stats("Before allocation: ")
        allocated_ptrs = []
        
        try:
            for i, size_mb in enumerate(sizes_mb):
                size_bytes = int(size_mb * 1024 * 1024)
                print(f"\nAllocating {size_mb} MB (attempt {i+1})...")
                
                # 记录分配前的状态
                _, _, available_before = self.get_memory_stats()
                print(f"Available before allocation: {available_before/1024/1024:.2f}MB")
                
                # 执行分配
                start_time = time.time()
                if self.shmem_lib:
                    ptr = self.shmem_lib.aclshmem_malloc(size_bytes)
                else:
                    # 回退到标准分配
                    ptr = ctypes.pythonapi.PyMem_Malloc(size_bytes)
                
                alloc_time = time.time() - start_time
                
                if ptr:
                    allocated_ptrs.append((ptr, size_bytes))
                    print(f"✓ Allocation successful! Pointer: {ptr}, Time: {alloc_time*1000:.2f}ms")
                    self.print_memory_stats("After allocation: ")
                else:
                    print(f"✗ Allocation failed for {size_mb} MB")
                    break
                    
        except Exception as e:
            print(f"Error during sequential allocation: {e}")
        
        # 释放内存
        print(f"\nReleasing {len(allocated_ptrs)} allocated blocks...")
        for ptr, size in allocated_ptrs:
            if self.shmem_lib:
                self.shmem_lib.aclshmem_free(ptr)
            else:
                ctypes.pythonapi.PyMem_Free(ptr)
        
        self.print_memory_stats("After cleanup: ")
        return len(allocated_ptrs) == len(sizes_mb)
    
    def test_large_single_allocation(self, size_mb: float):
        """测试大块单次分配"""
        print(f"\n=== 测试大块单次分配 ({size_mb} MB) ===")
        
        self.print_memory_stats("Before allocation: ")
        size_bytes = int(size_mb * 1024 * 1024)
        
        try:
            start_time = time.time()
            if self.shmem_lib:
                ptr = self.shmem_lib.aclshmem_malloc(size_bytes)
            else:
                ptr = ctypes.pythonapi.PyMem_Malloc(size_bytes)
            alloc_time = time.time() - start_time
            
            if ptr:
                print(f"✓ Large allocation successful! Pointer: {ptr}, Time: {alloc_time*1000:.2f}ms")
                self.print_memory_stats("After allocation: ")
                
                # 写入测试数据
                print("Writing test data...")
                test_data = (ctypes.c_float * (size_bytes // 4))()
                for i in range(min(1000, len(test_data))):
                    test_data[i] = float(i)
                
                if self.shmem_lib:
                    ctypes.memmove(ptr, test_data, min(size_bytes, 4000))
                else:
                    # 对于Python分配的内存，直接操作
                    pass
                
                # 释放内存
                if self.shmem_lib:
                    self.shmem_lib.aclshmem_free(ptr)
                else:
                    ctypes.pythonapi.PyMem_Free(ptr)
                
                self.print_memory_stats("After cleanup: ")
                return True
            else:
                print(f"✗ Large allocation failed for {size_mb} MB")
                return False
                
        except Exception as e:
            print(f"Error during large allocation test: {e}")
            return False
    
    def test_pytorch_integration(self):
        """测试与PyTorch的集成"""
        print("\n=== 测试PyTorch集成 ===")
        
        try:
            # 创建小张量（应该在初始内存池中）
            print("Creating small tensor (should use initial pool)...")
            small_tensor = torch.randn(1000, 1000, device='npu')
            print(f"Small tensor created: {small_tensor.shape}")
            
            self.print_memory_stats("After small tensor: ")
            
            # 创建大张量（可能触发扩容）
            print("Creating large tensor (may trigger expansion)...")
            large_tensor = torch.randn(8000, 8000, device='npu')  # 约244MB
            print(f"Large tensor created: {large_tensor.shape}")
            
            self.print_memory_stats("After large tensor: ")
            
            # 执行计算
            print("Performing computation...")
            result = torch.matmul(small_tensor, small_tensor.T)
            print(f"Computation result shape: {result.shape}")
            
            # 清理
            del small_tensor, large_tensor, result
            torch_npu.npu.empty_cache()
            
            self.print_memory_stats("After cleanup: ")
            print("✓ PyTorch integration test completed successfully")
            return True
            
        except Exception as e:
            print(f"Error during PyTorch integration test: {e}")
            return False
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("=" * 60)
        print("SHMEM Dynamic Memory Expansion Test Suite")
        print("=" * 60)
        
        # 加载库
        if not self.load_shmem_library():
            print("Proceeding with standard memory allocation tests...")
        
        # 初始化SHMEM
        if self.shmem_lib:
            if not self.setup_shmem_allocator():
                print("Failed to initialize SHMEM allocator")
                return False
        
        test_results = []
        
        # 测试1: 顺序分配测试
        sequential_sizes = [10, 20, 30, 50]  # MB
        result1 = self.test_sequential_allocation(sequential_sizes)
        test_results.append(("Sequential Allocation", result1))
        
        # 测试2: 大块分配测试
        result2 = self.test_large_single_allocation(100)  # 100MB
        test_results.append(("Large Single Allocation", result2))
        
        # 测试3: PyTorch集成测试
        if self.initialized:
            result3 = self.test_pytorch_integration()
            test_results.append(("PyTorch Integration", result3))
        else:
            print("Skipping PyTorch test (SHMEM not initialized)")
            test_results.append(("PyTorch Integration", False))
        
        # 打印总结
        print("\n" + "=" * 60)
        print("Test Results Summary:")
        print("=" * 60)
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "PASS" if result else "FAIL"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if self.shmem_lib and self.initialized:
            # 显示最终统计
            print("\nFinal Memory Statistics:")
            self.print_memory_stats()
            
            # 清理未使用内存
            print("\nCleaning up unused memory...")
            self.shmem_lib.aclshmem_cleanup_unused_memory()
            self.print_memory_stats("After cleanup: ")
        
        return passed == total

def main():
    """主函数"""
    tester = DynamicMemoryTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n🎉 All tests passed! Dynamic memory expansion is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())