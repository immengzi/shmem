#!/usr/bin/env python3
# coding=utf-8
"""
SHMEM NPUPluggableAllocator 使用示例

这个示例展示了如何将SHMEM作为NPUPluggableAllocator使用，
实现高效的分布式内存管理。
"""

import os
import torch
import torch_npu
import ctypes
import subprocess
import shutil
from pathlib import Path

# 设置环境变量
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"

class SHMEMPluggableAllocatorExample:
    def __init__(self):
        self.shmem_allocator = None
        self.build_directory = "build"
        self.extension_path = None
        
    def build_extension(self):
        """构建SHMEM可插拔分配器扩展"""
        print("Building SHMEM pluggable allocator extension...")
        
        # 创建构建目录
        build_path = Path(self.build_directory)
        if build_path.exists():
            shutil.rmtree(build_path)
        build_path.mkdir(exist_ok=True)
        
        # 获取必要的路径
        pytorch_path = os.path.dirname(os.path.realpath(torch.__file__))
        torch_npu_path = os.path.dirname(os.path.realpath(torch_npu.__file__))
        
        # 构建命令
        build_cmd = [
            "g++", "-shared", "-fPIC",
            "-I" + pytorch_path + "/include",
            "-I" + torch_npu_path + "/include",
            "-I../../include",  # SHMEM头文件路径
            "-L/usr/local/Ascend/ascend-toolkit/latest/lib64",  # CANN库路径
            "-lascendcl", "-lacl_shmem",
            "-o", str(build_path / "shmem_pluggable_allocator.so"),
            "shmem_pluggable_allocator.cpp"
        ]
        
        try:
            result = subprocess.run(build_cmd, capture_output=True, text=True, cwd=".")
            if result.returncode != 0:
                print(f"Build failed: {result.stderr}")
                return False
            print("Extension built successfully")
            self.extension_path = str(build_path / "shmem_pluggable_allocator.so")
            return True
        except Exception as e:
            print(f"Build error: {e}")
            return False
    
    def load_shmem_allocator(self):
        """加载SHMEM分配器"""
        if not self.extension_path or not os.path.exists(self.extension_path):
            print("Extension not built, building now...")
            if not self.build_extension():
                return False
        
        try:
            # 加载共享库
            myallocator = ctypes.CDLL(self.extension_path)
            
            # 获取函数指针
            malloc_fn = ctypes.cast(getattr(myallocator, "shmem_malloc"), ctypes.c_void_p).value
            free_fn = ctypes.cast(getattr(myallocator, "shmem_free"), ctypes.c_void_p).value
            
            # 创建NPUPluggableAllocator
            self.shmem_allocator = torch_npu.npu.memory.NPUPluggableAllocator(
                self.extension_path, 
                'shmem_malloc', 
                'shmem_free'
            )
            
            # 设置额外的函数
            get_device_stats_fn = ctypes.cast(getattr(myallocator, "shmem_get_device_stats"), ctypes.c_void_p).value
            reset_peak_status_fn = ctypes.cast(getattr(myallocator, "shmem_reset_peak_status"), ctypes.c_void_p).value
            
            self.shmem_allocator.allocator().set_get_device_stats_fn(get_device_stats_fn)
            self.shmem_allocator.allocator().set_reset_peak_status_fn(reset_peak_status_fn)
            
            print("SHMEM allocator loaded successfully")
            return True
            
        except Exception as e:
            print(f"Failed to load SHMEM allocator: {e}")
            return False
    
    def use_shmem_allocator(self):
        """使用SHMEM分配器进行内存分配"""
        if not self.shmem_allocator:
            print("SHMEM allocator not loaded")
            return False
            
        try:
            # 切换到SHMEM分配器
            print("Switching to SHMEM allocator...")
            torch_npu.npu.memory.change_current_allocator(self.shmem_allocator)
            
            # 在专用内存池中执行操作
            with torch_npu.npu.use_mem_pool(torch_npu.npu.MemPool(self.shmem_allocator._allocator)):
                print("Creating tensors with SHMEM allocator...")
                
                # 创建一些张量来测试分配器
                tensor1 = torch.randn(1000, 1000, device='npu')
                tensor2 = torch.randn(500, 500, device='npu')
                
                # 执行一些计算
                result = torch.matmul(tensor1, tensor2)
                print(f"Computation result shape: {result.shape}")
                
                # 清理
                del tensor1, tensor2, result
                
            print("SHMEM allocator usage completed")
            return True
            
        except Exception as e:
            print(f"Error using SHMEM allocator: {e}")
            return False
    
    def demonstrate_distributed_usage(self):
        """演示分布式场景下的使用"""
        print("\n=== 分布式SHMEM分配器演示 ===")
        
        try:
            import shmem as shm
            
            # 初始化SHMEM（分布式场景）
            shm.set_conf_store_tls(False, "")  # 关闭TLS认证
            
            # 获取唯一ID用于分布式初始化
            uid = shm.get_unique_id()
            print(f"Unique ID: {uid}")
            
            # 初始化SHMEM
            shm.init(rank=0, nranks=1, uid=uid, mem_size=1024*1024*1024)  # 1GB内存
            
            # 使用SHMEM创建张量
            shmem_tensor = shm.aclshmem_create_tensor((1000, 1000), dtype=torch.float32)
            print(f"SHMEM tensor created: {shmem_tensor.shape}")
            
            # 执行计算
            result = torch.sin(shmem_tensor)
            print(f"SHMEM computation completed, result shape: {result.shape}")
            
            # 清理
            shm.aclshmem_free_tensor(shmem_tensor)
            shm.finalize()
            
            print("Distributed SHMEM usage demonstrated successfully")
            
        except Exception as e:
            print(f"Distributed SHMEM demonstration failed: {e}")
    
    def run_complete_example(self):
        """运行完整的示例"""
        print("=== SHMEM NPUPluggableAllocator 完整示例 ===\n")
        
        # 1. 构建扩展
        if not self.build_extension():
            print("构建失败，退出示例")
            return
            
        # 2. 加载分配器
        if not self.load_shmem_allocator():
            print("加载分配器失败，退出示例")
            return
            
        # 3. 使用SHMEM分配器
        self.use_shmem_allocator()
        
        # 4. 演示分布式使用
        self.demonstrate_distributed_usage()
        
        print("\n=== 示例完成 ===")

def main():
    """主函数"""
    example = SHMEMPluggableAllocatorExample()
    example.run_complete_example()

if __name__ == "__main__":
    main()