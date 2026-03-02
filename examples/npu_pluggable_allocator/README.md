# SHMEM NPUPluggableAllocator 集成指南

## 概述

本文档详细介绍如何将SHMEM（Shared Memory）集成到PyTorch NPU的可插拔分配器系统中，实现高效的分布式内存管理。

## 架构设计

### 核心组件

1. **SHMEM Pluggable Allocator (C++)**: 
   - 实现标准的NPUPluggableAllocator接口
   - 封装SHMEM内存分配/释放功能
   - 提供回退机制到标准ACL分配器

2. **Python绑定层**:
   - 使用ctypes加载C++扩展
   - 提供易用的Python API
   - 支持上下文管理器模式

3. **分布式支持**:
   - 集成SHMEM的团队管理功能
   - 支持多进程间的内存共享
   - 提供远程内存访问能力

## 使用方法

### 1. 环境准备

```bash
# 设置必要的环境变量
export PYTHONPATH=/path/to/pytorch_npu:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH

# 安装依赖
pip install torch torch_npu
```

### 2. 构建扩展

```bash
# 方法1: 使用CMake (推荐)
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=`python -c 'import torch; print(torch.utils.cmake_prefix_path)'`
make

# 方法2: 直接使用g++
g++ -shared -fPIC \
    -I/path/to/pytorch/include \
    -I/path/to/torch_npu/include \
    -I../../include \
    -L/usr/local/Ascend/ascend-toolkit/latest/lib64 \
    -lascendcl -lacl_shmem \
    -o shmem_pluggable_allocator.so \
    shmem_pluggable_allocator.cpp
```

### 3. 基本使用示例

```python
import torch
import torch_npu
import ctypes

# 加载SHMEM分配器扩展
myallocator = ctypes.CDLL('./build/shmem_pluggable_allocator.so')

# 获取函数指针
malloc_fn = ctypes.cast(getattr(myallocator, "shmem_malloc"), ctypes.c_void_p).value
free_fn = ctypes.cast(getattr(myallocator, "shmem_free"), ctypes.c_void_p).value

# 创建NPUPluggableAllocator实例
shmem_allocator = torch_npu.npu.memory.NPUPluggableAllocator(
    './build/shmem_pluggable_allocator.so',
    'shmem_malloc',
    'shmem_free'
)

# 切换到SHMEM分配器
torch_npu.npu.memory.change_current_allocator(shmem_allocator)

# 在专用内存池中执行操作
with torch_npu.npu.use_mem_pool(torch_npu.npu.MemPool(shmem_allocator._allocator)):
    # 创建张量
    tensor = torch.randn(1000, 1000, device='npu')
    
    # 执行计算
    result = torch.matmul(tensor, tensor.T)
    
    print(f"Result shape: {result.shape}")

# 检查是否使用了SHMEM分配器
check_fn = getattr(myallocator, "check_shmem_allocator_used")
if check_fn():
    print("Successfully used SHMEM allocator!")
```

### 4. 分布式场景使用

```python
import shmem as shm
import torch
import torch_npu

# 初始化SHMEM分布式环境
shm.set_conf_store_tls(False, "")  # 内网环境关闭TLS
uid = shm.get_unique_id()

# 在多进程环境中初始化（每个进程）
shm.init(rank=local_rank, nranks=world_size, uid=uid, mem_size=2*1024*1024*1024)

# 创建分布式张量
dist_tensor = shm.aclshmem_create_tensor((2000, 2000), dtype=torch.float32)

# 获取远程PE的内存地址
remote_buffer = shm.get_peer_buffer(dist_tensor.data_ptr(), remote_pe_rank)

# 执行远程内存操作
shm.put(remote_dest_buffer, local_source_buffer, remote_pe_rank)

# 清理
shm.aclshmem_free_tensor(dist_tensor)
shm.finalize()
```

## 高级特性

### 1. 内存池管理

```python
# 创建专用内存池
mem_pool = torch_npu.npu.MemPool(shmem_allocator._allocator)

# 在特定内存池中分配
with torch_npu.npu.use_mem_pool(mem_pool):
    # 所有在此上下文中的分配都使用SHMEM
    large_tensor = torch.randn(10000, 10000, device='npu')
```

### 2. 性能监控

```python
# 设置统计函数
stats_fn = ctypes.cast(getattr(myallocator, "shmem_get_device_stats"), ctypes.c_void_p).value
reset_fn = ctypes.cast(getattr(myallocator, "shmem_reset_peak_status"), ctypes.c_void_p).value

shmem_allocator.allocator().set_get_device_stats_fn(stats_fn)
shmem_allocator.allocator().set_reset_peak_status_fn(reset_fn)

# 获取统计信息
stats = torch_npu.npu.memory_stats()
print(f"SHMEM allocation stats: {stats}")
```

### 3. 错误处理和回退

```python
# SHMEM分配器会自动回退到标准ACL分配器
# 当SHMEM不可用时不会导致程序崩溃

try:
    # 尝试使用SHMEM分配器
    tensor = torch.randn(large_size, device='npu')
    # 如果SHMEM分配失败，会自动使用标准分配器
except RuntimeError as e:
    print(f"Allocation failed: {e}")
    # 处理分配失败的情况
```

## 最佳实践

### 1. 性能优化建议

- **批量分配**: 尽量批量分配大块内存而不是频繁的小分配
- **内存池复用**: 使用内存池避免重复初始化开销
- **适当的内存大小**: 根据实际需求设置合适的共享内存大小

### 2. 分布式场景建议

- **团队管理**: 合理使用team_split功能组织通信组
- **内存对齐**: 注意跨设备内存访问的对齐要求
- **同步机制**: 正确使用信号量和等待机制

### 3. 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查分配器状态
print(f"Allocator initialized: {shmem_allocator.allocator().initialized()}")

# 监控内存使用
torch_npu.npu.memory._set_allocator_settings("expandable_segments:False")
```

## 故障排除

### 常见问题

1. **扩展加载失败**
   ```bash
   # 检查库路径
   ldd shmem_pluggable_allocator.so
   
   # 确保所有依赖库可见
   export LD_LIBRARY_PATH=/path/to/dependencies:$LD_LIBRARY_PATH
   ```

2. **内存分配失败**
   ```python
   # 检查SHMEM初始化状态
   status = shm.aclshmemx_init_status()
   print(f"SHMEM status: {status}")
   
   # 尝试增加共享内存大小
   shm.init(mem_size=4*1024*1024*1024)  # 4GB
   ```

3. **分布式同步问题**
   ```python
   # 检查团队配置
   team_size = shm.team_n_pes(team_id)
   my_rank = shm.team_my_pe(team_id)
   
   # 使用信号量确保同步
   signal_var = shm.buffer(8)  # 8字节信号量
   shm.signal_wait(signal_var, expected_value, shm.ComparisonType.CMP_EQ)
   ```

## API参考

### 核心函数

| 函数名 | 描述 | 参数 |
|--------|------|------|
| `shmem_malloc` | SHMEM内存分配 | size, device, stream |
| `shmem_free` | SHMEM内存释放 | ptr, size, device, stream |
| `shmem_get_device_stats` | 获取设备统计 | device |
| `shmem_reset_peak_status` | 重置峰值统计 | device |

### Python接口

```python
class SHMEMPluggableAllocator:
    def __init__(self, lib_path, malloc_func, free_func):
        """初始化SHMEM分配器"""
    
    def allocator(self):
        """获取底层分配器对象"""
    
    def set_get_device_stats_fn(self, func_ptr):
        """设置统计函数"""
    
    def set_reset_peak_status_fn(self, func_ptr):
        """设置重置函数"""
```

## 许可证

本项目遵循CANN Open Software License Agreement Version 2.0。

## 贡献

欢迎提交issue和pull request来改进这个集成方案。