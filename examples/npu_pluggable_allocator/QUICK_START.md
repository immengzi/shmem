# SHMEM NPUPluggableAllocator 快速开始指南

## 🚀 三步快速上手

### 步骤1: 环境准备
```bash
# 设置环境变量
export PYTHONPATH=/path/to/pytorch_npu:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
```

### 步骤2: 一键构建和运行
```bash
# Linux/macOS
chmod +x run_example.sh
./run_example.sh

# Windows
run_example.bat
```

### 步骤3: 手动使用
```python
import torch
import torch_npu
import ctypes

# 加载分配器
allocator_lib = ctypes.CDLL('./build/shmem_pluggable_allocator.so')
shmem_allocator = torch_npu.npu.memory.NPUPluggableAllocator(
    './build/shmem_pluggable_allocator.so',
    'shmem_malloc',
    'shmem_free'
)

# 使用SHMEM分配器
torch_npu.npu.memory.change_current_allocator(shmem_allocator)
tensor = torch.randn(1000, 1000, device='npu')
```

## 📋 主要特性

✅ **无缝集成**: 完全兼容PyTorch NPU的内存管理接口  
✅ **自动回退**: SHMEM不可用时自动回退到标准分配器  
✅ **分布式支持**: 支持多进程间的高效内存共享  
✅ **性能监控**: 提供详细的内存使用统计  

## 🔧 核心API

```python
# 基本使用
shmem_allocator = NPUPluggableAllocator(lib_path, malloc_func, free_func)

# 内存池管理
with torch_npu.npu.use_mem_pool(MemPool(allocator)):
    # 在SHMEM内存池中执行操作
    pass

# 分布式使用
import shmem as shm
shm.init(rank=0, nranks=1, mem_size=1024*1024*1024)
tensor = shm.aclshmem_create_tensor((1000, 1000))
```

## 📖 详细文档

查看 [README.md](README.md) 获取完整的使用说明和API参考。

## 🐛 常见问题

**Q: 构建失败怎么办？**
A: 确保安装了正确的编译器和依赖库，检查环境变量设置

**Q: 运行时报错找不到库？**
A: 检查LD_LIBRARY_PATH是否包含了CANN库路径

**Q: 如何调试内存分配问题？**
A: 启用详细日志：`export LOG_LEVEL=DEBUG`

---
*更多详情请查看完整文档*