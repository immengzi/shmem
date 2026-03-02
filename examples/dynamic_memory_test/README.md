# SHMEM动态内存扩容功能使用指南

## 📋 概述

本功能实现了SHMEM内存池的动态扩容能力，当分配的内存超过初始内存池大小时，会自动调用CANN相关接口进行内存分配，并更新当前的管理结构。

## 🚀 主要特性

✅ **自动扩容**: 内存不足时自动调用CANN接口分配新内存块  
✅ **智能管理**: 统一管理初始内存池和动态分配的内存块  
✅ **性能优化**: 使用合理的扩容策略避免频繁扩容  
✅ **兼容性**: 完全兼容现有的SHMEM API接口  
✅ **可配置**: 支持启用/禁用动态扩容功能  

## 🛠️ 核心接口

### 动态扩容控制接口

```cpp
// 启用/禁用动态扩容功能
void aclshmem_enable_dynamic_expansion(bool enable);

// 检查动态扩容是否启用
bool aclshmem_is_dynamic_expansion_enabled();

// 获取内存使用统计信息
void aclshmem_get_memory_stats(uint64_t* total_capacity, uint64_t* used_memory, uint64_t* available_memory);

// 清理未使用的内存块
void aclshmem_cleanup_unused_memory();
```

### 内存分配接口（保持兼容）

```cpp
// 基础分配接口（自动使用动态扩容）
void* aclshmem_malloc(size_t size);
void* aclshmem_calloc(size_t nmemb, size_t size);
void* aclshmem_align(size_t alignment, size_t size);
void aclshmem_free(void* ptr);

// 扩展分配接口
void* aclshmemx_malloc(size_t size, aclshmem_mem_type_t mem_type);
void* aclshmemx_calloc(size_t nmemb, size_t size, aclshmem_mem_type_t mem_type);
void* aclshmemx_align(size_t alignment, size_t size, aclshmem_mem_type_t mem_type);
void aclshmemx_free(void* ptr, aclshmem_mem_type_t mem_type);
```

## 📊 扩容策略

### 扩容算法

1. **最小扩容大小**: 256MB
2. **扩容因子**: 1.5倍当前所需大小
3. **最大单块大小**: 4GB
4. **内存对齐**: 256KB边界对齐

### 内存块管理

- **初始内存池**: 使用原有的固定大小内存池
- **动态内存块**: 通过CANN `aclrtMalloc` 接口分配
- **统一管理**: 所有内存块由动态内存管理器统一跟踪和管理

## 🔧 使用方法

### 1. 基本使用

```cpp
#include "shmem/host/mem/aclshmem_mem.h"

// 初始化SHMEM（小内存池，便于测试扩容）
aclshmemx_init_attr_t attr = {0};
attr.version = 1;
attr.my_rank = 0;
attr.n_ranks = 1;
attr.local_mem_size = 64 * 1024 * 1024; // 64MB初始内存池

// 启用动态扩容（默认启用）
aclshmem_enable_dynamic_expansion(true);

// 初始化SHMEM
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);

// 正常使用内存分配接口
void* ptr1 = aclshmem_malloc(10 * 1024 * 1024); // 10MB
void* ptr2 = aclshmem_malloc(100 * 1024 * 1024); // 100MB（触发扩容）

// 释放内存
aclshmem_free(ptr1);
aclshmem_free(ptr2);
```

### 2. Python集成使用

```python
import ctypes
import torch
import torch_npu

# 加载SHMEM库
shmem_lib = ctypes.CDLL('./build/libshmem_dynamic.so')

# 启用动态扩容
shmem_lib.aclshmem_enable_dynamic_expansion(ctypes.c_bool(True))

# 初始化SHMEM
init_attr = shmem_lib.aclshmemx_init_attr_t()
init_attr.version = 1
init_attr.my_rank = 0
init_attr.n_ranks = 1
init_attr.local_mem_size = 64 * 1024 * 1024  # 64MB

shmem_lib.aclshmemx_init_attr(0, ctypes.byref(init_attr))

# 使用PyTorch创建张量（会自动触发扩容）
large_tensor = torch.randn(10000, 10000, device='npu')  # 约381MB

# 获取内存统计
total_cap = ctypes.c_uint64()
used_mem = ctypes.c_uint64()
avail_mem = ctypes.c_uint64()

shmem_lib.aclshmem_get_memory_stats(
    ctypes.byref(total_cap),
    ctypes.byref(used_mem), 
    ctypes.byref(avail_mem)
)

print(f"Total: {total_cap.value/1024/1024:.2f}MB")
print(f"Used: {used_mem.value/1024/1024:.2f}MB")
print(f"Available: {avail_mem.value/1024/1024:.2f}MB")
```

### 3. 性能监控

```cpp
// 获取详细的内存使用情况
uint64_t total_capacity, used_memory, available_memory;
aclshmem_get_memory_stats(&total_capacity, &used_memory, &available_memory);

printf("Memory Usage:\n");
printf("  Total Capacity: %.2f MB\n", total_capacity / (1024.0 * 1024));
printf("  Used Memory: %.2f MB\n", used_memory / (1024.0 * 1024));
printf("  Available Memory: %.2f MB\n", available_memory / (1024.0 * 1024));

// 检查是否发生了扩容
static uint64_t last_capacity = 0;
if (last_capacity > 0 && total_capacity > last_capacity) {
    printf("Memory pool expanded from %.2f MB to %.2f MB\n", 
           last_capacity / (1024.0 * 1024),
           total_capacity / (1024.0 * 1024));
}
last_capacity = total_capacity;
```

## 🧪 测试验证

### 运行测试脚本

```bash
# 构建测试程序
chmod +x build_dynamic_test.sh
./build_dynamic_test.sh

# 运行Python测试
python test_dynamic_expansion.py
```

### 测试内容

1. **顺序分配测试**: 验证多个连续的内存分配
2. **大块分配测试**: 测试单次大内存分配触发扩容
3. **PyTorch集成测试**: 验证与深度学习框架的兼容性
4. **内存统计测试**: 验证内存使用情况的准确性

## ⚙️ 配置选项

### 环境变量

```bash
# 控制日志级别
export SHMEM_LOG_LEVEL=DEBUG

# 控制是否启用动态扩容（运行时）
export SHMEM_ENABLE_DYNAMIC_EXPANSION=1
```

### 编译时配置

在 `shmem_dynamic_mm.h` 中可以调整：

```cpp
// 扩容策略参数
constexpr uint64_t MIN_EXPANSION_SIZE = 256 * 1024 * 1024;  // 最小扩容大小
constexpr double EXPANSION_FACTOR = 1.5;                    // 扩容因子
constexpr uint64_t MAX_BLOCK_SIZE = 4ULL * 1024 * 1024 * 1024;  // 最大单块大小
```

## 📈 性能特点

### 优势

- **透明扩容**: 对用户完全透明，无需修改现有代码
- **智能策略**: 根据实际需求合理扩容，避免过度分配
- **高效管理**: 统一的内存管理减少碎片化
- **良好兼容**: 与现有SHMEM API完全兼容

### 性能考虑

- 首次扩容会有一定的延迟（CANN内存分配时间）
- 扩容后的内存访问性能与初始内存池相同
- 内存释放时会自动回收未使用的外部内存块

## 🐛 故障排除

### 常见问题

1. **扩容失败**
   ```bash
   # 检查CANN环境
   npu-smi info
   
   # 确认有足够的设备内存
   echo "Device memory usage:" && npu-smi info -t memory
   ```

2. **内存泄漏**
   ```cpp
   // 启用详细日志
   export SHMEM_LOG_LEVEL=DEBUG
   
   // 使用内存统计功能监控
   uint64_t used_before, used_after;
   aclshmem_get_memory_stats(nullptr, &used_before, nullptr);
   // ... 执行操作 ...
   aclshmem_get_memory_stats(nullptr, &used_after, nullptr);
   ```

3. **性能问题**
   ```cpp
   // 检查是否频繁扩容
   static int expansion_count = 0;
   static uint64_t last_capacity = 0;
   
   uint64_t current_capacity;
   aclshmem_get_memory_stats(&current_capacity, nullptr, nullptr);
   
   if (current_capacity > last_capacity) {
       expansion_count++;
       printf("Expansion #%d occurred\n", expansion_count);
   }
   last_capacity = current_capacity;
   ```

## 📚 相关文档

- [SHMEM官方文档](https://shmem-doc.pages.dev/)
- [CANN内存管理指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/infacldevg/aclcppdevg/aclcppdevg_03_0001.html)
- [PyTorch NPU集成指南](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0004.html)

---
*更多技术细节请参考源代码实现*