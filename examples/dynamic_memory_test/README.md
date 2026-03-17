# SHMEM 动态内存扩容功能使用指南

## 📋 概述

本功能实现了 SHMEM 内存池的动态扩容能力。当单次分配请求超出初始内存池剩余容量时，
自动调用 CANN `aclrtMalloc` 接口申请新内存块，并将其纳入统一管理结构，对调用方完全透明。

测试文件位于 `examples/dynamic_memory_test/`：

| 文件 | 说明 |
|------|------|
| `test_dynamic_cpp.cpp` | C++ 完整测试（顺序分配 / 大块分配 / 边界条件 / 数据校验） |
| `test_dynamic_expansion.py` | Python 等价测试（通过 ctypes 调用 libshmem.so） |
| `build_dynamic_test.sh` | 编译 C++ 测试二进制的脚本（libshmem.so 由主包构建） |

## 🚀 主要特性

| 特性 | 说明 |
|------|------|
| 自动扩容 | 内存不足时透明调用 CANN 接口分配新块，无需修改调用方代码 |
| 统一管理 | 初始内存池与动态块由同一管理器跟踪，`aclshmem_free` 自动识别归属 |
| 智能策略 | 扩容量取"1.5 × 请求大小"与最小扩容量（256 MB）的较大值，上限 4 GB/块 |
| 可配置 | `aclshmem_enable_dynamic_expansion()` 运行时开关，默认启用 |
| 完全兼容 | 不改变现有 `aclshmem_malloc` / `aclshmem_free` 等接口签名 |

## 🛠️ 核心接口

### 动态扩容控制

```cpp
// 启用或禁用动态扩容（必须在 aclshmemx_init_attr 之前调用）
void aclshmem_enable_dynamic_expansion(bool enable);

// 查询当前是否已启用
bool aclshmem_is_dynamic_expansion_enabled();

// 获取内存池整体使用统计（字节）
void aclshmem_get_memory_stats(uint64_t* total_capacity,
                               uint64_t* used_memory,
                               uint64_t* available_memory);

// 主动释放所有空闲的动态块（可选，finalize 时也会自动执行）
void aclshmem_cleanup_unused_memory();
```

### 初始化接口（UniqueID 方式）

测试代码均采用 UniqueID bootstrap，单进程场景下 `my_pe=0, n_pes=1`：

```cpp
aclshmemx_uniqueid_t    uid;
aclshmemx_init_attr_t   attributes;

aclshmemx_get_uniqueid(&uid);
aclshmemx_set_attr_uniqueid_args(
    /*my_pe*/        0,
    /*n_pes*/        1,
    /*local_mem_size*/ 32 * 1024 * 1024,   // 32 MB 初始池
    &uid, &attributes
);
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attributes);
```

> ⚠️ `aclshmem_enable_dynamic_expansion(true)` **必须在** `aclshmemx_init_attr` 之前调用，
> 两个测试文件均遵守此顺序。

### 内存分配接口（兼容现有）

```cpp
void* aclshmem_malloc(size_t size);          // 不足时自动扩容
void* aclshmem_calloc(size_t nmemb, size_t size);
void* aclshmem_align(size_t alignment, size_t size);
void  aclshmem_free(void* ptr);

void* aclshmemx_malloc(size_t size, aclshmem_mem_type_t mem_type);
void* aclshmemx_calloc(size_t nmemb, size_t size, aclshmem_mem_type_t mem_type);
void* aclshmemx_align(size_t alignment, size_t size, aclshmem_mem_type_t mem_type);
void  aclshmemx_free(void* ptr, aclshmem_mem_type_t mem_type);
```

## 📊 扩容策略

```
requested_size → expansion_size = max(requested_size × 1.5, 256 MB)
expansion_size = min(expansion_size, 4 GB)
expansion_size = align_up(expansion_size, 256 KB)
```

新块通过 `aclrtMalloc` 在 NPU HBM 上分配，与初始内存池处于同一设备地址空间。

## 🔧 使用方法

### C++ 测试

```cpp
#include "acl/acl.h"
#include "shmem.h"

// 1. 初始化 ACL 运行时
aclInit(nullptr);
aclrtSetDevice(0);

// 2. 启用动态扩容（在 init 之前）
aclshmem_enable_dynamic_expansion(true);

// 3. UniqueID 方式初始化 SHMEM（32 MB 初始池）
aclshmemx_uniqueid_t  uid;
aclshmemx_init_attr_t attr;
aclshmemx_get_uniqueid(&uid);
aclshmemx_set_attr_uniqueid_args(0, 1, 32 * 1024 * 1024, &uid, &attr);
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr);

// 4. 正常分配（超出 32 MB 时自动扩容）
void* p1 = aclshmem_malloc( 10 * 1024 * 1024);   // 10 MB —— 在初始池内
void* p2 = aclshmem_malloc(100 * 1024 * 1024);   // 100 MB —— 触发扩容

// 5. 数据读写：NPU HBM 地址，CPU 不可直接访问，必须通过 aclrtMemcpy
aclrtMemcpy(p2, bytes, host_src, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(host_dst, bytes, p2, bytes, ACL_MEMCPY_DEVICE_TO_HOST);

aclshmem_free(p1);
aclshmem_free(p2);

// 6. 清理
aclshmem_finalize();
aclrtResetDevice(0);
aclFinalize();
```

### Python 测试

Python 测试通过 ctypes 直接调用 libshmem.so，**无需** torch / torch_npu 依赖。

```python
import ctypes, os

# 加载库（优先从 build/lib/ 查找）
lib = ctypes.CDLL("build/lib/libshmem.so",
                  mode=ctypes.RTLD_LAZY | ctypes.RTLD_GLOBAL)

# 绑定 C++ mangled 符号（通过 nm -D libshmem.so 确认）
_enable_fn = lib["_Z33aclshmem_enable_dynamic_expansionb"]
_enable_fn.restype  = None
_enable_fn.argtypes = [ctypes.c_bool]

# 启用动态扩容
_enable_fn(True)

# ... UniqueID 初始化（见 test_dynamic_expansion.py）...

# 分配 100 MB（超出 32 MB 初始池，触发扩容）
lib.aclshmem_malloc.restype  = ctypes.c_void_p   # 必须声明为 c_void_p，避免截断
lib.aclshmem_malloc.argtypes = [ctypes.c_size_t]
ptr = lib.aclshmem_malloc(100 * 1024 * 1024)

lib.aclshmem_free(ptr)
```

> ⚠️ `aclshmem_malloc` 的 `restype` **必须**声明为 `c_void_p`。
> 若使用默认的 `c_int`（32 位），64 位设备指针会被截断，导致后续 `free` 时 segfault。

### 内存统计监控

```cpp
// C++
uint64_t total, used, avail;
aclshmem_get_memory_stats(&total, &used, &avail);
printf("Total=%.1fMB  Used=%.1fMB  Avail=%.1fMB  Util=%.1f%%\n",
       total/1e6, used/1e6, avail/1e6, used*100.0/total);

// 检测是否发生了扩容
static uint64_t prev_total = 0;
if (prev_total > 0 && total > prev_total)
    printf("Pool expanded: %.1fMB → %.1fMB\n", prev_total/1e6, total/1e6);
prev_total = total;
```

```python
# Python
total  = ctypes.c_uint64()
used   = ctypes.c_uint64()
avail  = ctypes.c_uint64()
lib._get_stats(ctypes.byref(total), ctypes.byref(used), ctypes.byref(avail))
print(f"Total={total.value/1e6:.1f}MB  Used={used.value/1e6:.1f}MB  Avail={avail.value/1e6:.1f}MB")
```

## 🧪 构建与运行

libshmem.so 由主包 CMake 构建，脚本只负责编译 C++ 测试二进制。

```bash
cd examples/dynamic_memory_test

# 编译 C++ 测试二进制（输出至当前目录）
chmod +x build_dynamic_test.sh
./build_dynamic_test.sh

# 将主包 build/lib 加入库搜索路径
export LD_LIBRARY_PATH=../../build/lib:$LD_LIBRARY_PATH

# 运行 C++ 测试
./test_dynamic_cpp

# 运行 Python 测试（脚本内部自动设置 SHMEM_UID_SESSION_ID=127.0.0.1:12345）
python3 test_dynamic_expansion.py

# 或手动指定会话地址
SHMEM_UID_SESSION_ID=127.0.0.1:12345 python3 test_dynamic_expansion.py
```

### 测试用例说明

| 测试 | C++ 方法 | Python 方法 | 验证内容 |
|------|----------|-------------|----------|
| 顺序分配 | `testSequentialAllocation` | `test_sequential_allocation` | 依次分配 5/10/15/25 MB，内存池在接近满载时自动扩容 |
| 大块分配 | `testLargeAllocation` | `test_large_allocation` | 单次分配 100 MB（超出 32 MB 初始池），并通过 `aclrtMemcpy` 验证数据正确性 |
| 边界条件 | `testBoundaryConditions` | `test_boundary_conditions` | 零字节分配返回 NULL；10 GB 超大分配失败返回 NULL |

## ⚙️ 配置参考

### 编译时常量（`shmem_dynamic_mm.h`）

```cpp
constexpr uint64_t MIN_EXPANSION_SIZE = 256 * 1024 * 1024;       // 最小扩容量
constexpr double   EXPANSION_FACTOR   = 1.5;                     // 扩容因子
constexpr uint64_t MAX_BLOCK_SIZE     = 4ULL * 1024 * 1024 * 1024; // 单块上限
```

### Python 测试关键常量（`test_dynamic_expansion.py`）

```python
_ACLSHMEMX_INIT_WITH_UNIQUEID = 1 << 3      # bootstrap 标志位
_ACLSHMEM_DATA_OP_MTE         = 0x01        # option_attr.data_op_engine_type 必须 > 0
_DEFAULT_TIMEOUT              = 120         # shm_init/create/control timeout（秒）
```

option_attr 字段必须手动填充（ctypes 不自动应用 C++ 默认值）：

```python
attr.option_attr.version                   = (1 << 16) + ctypes.sizeof(_OptionalAttr)
attr.option_attr.data_op_engine_type       = 0x01   # ACLSHMEM_DATA_OP_MTE，不能为 0
attr.option_attr.shm_init_timeout          = 120
attr.option_attr.shm_create_timeout        = 120
attr.option_attr.control_operation_timeout = 120
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `SHMEM_UID_SESSION_ID` | bootstrap 层单节点通信地址，Python 脚本内自动设置 | `127.0.0.1:12345` |
| `LD_LIBRARY_PATH` | 需包含主包的 `build/lib` 路径 | — |
| `ASCEND_HOME` | CANN Toolkit 安装路径 | `/usr/local/Ascend/ascend-toolkit/latest` |
| `SHMEM_LOG_LEVEL` | 日志级别（DEBUG / INFO / WARNING） | INFO |

## 🐛 故障排除

**扩容失败（`aclshmem_malloc` 返回 NULL）**

```bash
npu-smi info -t memory
```

**Python 测试 segfault（free 时崩溃）**

ctypes 绑定 `aclshmem_malloc` 时 `restype` 未声明为 `c_void_p`，导致 64 位指针被截断。
确认绑定代码如下：

```python
lib.aclshmem_malloc.restype  = ctypes.c_void_p   # ← 必须是 c_void_p
lib.aclshmem_malloc.argtypes = [ctypes.c_size_t]
```

**`check_attr` 报错（初始化失败）**

`option_attr.data_op_engine_type` 为 0 时内部校验会拒绝。确认已设置：

```python
attr.option_attr.data_op_engine_type = 0x01  # ACLSHMEM_DATA_OP_MTE
```

## 📚 相关文档

- [CANN 内存管理指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/infacldevg/aclcppdevg/aclcppdevg_03_0001.html)
- [SHMEM 官方文档](https://shmem-doc.pages.dev/)
- [PyTorch NPU 集成指南](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0004.html)