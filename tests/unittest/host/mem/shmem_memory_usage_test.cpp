/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file shmem_memory_usage_test.cpp
 * @brief SHMEM 框架内存占用测试
 *
 * 本测试度量 shmem 框架在各生命周期阶段的内存开销，涵盖：
 *  - 进程级内存（RSS/VmSize，读取 /proc/self/status）
 *  - Heap 级内存统计（通过 aclshmem_get_memory_stats()）
 *  - 框架固定开销（EXTRA_SIZE：NPU 同步池等元数据）
 *  - 不同分配模式下的内存表现
 *
 * 测试用例列表：
 *  1. MemoryUsageTest/baseline_and_init_overhead
 *     - 测量 init 前后进程内存与框架堆内存的变化量
 *  2. MemoryUsageTest/single_alloc_and_free
 *     - 验证 malloc / free 对堆可用内存的影响符合预期
 *  3. MemoryUsageTest/multi_size_alloc_overhead
 *     - 在多种分配大小下记录堆利用率，输出利用率表格
 *  4. MemoryUsageTest/align_overhead
 *     - 比较 aclshmem_malloc 与 aclshmem_align 在不同对齐值下的实际占用
 *  5. MemoryUsageTest/fragmentation_after_interleaved_free
 *     - 交叉释放后重新分配，评估堆碎片对大块分配的影响
 *  6. MemoryUsageTest/framework_fixed_overhead_report
 *     - 汇报框架常量开销：EXTRA_SIZE、SYNC_POOL_SIZE、CORE_SYNC_POOL_SIZE 等
 *  7. MemoryUsageTest/calloc_zeroing_overhead
 *     - calloc 与 malloc 之间进程 RSS 差异（体现 zero-page 触写开销）
 *  8. MemoryUsageTest/stress_alloc_memory_trend
 *     - 分多轮递增分配，每轮记录堆已用/可用量，输出趋势
 *
 * 使用方式（通过项目自带的 run.sh）：
 *   ./scripts/run.sh -test_filter MemoryUsage
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <acl/acl.h>

#include "shmem.h"
#include "shmemi_host_common.h"
#include "mem/shmemi_mm.h"                 // aclshmem_get_memory_stats()
#include "host_device/shmem_common_types.h"
#include "unittest_main_test.h"

// ---------------------------------------------------------------------------
// 辅助：读取 /proc/self/status 中的若干字段（KB）
// ---------------------------------------------------------------------------
struct ProcMemInfo {
    long vm_size_kb  = 0;   ///< VmSize:  进程虚拟地址空间大小
    long vm_rss_kb   = 0;   ///< VmRSS:   驻留物理内存
    long vm_peak_kb  = 0;   ///< VmPeak:  虚拟内存历史峰值
    long vm_hwm_kb   = 0;   ///< VmHWM:   RSS 历史峰值
};

static ProcMemInfo read_proc_mem()
{
    ProcMemInfo info{};
    std::ifstream ifs("/proc/self/status");
    if (!ifs.is_open()) {
        return info;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        auto parse = [&](const char *key, long &dst) {
            if (line.compare(0, strlen(key), key) == 0) {
                std::istringstream ss(line.substr(strlen(key)));
                ss >> dst;
            }
        };
        parse("VmSize:", info.vm_size_kb);
        parse("VmRSS:",  info.vm_rss_kb);
        parse("VmPeak:", info.vm_peak_kb);
        parse("VmHWM:",  info.vm_hwm_kb);
    }
    return info;
}

// ---------------------------------------------------------------------------
// 辅助：打印带标题的分隔线
// ---------------------------------------------------------------------------
static void print_separator(const char *title = nullptr)
{
    std::string line(72, '-');
    if (title && *title) {
        std::cout << "\n[MEM] " << title << "\n" << line << "\n";
    } else {
        std::cout << line << "\n";
    }
}

// ---------------------------------------------------------------------------
// 辅助：格式化字节数为可读字符串（自动选 B/KB/MB/GB）
// ---------------------------------------------------------------------------
static std::string fmt_bytes(uint64_t bytes)
{
    constexpr uint64_t KB = 1024ULL;
    constexpr uint64_t MB = 1024ULL * KB;
    constexpr uint64_t GB = 1024ULL * MB;
    char buf[64];
    if (bytes >= GB) {
        std::snprintf(buf, sizeof(buf), "%.3f GB", (double)bytes / GB);
    } else if (bytes >= MB) {
        std::snprintf(buf, sizeof(buf), "%.3f MB", (double)bytes / MB);
    } else if (bytes >= KB) {
        std::snprintf(buf, sizeof(buf), "%.3f KB", (double)bytes / KB);
    } else {
        std::snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)bytes);
    }
    return buf;
}

// ---------------------------------------------------------------------------
// 辅助：打印进程内存快照
// ---------------------------------------------------------------------------
static void print_proc_mem(const char *label, const ProcMemInfo &m)
{
    std::cout << std::left << std::setw(30) << label
              << "  VmRSS=" << std::setw(10) << (std::to_string(m.vm_rss_kb) + " KB")
              << "  VmSize=" << std::setw(12) << (std::to_string(m.vm_size_kb) + " KB")
              << "  VmHWM=" << std::setw(10) << (std::to_string(m.vm_hwm_kb) + " KB")
              << "\n";
}

// ---------------------------------------------------------------------------
// 辅助：打印框架堆统计
// ---------------------------------------------------------------------------
static void print_heap_stats(const char *label)
{
    uint64_t total = 0, used = 0, avail = 0;
    aclshmem_get_memory_stats(&total, &used, &avail);
    std::cout << std::left << std::setw(30) << label
              << "  total=" << std::setw(14) << fmt_bytes(total)
              << "  used=" << std::setw(14) << fmt_bytes(used)
              << "  avail=" << std::setw(14) << fmt_bytes(avail);
    if (total > 0) {
        std::cout << "  util=" << std::fixed << std::setprecision(2)
                  << (100.0 * used / total) << "%";
    }
    std::cout << "\n";
}

// ===========================================================================
// 测试夹具
// ===========================================================================
class MemoryUsageTest : public ::testing::Test {
protected:
    static constexpr uint64_t kLocalMemSize = 64UL * 1024 * 1024;  // 64 MB
};

// ---------------------------------------------------------------------------
// 测试 1：baseline vs init 阶段开销
// ---------------------------------------------------------------------------
TEST_F(MemoryUsageTest, baseline_and_init_overhead)
{
    test_single_task(
        [](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            print_separator("Test: baseline_and_init_overhead");

            // --- baseline（init 之前）---
            ProcMemInfo before = read_proc_mem();
            print_proc_mem("  [baseline] before init", before);

            // --- init ---
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);

            // --- post-init ---
            ProcMemInfo after_init = read_proc_mem();
            print_proc_mem("  [post-init]", after_init);
            print_heap_stats("  [heap@post-init]");

            long rss_delta = after_init.vm_rss_kb - before.vm_rss_kb;
            long vsz_delta = after_init.vm_size_kb - before.vm_size_kb;
            std::cout << "  => RSS delta after init:   " << rss_delta << " KB\n";
            std::cout << "  => VmSize delta after init: " << vsz_delta << " KB\n";

            // --- finalize ---
            test_finalize(stream, device_id);

            ProcMemInfo after_fin = read_proc_mem();
            print_proc_mem("  [post-finalize]", after_fin);
            long rss_fin_delta = after_fin.vm_rss_kb - before.vm_rss_kb;
            std::cout << "  => Residual RSS after finalize: " << rss_fin_delta << " KB\n";

            print_separator();
        },
        kLocalMemSize);
}

// ---------------------------------------------------------------------------
// 测试 2：单次 malloc / free 对堆可用内存的影响
// ---------------------------------------------------------------------------
TEST_F(MemoryUsageTest, single_alloc_and_free)
{
    test_single_task(
        [](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            print_separator("Test: single_alloc_and_free");

            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);

            static const size_t kSizes[] = {
                4 * 1024,          //  4 KB
                64 * 1024,         // 64 KB
                1 * 1024 * 1024,   //  1 MB
                16 * 1024 * 1024,  // 16 MB
            };

            for (size_t alloc_size : kSizes) {
                uint64_t total0 = 0, used0 = 0, avail0 = 0;
                aclshmem_get_memory_stats(&total0, &used0, &avail0);

                void *ptr = aclshmem_malloc(alloc_size);

                uint64_t total1 = 0, used1 = 0, avail1 = 0;
                aclshmem_get_memory_stats(&total1, &used1, &avail1);

                uint64_t used_delta  = (used1 > used0)  ? (used1 - used0)  : 0;
                uint64_t avail_delta = (avail0 > avail1) ? (avail0 - avail1) : 0;

                std::cout << "  alloc " << std::setw(10) << fmt_bytes(alloc_size)
                          << "  => used+" << std::setw(12) << fmt_bytes(used_delta)
                          << "  avail-" << std::setw(12) << fmt_bytes(avail_delta)
                          << "  ptr=" << ptr << "\n";

                if (ptr) {
                    aclshmem_free(ptr);
                    uint64_t total2 = 0, used2 = 0, avail2 = 0;
                    aclshmem_get_memory_stats(&total2, &used2, &avail2);
                    uint64_t recovered = (avail2 > avail1) ? (avail2 - avail1) : 0;
                    std::cout << "  free  " << std::setw(10) << fmt_bytes(alloc_size)
                              << "  => avail+" << fmt_bytes(recovered) << "\n";
                }
            }

            test_finalize(stream, device_id);
            print_separator();
        },
        kLocalMemSize);
}

// ---------------------------------------------------------------------------
// 测试 3：多种大小下堆利用率记录
// ---------------------------------------------------------------------------
TEST_F(MemoryUsageTest, multi_size_alloc_overhead)
{
    test_single_task(
        [](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            print_separator("Test: multi_size_alloc_overhead");

            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);

            // 列表：申请比例（相对于 local_mem_size）
            const double ratios[] = {0.1, 0.2, 0.3, 0.5, 0.8, 1.0};
            const int num_ratios  = static_cast<int>(sizeof(ratios) / sizeof(ratios[0]));

            std::cout << "  " << std::setw(10) << "Requested"
                      << std::setw(12) << "Total"
                      << std::setw(12) << "Used"
                      << std::setw(12) << "Avail"
                      << std::setw(10) << "Util%"
                      << "  Result\n";
            std::cout << "  " << std::string(60, '-') << "\n";

            for (int i = 0; i < num_ratios; ++i) {
                size_t req_size = static_cast<size_t>(local_mem_size * ratios[i]);
                void *ptr = aclshmem_malloc(req_size);

                uint64_t total = 0, used = 0, avail = 0;
                aclshmem_get_memory_stats(&total, &used, &avail);

                double util = (total > 0) ? (100.0 * used / total) : 0.0;
                std::cout << "  " << std::setw(10) << fmt_bytes(req_size)
                          << std::setw(12) << fmt_bytes(total)
                          << std::setw(12) << fmt_bytes(used)
                          << std::setw(12) << fmt_bytes(avail)
                          << std::setw(9) << std::fixed << std::setprecision(1) << util << "%"
                          << "  " << (ptr ? "OK" : "FAIL (OOM)") << "\n";

                if (ptr) {
                    aclshmem_free(ptr);
                }
            }

            test_finalize(stream, device_id);
            print_separator();
        },
        kLocalMemSize);
}

// ---------------------------------------------------------------------------
// 测试 4：对齐分配的额外开销
// ---------------------------------------------------------------------------
TEST_F(MemoryUsageTest, align_overhead)
{
    test_single_task(
        [](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            print_separator("Test: align_overhead");

            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);

            constexpr size_t kAllocSize = 1 * 1024 * 1024;  // 1 MB

            // 对比 malloc vs align 在不同对齐粒度下的实际堆消耗
            const size_t alignments[] = {16, 64, 512, 4096, 65536, 1024 * 1024};
            const int    num_align    = static_cast<int>(sizeof(alignments) / sizeof(alignments[0]));

            std::cout << "  alloc_size = " << fmt_bytes(kAllocSize) << "\n";
            std::cout << "  " << std::setw(14) << "Alignment"
                      << std::setw(14) << "used_before"
                      << std::setw(14) << "used_after"
                      << std::setw(14) << "delta_used"
                      << "  Result\n";
            std::cout << "  " << std::string(60, '-') << "\n";

            // --- baseline: plain malloc ---
            {
                uint64_t t0 = 0, u0 = 0, a0 = 0;
                aclshmem_get_memory_stats(&t0, &u0, &a0);
                void *p = aclshmem_malloc(kAllocSize);
                uint64_t t1 = 0, u1 = 0, a1 = 0;
                aclshmem_get_memory_stats(&t1, &u1, &a1);
                std::cout << "  " << std::setw(14) << "malloc(no align)"
                          << std::setw(14) << fmt_bytes(u0)
                          << std::setw(14) << fmt_bytes(u1)
                          << std::setw(14) << fmt_bytes(u1 > u0 ? u1 - u0 : 0)
                          << "  " << (p ? "OK" : "FAIL") << "\n";
                if (p) aclshmem_free(p);
            }

            // --- align variants ---
            for (int i = 0; i < num_align; ++i) {
                size_t al = alignments[i];
                uint64_t t0 = 0, u0 = 0, a0 = 0;
                aclshmem_get_memory_stats(&t0, &u0, &a0);
                void *p = aclshmem_align(al, kAllocSize);
                uint64_t t1 = 0, u1 = 0, a1 = 0;
                aclshmem_get_memory_stats(&t1, &u1, &a1);
                std::cout << "  " << std::setw(14) << fmt_bytes(al)
                          << std::setw(14) << fmt_bytes(u0)
                          << std::setw(14) << fmt_bytes(u1)
                          << std::setw(14) << fmt_bytes(u1 > u0 ? u1 - u0 : 0)
                          << "  " << (p ? "OK" : "FAIL") << "\n";
                if (p) aclshmem_free(p);
            }

            test_finalize(stream, device_id);
            print_separator();
        },
        kLocalMemSize);
}

// ---------------------------------------------------------------------------
// 测试 5：交叉释放后碎片对大块分配的影响
// ---------------------------------------------------------------------------
TEST_F(MemoryUsageTest, fragmentation_after_interleaved_free)
{
    test_single_task(
        [](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            print_separator("Test: fragmentation_after_interleaved_free");

            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);

            // 分配 4 块等大内存
            constexpr int kBlocks = 4;
            const size_t  kBlock  = local_mem_size / kBlocks;
            void *ptrs[kBlocks]   = {};

            for (int i = 0; i < kBlocks; ++i) {
                ptrs[i] = aclshmem_malloc(kBlock);
                ASSERT_NE(nullptr, ptrs[i])
                    << "Failed to allocate block " << i;
            }
            print_heap_stats("  [all blocks allocated]");

            // 释放偶数块（模拟碎片）
            for (int i = 0; i < kBlocks; i += 2) {
                aclshmem_free(ptrs[i]);
                ptrs[i] = nullptr;
            }
            print_heap_stats("  [even blocks freed]");

            // 尝试分配 kBlock*2（需要相邻空闲块合并）
            void *big = aclshmem_malloc(kBlock * 2);
            std::cout << "  alloc(" << fmt_bytes(kBlock * 2) << ")="
                      << (big ? "OK (no fragmentation / merged)" : "FAIL (fragmented)") << "\n";
            print_heap_stats("  [after big alloc attempt]");

            // 清理
            if (big) aclshmem_free(big);
            for (int i = 1; i < kBlocks; i += 2) {
                if (ptrs[i]) aclshmem_free(ptrs[i]);
            }
            print_heap_stats("  [all freed]");

            test_finalize(stream, device_id);
            print_separator();
        },
        kLocalMemSize);
}

// ---------------------------------------------------------------------------
// 测试 6：框架固定开销汇报（常量计算，不依赖硬件）
// ---------------------------------------------------------------------------
TEST_F(MemoryUsageTest, framework_fixed_overhead_report)
{
    // 本测试仅在主进程输出，无需 fork
    print_separator("Test: framework_fixed_overhead_report");

    // 从 shmem_common_types.h 中提取的常量
    constexpr uint64_t kPageSize            = ACLSHMEM_PAGE_SIZE;
    constexpr uint64_t kSyncBitSize         = ACLSHMEMI_SYNCBIT_SIZE;
    constexpr uint64_t kMaxTeams            = ACLSHMEM_MAX_TEAMS;
    constexpr uint64_t kMaxPes              = ACLSHMEM_MAX_PES;
    constexpr uint64_t kMaxAivPerNpu        = ACLSHMEM_MAX_AIV_PER_NPU;
    constexpr uint64_t kLogMaxAiv           = ACLSHMEM_LOG_MAX_AIV_PER_NPU;
    constexpr uint64_t kSyncLogMaxPes       = SYNC_LOG_MAX_PES;
    constexpr uint64_t kDissemKval          = ACLSHMEM_BARRIER_TG_DISSEM_KVAL;
    constexpr uint64_t kSyncArraySize       = SYNC_ARRAY_SIZE;
    constexpr uint64_t kSyncPoolSize        = SYNC_POOL_SIZE;
    constexpr uint64_t kSyncCountersSize    = SYNC_COUNTERS_SIZE;
    constexpr uint64_t kCoreSyncPoolSize    = ACLSHMEM_CORE_SYNC_POOL_SIZE;
    constexpr uint64_t kExtraSize           = ACLSHMEM_EXTRA_SIZE;
    constexpr uint64_t kHeapAlignment       = ACLSHMEM_HEAP_ALIGNMENT_SIZE;

    // team_t 结构体大小（每个 team 实例的内存占用）
    constexpr uint64_t kTeamStructSize      = sizeof(aclshmemx_team_t);
    constexpr uint64_t kMaxTeamPoolBytes    = kTeamStructSize * kMaxTeams;

    std::cout << "\n  === SHMEM Framework Constants ===\n\n";

    auto row = [](const char *name, uint64_t val, const char *note = "") {
        std::cout << "  " << std::left << std::setw(38) << name
                  << std::setw(18) << fmt_bytes(val)
                  << "  " << note << "\n";
    };

    std::cout << "  -- Architecture --\n";
    row("ACLSHMEM_PAGE_SIZE",            kPageSize,         "device memory page");
    row("ACLSHMEM_HEAP_ALIGNMENT_SIZE",  kHeapAlignment,    "heap base alignment (1 GB)");
    row("SCALAR_DATA_CACHELINE_SIZE",    SCALAR_DATA_CACHELINE_SIZE, "64 bytes");
    row("ACLSHMEM_MAX_PES",              kMaxPes,           "max 16384 PEs");
    row("ACLSHMEM_MAX_TEAMS",            kMaxTeams,         "max 2048 teams");

    std::cout << "\n  -- NPU-level Sync Pool --\n";
    row("ACLSHMEMI_SYNCBIT_SIZE",        kSyncBitSize,      "per sync-bit slot");
    row("SYNC_LOG_MAX_PES",              kSyncLogMaxPes,    "bits");
    row("ACLSHMEM_BARRIER_TG_DISSEM_KVAL", kDissemKval,    "k=8");
    row("SYNC_ARRAY_SIZE (per team)",    kSyncArraySize,    "= syncbit * log_pes * k");
    row("SYNC_POOL_SIZE (all teams)",    kSyncPoolSize,     "= sync_array * max_teams");
    row("SYNC_COUNTERS_SIZE",            kSyncCountersSize, "= syncbit * max_teams");

    std::cout << "\n  -- Core-level Sync Pool --\n";
    row("ACLSHMEM_MAX_AIV_PER_NPU",      kMaxAivPerNpu,     "48 AI cores");
    row("ACLSHMEM_LOG_MAX_AIV_PER_NPU",  kLogMaxAiv,        "6 bits");
    row("ACLSHMEM_CORE_SYNC_POOL_SIZE",  kCoreSyncPoolSize, "per NPU");

    std::cout << "\n  -- Total Fixed Overhead per PE --\n";
    row("ACLSHMEM_EXTRA_SIZE",           kExtraSize,        "aligned to PAGE_SIZE, incl. sync pools");

    std::cout << "\n  -- Team Pool (host-side, per PE) --\n";
    row("sizeof(aclshmemx_team_t)",      kTeamStructSize,   "single team descriptor");
    row("MAX_TEAMS * sizeof(team_t)",    kMaxTeamPoolBytes, "worst-case team pool");

    std::cout << "\n  -- Memory per PE for user-space heap --\n";
    std::cout << "  ACLSHMEM_EXTRA_SIZE is appended to user-requested local_mem_size\n";
    std::cout << "  Actual device allocation = local_mem_size + ACLSHMEM_EXTRA_SIZE\n";
    std::cout << "  => For local_mem_size=64 MB:  total = "
              << fmt_bytes(64ULL * 1024 * 1024 + kExtraSize) << "\n";
    std::cout << "  => For local_mem_size=1  GB:  total = "
              << fmt_bytes(1ULL * 1024 * 1024 * 1024 + kExtraSize) << "\n";
    std::cout << "  => For local_mem_size=10 GB:  total = "
              << fmt_bytes(10ULL * 1024 * 1024 * 1024 + kExtraSize) << "\n";

    print_separator();
}

// ---------------------------------------------------------------------------
// 测试 7：calloc 与 malloc 的进程 RSS 差异
// ---------------------------------------------------------------------------
TEST_F(MemoryUsageTest, calloc_zeroing_overhead)
{
    test_single_task(
        [](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            print_separator("Test: calloc_zeroing_overhead");

            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);

            constexpr size_t kAllocSize = 16 * 1024 * 1024;  // 16 MB

            // malloc
            {
                ProcMemInfo m0 = read_proc_mem();
                uint64_t t0 = 0, u0 = 0, a0 = 0;
                aclshmem_get_memory_stats(&t0, &u0, &a0);

                void *p = aclshmem_malloc(kAllocSize);

                ProcMemInfo m1 = read_proc_mem();
                uint64_t t1 = 0, u1 = 0, a1 = 0;
                aclshmem_get_memory_stats(&t1, &u1, &a1);

                std::cout << "  malloc(" << fmt_bytes(kAllocSize) << "):\n"
                          << "    heap used delta = " << fmt_bytes(u1 > u0 ? u1 - u0 : 0) << "\n"
                          << "    proc RSS  delta = " << (m1.vm_rss_kb - m0.vm_rss_kb) << " KB\n";
                if (p) aclshmem_free(p);
            }

            // calloc
            {
                ProcMemInfo m0 = read_proc_mem();
                uint64_t t0 = 0, u0 = 0, a0 = 0;
                aclshmem_get_memory_stats(&t0, &u0, &a0);

                void *p = aclshmem_calloc(1, kAllocSize);

                ProcMemInfo m1 = read_proc_mem();
                uint64_t t1 = 0, u1 = 0, a1 = 0;
                aclshmem_get_memory_stats(&t1, &u1, &a1);

                std::cout << "  calloc(1, " << fmt_bytes(kAllocSize) << "):\n"
                          << "    heap used delta = " << fmt_bytes(u1 > u0 ? u1 - u0 : 0) << "\n"
                          << "    proc RSS  delta = " << (m1.vm_rss_kb - m0.vm_rss_kb) << " KB\n";
                if (p) aclshmem_free(p);
            }

            test_finalize(stream, device_id);
            print_separator();
        },
        kLocalMemSize);
}

// ---------------------------------------------------------------------------
// 测试 8：递增分配趋势（stress）
// ---------------------------------------------------------------------------
TEST_F(MemoryUsageTest, stress_alloc_memory_trend)
{
    test_single_task(
        [](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            print_separator("Test: stress_alloc_memory_trend");

            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);

            // 每轮分配 kStepSize，累计，输出堆趋势
            constexpr int    kRounds    = 8;
            const     size_t kStepSize  = local_mem_size / (kRounds + 2);  // 保留余量

            std::vector<void *> ptrs;
            ptrs.reserve(kRounds);

            std::cout << "  step_size = " << fmt_bytes(kStepSize) << "\n";
            std::cout << "  " << std::setw(6)  << "Round"
                      << std::setw(14) << "Total"
                      << std::setw(14) << "Used"
                      << std::setw(14) << "Available"
                      << std::setw(10) << "Util%"
                      << "  RSS(KB)\n";
            std::cout << "  " << std::string(65, '-') << "\n";

            for (int r = 0; r < kRounds; ++r) {
                void *p = aclshmem_malloc(kStepSize);
                if (!p) {
                    std::cout << "  Round " << r << ": malloc failed (OOM)\n";
                    break;
                }
                ptrs.push_back(p);

                uint64_t total = 0, used = 0, avail = 0;
                aclshmem_get_memory_stats(&total, &used, &avail);
                ProcMemInfo pm = read_proc_mem();
                double util = (total > 0) ? (100.0 * used / total) : 0.0;

                std::cout << "  " << std::setw(6) << r
                          << std::setw(14) << fmt_bytes(total)
                          << std::setw(14) << fmt_bytes(used)
                          << std::setw(14) << fmt_bytes(avail)
                          << std::setw(9) << std::fixed << std::setprecision(1) << util << "%"
                          << "  " << pm.vm_rss_kb << "\n";
            }

            // 全部释放，确认堆恢复
            for (void *p : ptrs) {
                aclshmem_free(p);
            }
            ptrs.clear();
            print_heap_stats("  [after all freed]");

            test_finalize(stream, device_id);
            print_separator();
        },
        kLocalMemSize);
}
