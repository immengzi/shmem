/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <sstream>
#include <functional>
#include <unordered_set>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "shmemi_host_common.h"
#include "shmemi_init.h"
#include "host/shmem_host_def.h"
#include "prof/prof_util.h"

using namespace std;

#define DEFAULT_MY_PE (-1)
#define DEFAULT_N_PES (-1)

constexpr int DEFAULT_TEVENT = 0;
constexpr int DEFAULT_BLOCK_NUM = 1;

constexpr uint32_t DEFAULT_MTE_UB_SIZE = 16 * 1024;
constexpr uint32_t DEFAULT_SDMA_UB_SIZE = 64;
constexpr int64_t DEFAULT_SDMA_UB_OFFSET = 191 * 1024;
constexpr uint32_t DEFAULT_RDMA_UB_SIZE = 64;
constexpr int64_t DEFAULT_RDMA_UB_OFFSET = 190 * 1024;
constexpr int64_t LOCAL_MEM_ALIGNMENT = 2LL * 1024 * 1024;

namespace {

int64_t align_up_local_mem_size(int64_t value)
{
    return ((value + LOCAL_MEM_ALIGNMENT - 1) / LOCAL_MEM_ALIGNMENT) *
           LOCAL_MEM_ALIGNMENT;
}

void normalize_local_mem_size(int64_t &local_mem_size)
{
    if (local_mem_size <= 0) {
        return;
    }
    int64_t aligned_local_mem_size = align_up_local_mem_size(local_mem_size);
    if (aligned_local_mem_size != local_mem_size) {
        SHM_LOG_WARN("Round local_mem_size from " << local_mem_size
                     << " to " << aligned_local_mem_size
                     << " bytes to satisfy 2MB heap alignment");
        local_mem_size = aligned_local_mem_size;
    }
}

} // namespace

// initializer
#define ACLSHMEM_DEVICE_HOST_STATE_INITIALIZER                                                         \
    {                                                                                                  \
        (1 << 16) + sizeof(aclshmem_device_host_state_t),       /* version */                          \
            (DEFAULT_MY_PE),                                    /* mype */                             \
            (DEFAULT_N_PES),                                    /* npes */                             \
            NULL,                                               /* heap_base */                        \
            NULL,                                               /* host_heap_base */                   \
            NULL,                                               /* p2p_device_heap_base */             \
            NULL,                                               /* rdma_device_heap_base */            \
            NULL,                                               /* sdma_device_heap_base */            \
            NULL,                                               /* p2p_heap_host_base */               \
            NULL,                                               /* rdma_heap_host_base */              \
            NULL,                                               /* sdma_heap_host_base */              \
            {},                                                 /* topo_list */                        \
            SIZE_MAX,                                           /* heap_size */                        \
            {NULL},                                             /* team_pools */                       \
            0,                                                  /* sync_pool */                        \
            0,                                                  /* sync_counter */                     \
            0,                                                  /* core_sync_pool */                   \
            0,                                                  /* core_sync_counter */                \
            false,                                              /* aclshmem_is_aclshmem_initialized */ \
            false,                                              /* aclshmem_is_aclshmem_created */     \
            {0, DEFAULT_MTE_UB_SIZE, 0},                        /* aclshmem_mte_config */              \
            {DEFAULT_SDMA_UB_OFFSET, DEFAULT_SDMA_UB_SIZE, 0},  /* aclshmem_sdma_config */             \
            {DEFAULT_RDMA_UB_OFFSET, DEFAULT_RDMA_UB_SIZE, 0},  /* aclshmem_rdma_config */             \
            0,                                                  /* qp_info */                          \
            0,                                                  /* sdma_workspace_addr */              \
            NULL,                                               /* aclshmem_prof_pe_t */             \
    }

aclshmem_device_host_state_t g_state = ACLSHMEM_DEVICE_HOST_STATE_INITIALIZER;
aclshmem_host_state_t g_state_host = {nullptr, DEFAULT_TEVENT, DEFAULT_BLOCK_NUM};
aclshmem_prof_pe_t g_host_profs;

aclshmemi_init_backend* init_manager = nullptr;

int32_t version_compatible()
{
    int32_t status = ACLSHMEM_SUCCESS;
    return status;
}

int32_t aclshmemi_state_init_attr(aclshmemx_init_attr_t *attributes)
{
    int32_t status = ACLSHMEM_SUCCESS;
    g_state.mype = attributes->my_pe;
    g_state.npes = attributes->n_pes;
    g_state.heap_size = attributes->local_mem_size + ACLSHMEM_EXTRA_SIZE;

    aclrtStream stream = nullptr;
    ACLSHMEM_CHECK_RET(aclrtCreateStream(&stream));
    g_state_host.default_stream = stream;
    g_state_host.default_event_id = DEFAULT_TEVENT;
    g_state_host.default_block_num = DEFAULT_BLOCK_NUM;
    return status;
}

bool is_valid_data_op_engine_type(data_op_engine_type_t value)
{
    static const std::unordered_set<int> valid_values = {
        ACLSHMEM_DATA_OP_MTE,
        ACLSHMEM_DATA_OP_SDMA,
        ACLSHMEM_DATA_OP_ROCE
    };

    return valid_values.find(static_cast<int>(value)) != valid_values.end();
}

int32_t check_attr(aclshmemx_init_attr_t *attributes)
{
    normalize_local_mem_size(attributes->local_mem_size);
    SHM_LOG_DEBUG("check_attr my_pe=" << attributes->my_pe << " n_pes=" << attributes->n_pes << " local_mem_size=" << attributes->local_mem_size
                                      << " shm_init_timeout=" << attributes->option_attr.shm_init_timeout
                                      << " control_operation_timeout=" << attributes->option_attr.control_operation_timeout);
    SHM_VALIDATE_RETURN(attributes->my_pe >= 0, "my_pe is less than zero", ACLSHMEM_INVALID_VALUE);
    SHM_VALIDATE_RETURN(attributes->n_pes > 0, "n_pes is less than or equal to zero", ACLSHMEM_INVALID_VALUE);
    SHM_VALIDATE_RETURN(attributes->n_pes <= ACLSHMEM_MAX_PES, "n_pes is too large", ACLSHMEM_INVALID_VALUE);
    SHM_VALIDATE_RETURN(attributes->my_pe < attributes->n_pes, "my_pe is greater than or equal to n_pes", ACLSHMEM_INVALID_PARAM);
    SHM_VALIDATE_RETURN(attributes->local_mem_size > 0, "local_mem_size less than or equal to 0", ACLSHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(attributes->local_mem_size <= ACLSHMEM_MAX_LOCAL_SIZE, ACLSHMEM_INVALID_VALUE);

    SHM_VALIDATE_RETURN(attributes->option_attr.shm_init_timeout != 0, "shm_init_timeout is zero", ACLSHMEM_INVALID_VALUE);
    SHM_VALIDATE_RETURN(attributes->option_attr.control_operation_timeout != 0, "control_operation_timeout is zero", ACLSHMEM_INVALID_VALUE);
    SHM_VALIDATE_RETURN(attributes->option_attr.data_op_engine_type > 0, "sockFd is invalid", ACLSHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(is_valid_data_op_engine_type(attributes->option_attr.data_op_engine_type), ACLSHMEM_INVALID_VALUE);
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_control_barrier_all()
{
    return init_manager->aclshmemi_control_barrier_all();
}

int32_t is_alloc_size_symmetric(size_t size)
{
    return init_manager->is_alloc_size_symmetric(size);
}

int32_t update_device_state()
{
    return init_manager->update_device_state((void *)&g_state, sizeof(aclshmem_device_host_state_t));
}


int32_t aclshmemx_init_status(void)
{
    if (!g_state.is_aclshmem_created)
        return ACLSHMEM_STATUS_NOT_INITIALIZED;
    else if (!g_state.is_aclshmem_initialized)
        return ACLSHMEM_STATUS_SHM_CREATED;
    else if (g_state.is_aclshmem_initialized)
        return ACLSHMEM_STATUS_IS_INITIALIZED;
    else
        return ACLSHMEM_STATUS_INVALID;
}

int aclshmemx_set_attr_uniqueid_args(int my_pe, int n_pes, int64_t local_mem_size,
                                    aclshmemx_uniqueid_t *uid,
                                    aclshmemx_init_attr_t *aclshmem_attr) {
    /* Save to uid_args */
    SHM_ASSERT_RETURN(local_mem_size > 0, ACLSHMEM_INVALID_VALUE);
    normalize_local_mem_size(local_mem_size);
    SHM_ASSERT_RETURN(local_mem_size <= ACLSHMEM_MAX_LOCAL_SIZE, ACLSHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(n_pes <= ACLSHMEM_MAX_PES, ACLSHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(my_pe < ACLSHMEM_MAX_PES, ACLSHMEM_INVALID_VALUE);
    aclshmemi_bootstrap_uid_state_t *uid_args = (aclshmemi_bootstrap_uid_state_t *)(uid);
    void * comm_args = reinterpret_cast<void *>(uid_args);
    aclshmem_attr->comm_args = comm_args;
    aclshmem_attr->my_pe = my_pe;
    aclshmem_attr->n_pes = n_pes;
    aclshmem_attr->local_mem_size = local_mem_size;

    return ACLSHMEM_SUCCESS;
}

bool check_support_d2h()
{
    int32_t major_version = -1;
    int32_t minor_version = -1;
    int32_t patch_version = -1;
    if (aclrtGetVersion(&major_version, &minor_version, &patch_version) != 0) {
        SHM_LOG_INFO("aclrtGetVersion failed, disable d2h feature");
        return false;
    }
    if (major_version <= 1 && minor_version < 15) {
        SHM_LOG_INFO("The current AscendCL version is "
            << major_version << "." << minor_version << "." << patch_version << ", which does not support d2h.");
        return false;
    }
    return true;
}

int32_t aclshmemx_init_attr(aclshmemx_bootstrap_t bootstrap_flags, aclshmemx_init_attr_t *attributes)
{
    int32_t ret;
    // config init
    SHM_ASSERT_RETURN(attributes != nullptr, ACLSHMEM_INVALID_PARAM);
    ACLSHMEM_CHECK_RET(aclshmemx_init_status() != ACLSHMEM_STATUS_NOT_INITIALIZED, "SHMEM has been initialized, do not call init interface repeatedly!", ACLSHMEM_INNER_ERROR);
    ACLSHMEM_CHECK_RET(aclshmemx_set_log_level(aclshmem_log::ERROR_LEVEL));
    ACLSHMEM_CHECK_RET(check_attr(attributes), "An error occurred while checking the initialization attributes. Please check the initialization parameters.");
    ACLSHMEM_CHECK_RET(version_compatible(), "ACLSHMEM Version mismatch.");
    // init bootstrap
    ACLSHMEM_CHECK_RET(aclshmemi_bootstrap_init(bootstrap_flags, attributes));

    // init backend for memory manager
    init_manager = new aclshmemi_init_backend(attributes, &g_state, &g_boot_handle);
    ACLSHMEM_CHECK_RET(aclshmemi_state_init_attr(attributes));

    ACLSHMEM_CHECK_RET(init_manager->init_device_state());
    ACLSHMEM_CHECK_RET(init_manager->reserve_heap());
    ACLSHMEM_CHECK_RET(init_manager->setup_heap());

    // shmem submodules init
    ACLSHMEM_CHECK_RET(memory_manager_initialize(g_state.heap_base, g_state.heap_size));

#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE
    if (check_support_d2h()) {
        // only reserve dramp heap, skip setup_heap for host dram, setup heap when malloc on host
        ACLSHMEM_CHECK_RET(init_manager->reserve_heap(HOST_SIDE));
    }
#endif

    ACLSHMEM_CHECK_RET(aclshmemi_team_init(g_state.mype, g_state.npes));
    ACLSHMEM_CHECK_RET(aclshmemi_sync_init());	
    g_state.is_aclshmem_initialized = true;
    ACLSHMEM_CHECK_RET(prof_util_init(&g_host_profs, &g_state));
    ACLSHMEM_CHECK_RET(update_device_state());
    ACLSHMEM_CHECK_RET(aclshmemi_control_barrier_all());
    SHM_LOG_INFO("The ACLSHMEM pe: " << aclshmem_my_pe() << " init success.");
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmem_finalize()
{
    SHM_LOG_INFO("The pe: " << aclshmem_my_pe() << " begins to finalize.");
    // shmem submodules finalize
    ACLSHMEM_CHECK_RET(aclshmemi_team_finalize());
    if (init_manager == nullptr) {
        SHM_LOG_INFO("init_manager is null finalize success.");
        g_state.is_aclshmem_initialized = false;
        return ACLSHMEM_SUCCESS;
    }
    // shmem basic finalize
    ACLSHMEM_CHECK_RET(init_manager->remove_heap());
    ACLSHMEM_CHECK_RET(init_manager->release_heap());
    SHM_LOG_INFO("release_heap success.");
#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE
    if (check_support_d2h()) {
        ACLSHMEM_CHECK_RET(init_manager->remove_heap(HOST_SIDE));
        ACLSHMEM_CHECK_RET(init_manager->release_heap(HOST_SIDE));
    }
#endif
    ACLSHMEM_CHECK_RET(init_manager->finalize_device_state());
    delete init_manager;
    init_manager = nullptr;
    if (g_state_host.default_stream != nullptr) {
      ACLSHMEM_CHECK_RET(aclrtSynchronizeStream(g_state_host.default_stream));
      ACLSHMEM_CHECK_RET(aclrtDestroyStream(g_state_host.default_stream));
      g_state_host.default_stream = nullptr;
    }
    memory_manager_destroy();
    aclshmemi_bootstrap_finalize();
    SHM_LOG_INFO("The pe: " << aclshmem_my_pe() << " finalize success.");
    g_state.is_aclshmem_initialized = false;
    return ACLSHMEM_SUCCESS;
}

void aclshmem_info_get_version(int *major, int *minor)
{
    SHM_ASSERT_RET_VOID(major != nullptr && minor != nullptr);
    *major = ACLSHMEM_MAJOR_VERSION;
    *minor = ACLSHMEM_MINOR_VERSION;
}

void aclshmem_info_get_name(char *name)
{
    SHM_ASSERT_RET_VOID(name != nullptr);
    std::ostringstream oss;
    oss << "ACLSHMEM v" << ACLSHMEM_VENDOR_MAJOR_VER << "." << ACLSHMEM_VENDOR_MINOR_VER << "." << ACLSHMEM_VENDOR_PATCH_VER;
    auto version_str = oss.str();
    size_t i;
    for (i = 0; i < ACLSHMEM_MAX_NAME_LEN - 1 && version_str[i] != '\0'; i++) {
        name[i] = version_str[i];
    }
    name[i] = '\0';
}

int32_t aclshmemx_get_uniqueid(aclshmemx_uniqueid_t *uid)
{
    int status = aclshmemx_set_log_level(aclshmem_log::ERROR_LEVEL);

    ACLSHMEM_CHECK_RET(aclshmemi_bootstrap_pre_init(ACLSHMEMX_INIT_WITH_UNIQUEID, &g_boot_handle), "Get uniqueid failed during the bootstrap preloading step.");

    *uid = ACLSHMEM_UNIQUEID_INITIALIZER;
    if (g_boot_handle.pre_init_ops) {
        ACLSHMEM_CHECK_RET(g_boot_handle.pre_init_ops->get_unique_id((void *)uid), "Get uniqueid failed during the get uniqueid step.");
    } else {
        SHM_LOG_ERROR("Pre_init_ops is empty, unique_id cannot be obtained.");
        status = ACLSHMEM_INVALID_PARAM;
    }
    return status;
}

int32_t aclshmemx_set_log_level(int level)
{
    // use env first, input level secondly, user may change level from env instead call func
    const char *in_level = std::getenv("SHMEM_LOG_LEVEL");
    if (in_level != nullptr) {
        auto tmp_level = std::string(in_level);
        if (tmp_level == "DEBUG") {
            level = aclshmem_log::DEBUG_LEVEL;
        } else if (tmp_level == "INFO") {
            level = aclshmem_log::INFO_LEVEL;
        } else if (tmp_level == "WARN") {
            level = aclshmem_log::WARN_LEVEL;
        } else if (tmp_level == "ERROR") {
            level = aclshmem_log::ERROR_LEVEL;
        } else if (tmp_level == "FATAL") {
            level = aclshmem_log::FATAL_LEVEL;
        }
    }

    return aclshmem_log::aclshmem_out_logger::Instance().set_log_level(static_cast<aclshmem_log::log_level>(level));
}

int32_t aclshmemx_set_conf_store_tls(bool enable, const char *tls_info, const uint32_t tls_info_len)
{
    g_boot_handle.tls_enable = enable;
    g_boot_handle.tls_info = tls_info;
    g_boot_handle.tls_info_len = tls_info_len;

    return ACLSHMEM_SUCCESS;
}

void aclshmem_rank_exit(int status)
{
    SHM_LOG_DEBUG("aclshmem_rank_exit is work ,status: " << status);
    exit(status);
}

int32_t aclshmemx_set_config_store_tls_key(const char *tls_pk, const uint32_t tls_pk_len,
    const char *tls_pk_pw, const uint32_t tls_pk_pw_len, const aclshmem_decrypt_handler decrypt_handler)
{
    g_boot_handle.tls_pk = tls_pk;
    g_boot_handle.tls_pk_len = tls_pk_len;
    g_boot_handle.tls_pk_pw = tls_pk_pw;
    g_boot_handle.tls_pk_pw_len = tls_pk_pw_len;
    g_boot_handle.decrypt_handler = decrypt_handler;

    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemx_set_extern_logger(void (*func)(int level, const char *msg))
{
    aclshmem_log::aclshmem_out_logger::Instance().set_extern_log_func(func);
    return ACLSHMEM_SUCCESS;
}

void aclshmem_global_exit(int status)
{
    if (g_boot_handle.is_bootstraped == true) {
        g_boot_handle.global_exit(status);
    }
    SHM_LOG_WARN("Bootstrap not initialized. Global_exit Do nothing. ");
}

void aclshmemx_show_prof()
{
    prof_data_print(&g_host_profs, &g_state);
}
