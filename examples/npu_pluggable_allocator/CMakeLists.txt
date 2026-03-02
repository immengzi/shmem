cmake_minimum_required(VERSION 3.19)
project(shmem_pluggable_allocator)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找PyTorch
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 设置包含目录
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)
include_directories(/usr/local/Ascend/ascend-toolkit/latest/include)

# 设置链接目录
link_directories(/usr/local/Ascend/ascend-toolkit/latest/lib64)

# 创建共享库
add_library(shmem_pluggable_allocator SHARED
    shmem_pluggable_allocator.cpp
)

# 链接库
target_link_libraries(shmem_pluggable_allocator
    ${TORCH_LIBRARIES}
    ascendcl
    acl_shmem
    ${Python3_LIBRARIES}
)

# 设置输出目录
set_target_properties(shmem_pluggable_allocator PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# 安装规则
install(TARGETS shmem_pluggable_allocator
    LIBRARY DESTINATION lib
)