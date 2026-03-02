#!/bin/bash

# SHMEM NPUPluggableAllocator 一键运行脚本

set -e  # 遇到错误立即退出

echo "=== SHMEM NPUPluggableAllocator 示例运行脚本 ==="

# 检查环境变量
echo "检查环境变量..."
if [ -z "$ASCEND_HOME" ]; then
    echo "警告: ASCEND_HOME 未设置，使用默认路径"
    export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
fi

if [ -z "$PYTHONPATH" ]; then
    echo "设置PYTHONPATH..."
    export PYTHONPATH=$(pwd)/../../../pytorch_npu:$PYTHONPATH
fi

# 检查必要文件
echo "检查必要文件..."
if [ ! -f "shmem_pluggable_allocator.cpp" ]; then
    echo "错误: 找不到 shmem_pluggable_allocator.cpp"
    exit 1
fi

if [ ! -f "example_usage.py" ]; then
    echo "错误: 找不到 example_usage.py"
    exit 1
fi

# 创建构建目录
echo "创建构建目录..."
mkdir -p build

# 构建C++扩展
echo "构建C++扩展..."
cd build

# 使用CMake构建（如果可用）
if command -v cmake &> /dev/null; then
    echo "使用CMake构建..."
    cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
    make -j$(nproc)
else
    # 使用直接编译
    echo "使用直接编译..."
    g++ -shared -fPIC \
        -I$(python3 -c 'import torch; print(torch.__path__[0])')/include \
        -I$(python3 -c 'import torch_npu; print(torch_npu.__path__[0])')/include \
        -I../../include \
        -I$ASCEND_HOME/include \
        -L$ASCEND_HOME/lib64 \
        -lascendcl -lacl_shmem \
        -o shmem_pluggable_allocator.so \
        ../shmem_pluggable_allocator.cpp
fi

cd ..

# 检查构建结果
if [ ! -f "build/shmem_pluggable_allocator.so" ]; then
    echo "错误: 构建失败，找不到生成的.so文件"
    exit 1
fi

echo "构建成功！"

# 运行Python示例
echo "运行Python示例..."
python3 example_usage.py

echo "=== 示例运行完成 ==="