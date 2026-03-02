#!/bin/bash

# SHMEM动态内存扩容功能构建脚本

set -e  # 遇到错误立即退出

echo "=== Building SHMEM Dynamic Memory Expansion Test ==="

# 检查环境变量
if [ -z "$ASCEND_HOME" ]; then
    echo "Warning: ASCEND_HOME not set, using default path"
    export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
fi

# 创建构建目录
BUILD_DIR="build"
mkdir -p $BUILD_DIR

# 获取必要的路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHMEM_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
SHMEM_SRC="$SHMEM_ROOT/src"

echo "SHMEM root: $SHMEM_ROOT"
echo "Build directory: $BUILD_DIR"

# 编译动态内存管理器
echo "Compiling dynamic memory manager..."

g++ -std=c++11 -fPIC -shared \
    -I$SHMEM_ROOT/include \
    -I$ASCEND_HOME/include \
    -I$SHMEM_SRC/host/mem \
    -L$ASCEND_HOME/lib64 \
    -lascendcl \
    -o $BUILD_DIR/libshmem_dynamic.so \
    $SHMEM_SRC/host/mem/shmem_dynamic_mm.cpp \
    $SHMEM_SRC/host/mem/shmem_mm.cpp \
    $SHMEM_SRC/host/mem/shmem_mgr.cpp

echo "Dynamic memory manager compiled successfully!"

# 验证构建结果
if [ -f "$BUILD_DIR/libshmem_dynamic.so" ]; then
    echo "Build successful! Library created at: $BUILD_DIR/libshmem_dynamic.so"
    
    # 显示库信息
    echo "Library info:"
    ls -lh $BUILD_DIR/libshmem_dynamic.so
    
    # 如果有readelf命令，显示符号表
    if command -v readelf &> /dev/null; then
        echo "Exported symbols:"
        readelf -Ws $BUILD_DIR/libshmem_dynamic.so | grep "aclshmem\|dynamic_" | head -20
    fi
else
    echo "Build failed: Library not found!"
    exit 1
fi

echo "=== Build completed successfully ==="