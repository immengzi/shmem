#!/bin/bash

# 编译 SHMEM 动态内存扩容 C++ 测试程序
# libshmem.so 由主包构建，本脚本只负责编译 test_dynamic_cpp

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHMEM_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"

g++ "$SCRIPT_DIR/test_dynamic_cpp.cpp" \
    -I "$SHMEM_ROOT/include" \
    -I "$ASCEND_HOME/include" \
    -L "$SHMEM_ROOT/build/lib" \
    -L "$ASCEND_HOME/lib64" \
    -lshmem \
    -lascendcl \
    -Wl,-rpath,"$SHMEM_ROOT/build/lib" \
    -o "$SCRIPT_DIR/test_dynamic_cpp"

echo "Build successful: $SCRIPT_DIR/test_dynamic_cpp"