@echo off
setlocal enabledelayedexpansion

echo === SHMEM NPUPluggableAllocator 示例运行脚本 (Windows) ===

REM 检查环境变量
echo 检查环境变量...
if "%ASCEND_HOME%"=="" (
    echo 警告: ASCEND_HOME 未设置，使用默认路径
    set ASCEND_HOME=C:\Program Files\Ascend\ascend-toolkit\latest
)

if "%PYTHONPATH%"=="" (
    echo 设置PYTHONPATH...
    set PYTHONPATH=%cd%\..\..\..\pytorch_npu;%PYTHONPATH%
)

REM 检查必要文件
echo 检查必要文件...
if not exist "shmem_pluggable_allocator.cpp" (
    echo 错误: 找不到 shmem_pluggable_allocator.cpp
    exit /b 1
)

if not exist "example_usage.py" (
    echo 错误: 找不到 example_usage.py
    exit /b 1
)

REM 创建构建目录
echo 创建构建目录...
if not exist "build" mkdir build

REM 尝试使用Visual Studio构建
echo 尝试使用Visual Studio构建...
cd build

REM 检查是否有cl.exe (Visual Studio)
where cl >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo 使用Visual Studio编译...
    cl /LD /EHsc ^
        /I"%PYTHONPATH%\torch\include" ^
        /I"%PYTHONPATH%\torch_npu\include" ^
        /I"..\..\include" ^
        /I"%ASCEND_HOME%\include" ^
        /link ^
        /LIBPATH:"%ASCEND_HOME%\lib64" ^
        ascendcl.lib acl_shmem.lib ^
        /OUT:shmem_pluggable_allocator.dll ^
        ..\shmem_pluggable_allocator.cpp
) else (
    REM 尝试使用g++ (MinGW)
    where g++ >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo 使用g++编译...
        g++ -shared -fPIC ^
            -I"%PYTHONPATH%\torch\include" ^
            -I"%PYTHONPATH%\torch_npu\include" ^
            -I"..\..\include" ^
            -I"%ASCEND_HOME%\include" ^
            -L"%ASCEND_HOME%\lib64" ^
            -lascendcl -lacl_shmem ^
            -o shmem_pluggable_allocator.dll ^
            ..\shmem_pluggable_allocator.cpp
    ) else (
        echo 错误: 找不到编译器 (cl.exe 或 g++)
        echo 请安装Visual Studio或MinGW
        exit /b 1
    )
)

cd ..

REM 检查构建结果
if not exist "build\shmem_pluggable_allocator.dll" (
    echo 错误: 构建失败，找不到生成的.dll文件
    exit /b 1
)

echo 构建成功！

REM 运行Python示例
echo 运行Python示例...
python example_usage.py

echo === 示例运行完成 ===
pause