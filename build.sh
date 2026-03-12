#!/bin/bash
# 快速构建和测试脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Nozom1LLama 构建脚本"
echo "=========================================="
echo ""

# 检查 CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ 错误: 未找到 nvcc，请先安装 CUDA Toolkit"
    exit 1
fi

echo "✅ CUDA 版本: $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"
echo ""

# 创建构建目录
echo "📁 创建构建目录..."
mkdir -p build
cd build

# 配置 CMake
echo "🔧 配置 CMake..."
if [ "$1" == "--cpm" ]; then
    echo "   使用 CPM 下载依赖..."
    cmake -DUSE_CPM=ON ..
else
    echo "   使用系统依赖..."
    cmake ..
fi

echo ""
echo "🔨 编译项目..."
make -j$(nproc)

echo ""
echo "=========================================="
echo "✅ 编译成功！"
echo "=========================================="
echo ""

# 检查是否生成了测试可执行文件
if [ -f "tests/test_status" ]; then
    echo "🧪 运行测试..."
    echo ""
    ctest --output-on-failure
    echo ""
    echo "=========================================="
    echo "✅ 所有测试通过！"
    echo "=========================================="
else
    echo "⚠️  测试未生成，请检查编译错误"
fi
