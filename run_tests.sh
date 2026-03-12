#!/bin/bash
# 快速运行测试脚本

set -e

echo "=========================================="
echo "Nozom1LLama 测试脚本"
echo "=========================================="
echo ""

# 检查构建目录
if [ ! -d "build/test" ]; then
    echo "❌ 错误: 测试文件不存在，请先运行 build.sh 编译项目"
    exit 1
fi

echo "🧪 运行测试..."
echo ""

# 运行所有测试
ctest --output-on-failure --test-dir build

echo ""
echo "=========================================="
echo "✅ 测试完成！"
echo "=========================================="
echo ""

# 显示测试文件位置
echo "📁 测试可执行文件位置:"
find build/test -type f -executable 2>/dev/null | while read file; do
    echo "  - $file"
done

echo ""
echo "💡 手动运行特定测试:"
echo "  ./build/test/test_base/test_status"
echo "  ./build/test/test_base/test_datatype"
