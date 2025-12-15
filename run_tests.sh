#!/bin/bash
# 运行所有测试的快捷脚本
# 使用方法: ./run_tests.sh

echo "==================================="
echo "运行测试套件"
echo "==================================="
echo ""

echo "1. 测试修复..."
python tests/test_fixes.py
echo ""

echo "2. 测试相对特征..."
python tests/test_relative_features.py
echo ""

echo "3. 测试连续性..."
python tests/test_continuity.py
echo ""

echo "==================================="
echo "测试完成!"
echo "==================================="
