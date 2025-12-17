#!/bin/bash
# 数据验证快捷脚本
# 使用方法: ./validate.sh [参数]
# 示例: ./validate.sh --processed_dir data/processed_siteA

python src/data/validate_preprocessing.py "$@"
