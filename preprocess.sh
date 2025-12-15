#!/bin/bash
# 数据预处理快捷脚本
# 使用方法: ./preprocess.sh [参数]
# 示例: ./preprocess.sh --sites A --use_relative_features

python src/data/preprocess_multisite.py "$@"
