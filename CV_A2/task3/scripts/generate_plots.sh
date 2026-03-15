#!/bin/bash
set -e

# 确保脚本在 task3 根目录下执行
cd "$(dirname "$0")/.."

echo ">>> [1/2] Generating ablation curve..."
python plot_results.py

echo ">>> [2/2] Generating qualitative visualizations..."
python visualize.py

echo ">>> All plots generated successfully in ./results"