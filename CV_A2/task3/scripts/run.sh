#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# 环境变量与通用配置
export HF_ENDPOINT=https://hf-mirror.com
SCANNET_ROOT="./data/scannet"
# TRAIN_SPLIT="./data/scannetv2_train.txt"
# VAL_SPLIT="./data/scannetv2_val.txt"
TRAIN_SPLIT="./data/scannet/scannetv2_train.txt"
VAL_SPLIT="./data/scannet/scannetv2_val.txt"

# 目录创建
OUT_DIR_BASELINE="./checkpoints/baseline"
RESULTS_DIR_BASELINE="./results/baseline"
RESULTS_DIR_ABLATION="./results/ablation"

mkdir -p $OUT_DIR_BASELINE
mkdir -p $RESULTS_DIR_BASELINE
mkdir -p $RESULTS_DIR_ABLATION

echo "========================================"
echo "Task 3: Monocular Depth Estimation Pipeline"
echo "========================================"

# 1. Baseline 实验
echo ">>> [1/3] 开始训练与评估 Baseline 模型..."
python train.py --scannet_root $SCANNET_ROOT --output_dir $OUT_DIR_BASELINE --train_split_file $TRAIN_SPLIT --epochs 32 --max_train_samples 1000 --batch_size 8 
python test.py --scannet_root $SCANNET_ROOT --split_file $VAL_SPLIT --checkpoint $OUT_DIR_BASELINE/31.pth --save_json $RESULTS_DIR_BASELINE/metrics.json
echo ">>> Baseline 实验完成！"

# # 2. 数据规模消融实验 (Ablation Study)
EPOCHS=(16 8 4 2)
SAMPLES=(20000 40000 80000 160000)

echo ">>> [2/3] 开始数据规模对比实验 (Ablation Study)..."
for i in "${!EPOCHS[@]}"; do
    E=${EPOCHS[$i]}
    S=${SAMPLES[$i]}
    OUT_DIR="./checkpoints/exp_${E}ep_${S}k"
    
    echo "--- Training Exp: ${E} Epochs, ${S} Samples ---"
    python train.py --scannet_root $SCANNET_ROOT --output_dir $OUT_DIR --train_split_file $TRAIN_SPLIT --epochs $E --max_train_samples $S --batch_size 16
    
    # 训练从 0 开始计数，因此最后一个 epoch 的 checkpoint 是 E-1
    LAST_EPOCH=$((E - 1))
    CKPT_FILE="${OUT_DIR}/${LAST_EPOCH}.pth"
    JSON_FILE="${RESULTS_DIR_ABLATION}/metrics_${E}ep_${S}k.json"
    
    echo "--- Evaluating Exp: ${E} Epochs, ${S} Samples ---"
    python test.py --scannet_root $SCANNET_ROOT --split_file $VAL_SPLIT --checkpoint $CKPT_FILE --save_json $JSON_FILE
done
echo ">>> 所有数据规模实验训练与评估完成！"

# 3. 大模型评估 (Foundation Models)
echo ">>> [3/3] 开始评估 Foundation Models..."
echo "--- Evaluating Depth Anything 3 (Large) ---"
python test_foundation.py --scannet_root $SCANNET_ROOT --split_file $VAL_SPLIT --model_name da3 --batch_size 1

echo "--- Evaluating VGGT (1B) ---"
python test_foundation.py --scannet_root $SCANNET_ROOT --split_file $VAL_SPLIT --model_name vggt --batch_size 1
echo ">>> Foundation Models 评估完成！"

echo "========================================"
echo "所有 Task 3 实验已全部运行完毕！"
echo "========================================"

# nohup bash scripts/run.sh > results/all_execution.log 2>&1 &