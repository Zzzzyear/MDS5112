import os
import json
import re
import matplotlib.pyplot as plt

# 1. 从 JSON 文件动态提取有监督模型的 AbsRel 结果
samples = [1000, 20000, 40000, 80000, 160000]
json_paths = [
    "results/baseline/metrics.json",
    "results/ablation/metrics_16ep_20000k.json",
    "results/ablation/metrics_8ep_40000k.json",
    "results/ablation/metrics_4ep_80000k.json",
    "results/ablation/metrics_2ep_160000k.json"
]

abs_rel = []
for path in json_paths:
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            abs_rel.append(data["abs_rel"])
    else:
        print(f"警告：未找到文件 {path}，使用默认值 0.0 占位。")
        abs_rel.append(0.0)

# 2. 从执行日志动态提取 Foundation Models 的 Zero-shot 结果
log_file = "results/all_execution.log"
vggt_abs_rel = None
da3_abs_rel = None

if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        content = f.read()
        # 正则匹配日志中的指标数据
        vggt_match = re.search(r'===\s*VGGT Evaluation Results\s*===\nAbsRel:\s*([0-9.]+)', content, re.IGNORECASE)
        da3_match = re.search(r'===\s*DA3 Evaluation Results\s*===\nAbsRel:\s*([0-9.]+)', content, re.IGNORECASE)
        
        if vggt_match:
            vggt_abs_rel = float(vggt_match.group(1))
        if da3_match:
            da3_abs_rel = float(da3_match.group(1))

# 容错：如果提取失败，则回退到备用数值
if vggt_abs_rel is None:
    print("警告：未在日志中提取到 VGGT 结果。")
    vggt_abs_rel = 0.227442
if da3_abs_rel is None:
    print("警告：未在日志中提取到 DA3 结果。")
    da3_abs_rel = 0.056868

# 3. 开始绘图
plt.figure(figsize=(10, 6))

# 绘制有监督模型的 Scaling 曲线
plt.plot(samples, abs_rel, marker='o', linestyle='-', color='#1f77b4', 
         linewidth=2.5, markersize=8, label='Supervised ResNet-50 (Mine)')

# 在数据点上标注具体数值
for i, txt in enumerate(abs_rel):
    plt.annotate(f"{txt:.4f}", (samples[i], abs_rel[i]), 
                 textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)

# 添加基础模型的水平参考线
plt.axhline(y=vggt_abs_rel, color='#d62728', linestyle='--', linewidth=2, 
            label=f'VGGT (1B) Zero-shot: {vggt_abs_rel:.4f}')
plt.axhline(y=da3_abs_rel, color='#2ca02c', linestyle='--', linewidth=2, 
            label=f'DA3 (Large) Zero-shot: {da3_abs_rel:.4f}')

# 4. 设置坐标轴与排版
plt.xscale('log')
plt.xticks(samples, ['1k', '20k', '40k', '80k', '160k'])

plt.xlabel('Number of Training Samples', fontsize=12, fontweight='bold')
plt.ylabel('Absolute Relative Error (AbsRel) ↓ Lower is better', fontsize=12, fontweight='bold')
plt.title('Data Scaling Effect vs. Foundation Models (ScanNet Val)', fontsize=14, fontweight='bold')

plt.grid(True, which="major", linestyle="--", alpha=0.6)
plt.legend(fontsize=11, loc='upper right')
plt.tight_layout()

# 5. 保存图表
save_path = 'results/ablation_curve.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图表已成功保存至: {save_path}")