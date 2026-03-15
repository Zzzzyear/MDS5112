import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from depth_model import ResNet50DepthModel
from scannet_dataset import ScanNetDepthDataset
from metrics import solve_scale_shift

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scannet_root = "./data/scannet"
    split_file = "./data/scannet/scannetv2_val.txt"
    checkpoint_path = "./checkpoints/baseline/31.pth"
    output_path = "./results/qualitative_results.png"

    # 1. 载入 Baseline 模型
    model = ResNet50DepthModel(min_depth=0.1, max_depth=10.0, pretrained_backbone=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # 2. 载入数据集 (开启 shuffle 随机抽取)
    with open(split_file, 'r') as f:
        scenes = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    dataset = ScanNetDepthDataset(scannet_root=scannet_root, scenes=scenes, augment=False)
    
    # 设置提取 5 个样本
    loader = DataLoader(dataset, batch_size=5, shuffle=True)

    # 3. 取一个 Batch 进行推理
    batch = next(iter(loader))
    image = batch["image"].to(device)
    depth = batch["depth"].to(device)
    valid_mask = batch["valid_mask"].to(device)

    with torch.no_grad():
        pred = model(image)
        # 尺度对齐以进行公平的可视化对比
        aligned_pred = solve_scale_shift(pred, depth, valid_mask)

    # 4. 数据反归一化与绘图准备
    image_np = image.cpu().permute(0, 2, 3, 1).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)

    gt_np = depth.cpu().squeeze(1).numpy()
    pred_np = aligned_pred.cpu().squeeze(1).numpy()

    # 5. 绘制 5x3 对比图
    fig, axes = plt.subplots(5, 3, figsize=(12, 16))
    for i in range(5):
        # RGB Input
        axes[i, 0].imshow(image_np[i])
        if i == 0: axes[i, 0].set_title("RGB Input")
        axes[i, 0].axis('off')
        
        # Ground Truth
        axes[i, 1].imshow(gt_np[i], cmap='plasma', vmin=0.1, vmax=5.0)
        if i == 0: axes[i, 1].set_title("Ground Truth Depth")
        axes[i, 1].axis('off')
        
        # Predicted Depth
        axes[i, 2].imshow(pred_np[i], cmap='plasma', vmin=0.1, vmax=5.0)
        if i == 0: axes[i, 2].set_title("Predicted Depth (Aligned)")
        axes[i, 2].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Qualitative results successfully saved to {output_path}")

if __name__ == "__main__":
    main()