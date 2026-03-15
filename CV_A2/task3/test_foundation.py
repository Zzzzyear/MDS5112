#!/usr/bin/env python
import argparse
import torch
import math
import sys
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scannet_dataset import ScanNetDepthDataset
from metrics import abs_rel_metric, solve_scale_shift

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "models", "da3", "Depth-Anything-3", "src"))
sys.path.append(os.path.join(ROOT_DIR, "models", "vggt", "vggt"))

def get_foundation_model(model_name, device):
    print(f"Loading {model_name}...")
    if model_name == "da3":
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1").to(device).eval()
    elif model_name == "vggt":
        from vggt.models.vggt import VGGT
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

@torch.no_grad()
def evaluate_foundation_model(args, dataset, device):
    model = get_foundation_model(args.model_name, device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    total_abs_rel = 0.0
    count = 0

    for i, batch in enumerate(loader):
        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        # ==========================================
        # 1. 统一的图像预处理 (适应 ViT 的 14x14 Patch)
        # ==========================================
        image_4d = image.squeeze(1) if image.dim() == 5 else image
        original_h, original_w = image_4d.shape[-2:]

        new_h = (original_h // 14) * 14
        new_w = (original_w // 14) * 14

        image_resized = F.interpolate(image_4d, size=(new_h, new_w), mode='bilinear', align_corners=False)
        image_5d = image_resized.unsqueeze(1) # 转为 (B, 1, C, H, W)

        # ==========================================
        # 2. 模型前向传播
        # ==========================================
        if args.model_name == "vggt":
            with torch.amp.autocast('cuda', dtype=torch.float16):
                aggregated_tokens_list, ps_idx = model.aggregator(image_5d)
                pred_out, _ = model.depth_head(aggregated_tokens_list, image_5d, ps_idx)
                pred_5d = pred_out
                
        elif args.model_name == "da3":
            net = getattr(model, 'model', model) 
            pred_out = net(image_5d)

            if isinstance(pred_out, dict):
                if 'depth' in pred_out:
                    pred_5d = pred_out['depth']
                elif 'pred' in pred_out:
                    pred_5d = pred_out['pred']
                else:
                    pred_5d = list(pred_out.values())[0]
            elif isinstance(pred_out, (list, tuple)):
                pred_5d = pred_out[0]
            else:
                pred_5d = pred_out

        # ==========================================
        # 3. 统一的预测结果后处理 (还原原始尺寸)
        # ==========================================
        # 1. 安全降维：将模型输出转换为 4D (Batch, Channel, Height, Width) 用于插值
        B = image.shape[0]  # 获取真实的 Batch Size
        if pred_5d.dim() >= 3:
            # 忽略多余的 size 为 1 的维度，强制规整为 (B, C, H_new, W_new)
            pred_4d = pred_5d.view(B, -1, pred_5d.shape[-2], pred_5d.shape[-1])
            # 如果模型输出了多个通道（如包含置信度），仅提取第一个深度通道
            if pred_4d.shape[1] > 1:
                pred_4d = pred_4d[:, 0:1, :, :]
        else:
            pred_4d = pred_5d

        # 2. 插值恢复到输入图像的原始分辨率
        pred = F.interpolate(pred_4d.float(), size=(original_h, original_w), mode='bilinear', align_corners=False)

        # 3. 【核心修复】强制对齐：剥离所有冗余维度，与 Ground Truth 标签形状保持绝对一致
        pred = pred.reshape(depth.shape)

        # 尺度对齐与误差计算
        aligned_pred = solve_scale_shift(pred, depth, valid_mask)
        abs_rel = abs_rel_metric(aligned_pred, depth, valid_mask)
        
        if math.isnan(abs_rel): 
            continue
            
        total_abs_rel += abs_rel
        count += 1
        
        if i % 10 == 0: 
            print(f"[{args.model_name}] step {i}/{len(loader)} abs_rel={abs_rel:.4f}")

    return total_abs_rel / count if count > 0 else float('inf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannet_root", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, choices=["da3", "vggt"], required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.split_file, 'r') as f:
        scenes = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
    dataset = ScanNetDepthDataset(scannet_root=args.scannet_root, scenes=scenes, augment=False)
    final_abs_rel = evaluate_foundation_model(args, dataset, device)
    
    print(f"\n=== {args.model_name.upper()} Evaluation Results ===")
    print(f"AbsRel: {final_abs_rel:.6f}")