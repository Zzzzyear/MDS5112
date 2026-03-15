import os
import numpy as np
import cv2

# 设置假数据的存放路径
scannet_root = "./scannet"
# 我们伪造两个场景：一个用于训练，一个用于验证/测试
scenes = ["scene0000_00", "scene0001_00"]

os.makedirs(scannet_root, exist_ok=True)

for scene in scenes:
    scene_dir = os.path.join(scannet_root, scene)
    color_dir = os.path.join(scene_dir, "color")
    depth_dir = os.path.join(scene_dir, "depth")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    # 每个场景生成 100 张随机图片和深度图
    print(f"Generating dummy data for {scene}...")
    for i in range(100):
        # 1. 生成随机的 RGB 图像 (240x320)
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(color_dir, f"{i:05d}.jpg"), img)
        
        # 2. 生成随机的深度图
        # ScanNet 的深度图是 16位 PNG，单位是毫米 (mm)。
        # 这里生成 1000mm 到 5000mm (即 1米 到 5米) 的随机深度
        depth = np.random.randint(1000, 5000, (240, 320), dtype=np.uint16)
        cv2.imwrite(os.path.join(depth_dir, f"{i:05d}.png"), depth)

# 3. 生成数据集划分文件
with open("scannetv2_train.txt", "w") as f:
    f.write("scene0000_00\n")
    
with open("scannetv2_val.txt", "w") as f:
    f.write("scene0001_00\n")

print("\nSuccess! Dummy ScanNet dataset (200 frames total) created.")
print("Directory structure:")
print("data/")
print("├── scannet/")
print("│   ├── scene0000_00/ (color/, depth/)")
print("│   └── scene0001_00/ (color/, depth/)")
print("├── scannetv2_train.txt")
print("└── scannetv2_val.txt")
