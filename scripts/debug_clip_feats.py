# core/scripts/debug_clip_feats.py
# -*- coding: utf-8 -*-
import os, sys, math
import numpy as np
import torch

# 把项目根目录加入 sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.scripts.train_min import parse_args, build_dataloader
from core.modeling.vision_encoders import CLIPVisionTower


@torch.no_grad()
def main():
    # 复用原来的命令行参数
    args = parse_args()
    args.dataset = "real"   # 我们要看真实数据上的分布

    # 构造 dataloader（里面会创建 OceanVLADataset）
    loader = build_dataloader(args)
    ds = loader.dataset
    print(f"[debug] dataset size = {len(ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 和训练时一致的 CLIP 模型配置
    clip = CLIPVisionTower(
        name=args.clip_path,
        pool=args.vision_pool if hasattr(args, "vision_pool") else "projected",
        out_dim=None,      # 保持原始维度
        normalize=True,    # 这里手动归一化，便于看 cos 分布
    ).to(device).eval()

    feats = []
    labels = []

    # 每隔 step 采一个样本，避免全部跑一遍太慢
    step = max(1, len(ds) // 500)  # 最多 ~500 个样本
    for idx in range(0, len(ds), step):
        sample = ds[idx]

        # OceanVLADataset 返回 "image"（最后一帧）和可选 "images"（多帧）
        img = sample.get("image", None)
        if img is None:
            continue

        # img: [3,H,W] -> [1,3,H,W]
        img = img.unsqueeze(0).to(device)

        f = clip(img)[0].cpu()  # [D]
        feats.append(f)

        # 拿第一个时间步的离散动作标签看下分布
        y = int(sample["action_targets"][0].item())
        labels.append(y)

    feats = torch.stack(feats, dim=0)   # [N,D]
    labels = np.array(labels, dtype=np.int64)
    print(f"[debug] collected {feats.size(0)} samples for stats")

    # 归一化后算两两 cosine 相似度
    feats = torch.nn.functional.normalize(feats, dim=-1)
    sim = feats @ feats.T   # [N,N]

    # 去掉对角线（自己和自己）
    mask = ~torch.eye(sim.size(0), dtype=torch.bool)
    sims = sim[mask].numpy()

    print(f"[cos] mean={sims.mean():.4f}, std={sims.std():.4f}, "
          f"min={sims.min():.4f}, max={sims.max():.4f}")

    # 简单看一下各类的平均特征之间的距离
    num_classes = int(labels.max()) + 1
    class_means = []
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            class_means.append(None)
            continue
        mu = feats[idx_c].mean(dim=0)   # [D]
        mu = torch.nn.functional.normalize(mu, dim=-1)
        class_means.append(mu)

    print("[cos between class means]")
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            if class_means[i] is None or class_means[j] is None:
                continue
            v = torch.dot(class_means[i], class_means[j]).item()
            print(f"  class {i} vs {j}: cos={v:.4f}")


if __name__ == "__main__":
    main()
