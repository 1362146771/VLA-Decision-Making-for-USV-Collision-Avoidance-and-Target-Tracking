# core/scripts/build_vq_tokens_from_pairs.py
import os, json, glob, re, argparse
from PIL import Image
import numpy as np
from typing import List, Dict

import torch

# 允许从仓库根目录运行：python -m core.scripts.build_vq_tokens_from_pairs
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.world.vq_tokenizer import VQTokenizer

def load_list_or_root(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["root"] if isinstance(data, dict) and "root" in data else data

def walk_groups(pairs_dir: str, k: int) -> List[List[Dict]]:
    """
    读取 数据对/*.json 或 *.jsonl，把每相邻 k 条组成一组（与之前生成 samples.jsonl 的逻辑一致）。
    若条目本身有 meta.group_size / group_index，仍按顺序凑满 k 个一组。
    """
    groups = []
    for p in sorted(glob.glob(os.path.join(pairs_dir, "*.json*"))):
        try:
            items = load_list_or_root(p)
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")
            continue
        buf = []
        for it in items:
            img = it.get("image", None)
            if not img:
                continue
            buf.append(it)
            if len(buf) == k:
                groups.append(buf); buf = []
        # 末尾不足 k 的丢弃（与之前一致）
    print(f"[INFO] collected {len(groups)} groups, K={k}")
    return groups

def infer_logid(sample: Dict) -> str:
    # 从 actions / wm_targets / image 的相对路径中提取 logXXXX
    for key in ("actions", "wm_targets", "image"):
        rel = sample.get(key, "")
        if isinstance(rel, str):
            m = re.search(r"(log\d+)", rel)
            if m: return m.group(1)
    # 有些 pairs 里只有文件名，用不到 logid，这里返回空串
    return ""

def resolve_image_abs(root: str, image_rel: str, logid: str) -> str:
    base = os.path.basename(image_rel or "")
    candidates = []
    # 已经带 IMAGES/logXXXX/ 的
    if image_rel and re.search(r"IMAGES/(log\d+)/", image_rel):
        candidates.append(os.path.join(root, image_rel))
    # 推断 logXXXX
    if logid and base:
        candidates.append(os.path.join(root, "IMAGES", logid, base))
    # 兜底
    if image_rel:
        candidates.append(os.path.join(root, image_rel))
    for p in dict.fromkeys(candidates).keys():
        if p and os.path.isfile(p):
            return p
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="数据根，例如 /root/autodl-tmp/数据")
    ap.add_argument("--pairs_dir", default="数据对", help="动作&meta 对的目录")
    ap.add_argument("--k", type=int, default=4, help="每组帧数 K")
    ap.add_argument("--m_per_frame", type=int, default=256, help="每帧 token 数(通常 16x16=256)")
    ap.add_argument("--vq_config", type=str, default="/root/autodl-tmp/vqgan/vqgan_imagenet_f16_16384.yaml")
    ap.add_argument("--vq_ckpt", type=str,   default="/root/autodl-tmp/vqgan/vqgan_imagenet_f16_16384.ckpt")
    ap.add_argument("--out_dir", type=str, default="wm_targets", help="相对 root 的输出目录")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    root = args.root
    P_PAIRS = os.path.join(root, args.pairs_dir)
    OUT_DIR = os.path.join(root, args.out_dir)
    os.makedirs(OUT_DIR, exist_ok=True)

    vq = VQTokenizer(args.vq_config, args.vq_ckpt, device=args.device)

    groups = walk_groups(P_PAIRS, k=args.k)

    n_ok, n_miss = 0, 0
    for g in groups:
        # 组首帧作为样本 stem
        img0_rel = g[0].get("image", "")
        stem = os.path.splitext(os.path.basename(img0_rel))[0]
        logid = infer_logid(g[0])

        # 逐帧取图并量化
        tokens_list = []
        for it in g:
            abs_img = resolve_image_abs(root, it.get("image",""), logid)
            if not abs_img:
                n_miss += 1
                tokens_list = []
                break
            with Image.open(abs_img) as im:
                idx = vq.encode_pil(im)      # np.int64, shape [Hc*Wc]
            tokens_list.append(idx)

        if not tokens_list:
            continue

        arr = np.concatenate(tokens_list, axis=0)  # [K*(Hc*Wc)]
        # 形状自检对齐
        expect = args.k * args.m_per_frame
        if arr.shape[0] > expect:
            arr = arr[:expect]
        elif arr.shape[0] < expect:
            pad = -100 * np.ones((expect - arr.shape[0],), dtype=np.int64)  # IGNORE_INDEX 补齐
            arr = np.concatenate([arr, pad], axis=0)

        # 保存到 wm_targets/logXXXX/frame_xxxxx.npy
        subdir = os.path.join(OUT_DIR, logid) if logid else OUT_DIR
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"{stem}.npy")
        np.save(out_path, arr)
        n_ok += 1

        if n_ok % 100 == 0:
            print(f"[OK {n_ok}] -> {out_path}")

    print(f"[DONE] wrote {n_ok} wm files; missing frames groups: {n_miss}")
    print(f"输出目录：{OUT_DIR}")
    print("提示：训练时把 --h_world 设成 K*m_per_frame（例如 K=4, m=256 -> 1024）")

if __name__ == "__main__":
    main()
