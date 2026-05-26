# core/scripts/vq_tokenize_hf.py
import os, json, argparse, re, sys
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

import torch
from torchvision import transforms as T

# 允许从仓库根目录: python -m core.scripts.vq_tokenize_hf
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------- utils ----------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def save_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def extract_logid(item: Dict[str, Any], image_key="image", actions_key="actions", wm_key="wm_targets") -> Optional[str]:
    for k in (actions_key, wm_key, image_key):
        rel = item.get(k, "")
        if not isinstance(rel, str) or not rel:
            continue
        m = re.search(r"(log\d+)", rel)
        if m:
            return m.group(1)
    return None

def resolve_image_abs(root: str, item: Dict[str, Any], image_key="image") -> str:
    """
    统一按以下顺序找图像：
      1) 如果 image 已是 IMAGES/logXXXX/frame_xxx.png → 直接拼 root
      2) 否则根据 actions/wm 推断 logXXXX，再到 IMAGES/logXXXX/ 里按 basename 找
      3) 最后兜底用 root + image 相对路径
    找不到返回空串
    """
    image_rel = (item.get(image_key, "") or "").strip()
    base = os.path.basename(image_rel) if image_rel else ""
    # A: 已带 logXXXX
    if image_rel and re.search(r"IMAGES/(log\d+)/", image_rel):
        p = os.path.join(root, image_rel)
        if os.path.isfile(p): return p
    # B: 由其他字段推断
    logid = extract_logid(item, image_key=image_key)
    if logid and base:
        p = os.path.join(root, "IMAGES", logid, base)
        if os.path.isfile(p): return p
    # C: 兜底
    if image_rel:
        p = os.path.join(root, image_rel)
        if os.path.isfile(p): return p
    return ""

def default_wm_rel(item: Dict[str, Any], out_dir="wm_targets", image_key="image") -> Optional[str]:
    """
    若 jsonl 没给 wm_targets 路径，按 IMAGES/logXXXX/frame_xxx.png → wm_targets/logXXXX/frame_xxx.npy 生成一个默认相对路径。
    找不到 log 或 basename 就返回 None。
    """
    image_rel = (item.get(image_key, "") or "").strip()
    base = os.path.splitext(os.path.basename(image_rel))[0]
    logid = extract_logid(item, image_key=image_key)
    if not base or not logid:
        return None
    return f"{out_dir}/{logid}/{base}.npy"

def build_preprocess(size: int):
    # PIL → [0,1] tensor → [-1,1]；VQ(VAE) 通常要求 [-1,1] 输入
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),            # [0,1]
        lambda x: x.mul(2.0).sub(1.0).clamp(-1.0, 1.0)
    ])

# ---------- main ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="数据根目录，例如 /root/autodl-tmp/数据")
    ap.add_argument("--jsonl", default="samples.jsonl", help="样本清单（相对 root）")
    ap.add_argument("--vq_path", default="/root/autodl-tmp/vqgan_hf", help="HF 模型目录（含 config.json + pytorch_model.bin）")
    ap.add_argument("--size", type=int, default=256, help="VQ 编码输入分辨率（通常 256）")
    ap.add_argument("--out_key", default="wm_targets", help="jsonl 中 wm 路径的键名")
    ap.add_argument("--image_key", default="image")
    ap.add_argument("--overwrite", action="store_true", help="已存在则覆盖")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（调试）")
    ap.add_argument("--write_back", action="store_true", help="若原 jsonl 缺失 wm_targets，则写回一个新 jsonl：<jsonl>.with_wm.jsonl")
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 载入 VQ 模型（优先 VQModel；不行再尝试 VQGanVAE）
    vq = None
    from diffusers import VQModel
    try:
        vq = VQModel.from_pretrained(args.vq_path)
    except Exception as e_vq:
        try:
            from diffusers import VQGanVAE
            vq = VQGanVAE.from_pretrained(args.vq_path)
        except Exception as e_vqgan:
            raise RuntimeError(f"无法加载 VQ 模型：{args.vq_path}\nVQModel 错误: {e_vq}\nVQGanVAE 错误: {e_vqgan}")
    vq.eval().to(device)

    if not hasattr(vq, "get_codebook_indices"):
        raise AttributeError("该 HF 模型未暴露 get_codebook_indices 方法，无法直接导出离散 token。请换一个带该方法的 VQ 模型，或让我给你一版'手动量化最近邻'的实现。")

    preprocess = build_preprocess(args.size)
    src = os.path.join(args.root, args.jsonl)
    items = list(load_jsonl(src))
    out_items = []

    os.makedirs(os.path.join(args.root, "wm_targets"), exist_ok=True)

    done = 0
    for it in items:
        if args.limit and done >= args.limit:
            out_items.append(it)
            continue

        abs_img = resolve_image_abs(args.root, it, image_key=args.image_key)
        if not abs_img:
            # 找不到图像：跳过，但保留原条目
            out_items.append(it)
            continue

        # 目标相对路径
        wm_rel = it.get(args.out_key)
        if not wm_rel or not isinstance(wm_rel, str) or not len(wm_rel.strip()):
            wm_rel = default_wm_rel(it, out_dir="wm_targets", image_key=args.image_key)

        if not wm_rel:
            # 无法推断输出路径：跳过
            out_items.append(it)
            continue

        out_path = os.path.join(args.root, wm_rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if (not args.overwrite) and os.path.isfile(out_path):
            # 已存在就复用
            if args.write_back:
                it[args.out_key] = wm_rel
                out_items.append(it)
            done += 1
            continue

        # 2) 读图并编码到离散 token
        with Image.open(abs_img) as im:
            im = im.convert("RGB")
        x = preprocess(im).unsqueeze(0).to(device)  # [1,3,H,W] in [-1,1]

        with torch.no_grad():
            idx = vq.get_codebook_indices(x)        # [1, h, w] 例如 [1,16,16]
        tokens = idx[0].flatten().to(torch.int64).cpu().numpy()  # 长度 = h*w，通常 256

        # 3) 保存
        np.save(out_path, tokens)

        # 写回 jsonl 的 wm_targets 字段（可选）
        if args.write_back:
            it[args.out_key] = wm_rel

        out_items.append(it)
        done += 1
        if done % 100 == 0:
            print(f"[{done}] {abs_img} -> {wm_rel} ({tokens.shape[0]} tokens)")

    # 若需要，把带有 wm_targets 的版本写回
    if args.write_back:
        dst = os.path.splitext(src)[0] + ".with_wm.jsonl"
        save_jsonl(dst, out_items)
        print(f"[OK] 写回 {dst}（原文件不覆盖）")
    else:
        print(f"[OK] 已生成 {done} 个 wm_targets（未改写 jsonl）")

if __name__ == "__main__":
    main()
