#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust inference script for OceanVLA (image+text -> discrete actions).
- Accepts single image, a list of images, wildcards, or directories.
- Works with CLIP AutoImageProcessor or CLIPImageProcessor.
- Tolerates checkpoint formats: raw state_dict, {"state_dict": ...}, {"model": ...}.
- Tolerates model.forward signature differences (filters kwargs by signature).
- Dynamically adapts class names if v_act != 6.
- Temporal aggregator is used automatically when multiple frames are provided.
- NEW:
  * --pick_step: show only one step (e.g. 0)
  * --aggregate: aggregate K steps into one prediction: 'mean' or 'vote'
  * Friendly notices about temporal mismatch.

Example:
  python -m core.scripts.infer_min_fixed \
    --ckpt checkpoints/oceanvla_step20000.pt \
    --image "/root/autodl-tmp/数据/IMAGES/log334/frame_01928.png" \
            "/root/autodl-tmp/数据/IMAGES/log334/frame_01929.png" \
            "/root/autodl-tmp/数据/IMAGES/log334/frame_01930.png" \
            "/root/autodl-tmp/数据/IMAGES/log334/frame_01931.png" \
    --text "The vessel is slowing down near the buoy." \
    --k_act 6 --a_dim 6 --h_world 32 --v_wm 8192 --v_act 6 \
    --precision bf16 --aggregate mean
"""

import os
import sys
import re
import glob
import json
import argparse
import inspect
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

# allow: python -m core.scripts.infer_min_fixed
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from transformers import AutoImageProcessor
    _HAS_AUTO_IMAGE = True
except Exception:
    _HAS_AUTO_IMAGE = False
try:
    from transformers import CLIPImageProcessor
    _HAS_CLIP_IMAGE = True
except Exception:
    _HAS_CLIP_IMAGE = False

from core.modeling.ocean_model import OceanVLA

# -----------------------
# Helpers
# -----------------------

def _filter_kwargs_for(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except (TypeError, ValueError):
        return dict(kwargs)

def _natural_key(p: str):
    base = os.path.basename(p)
    parts = re.split(r"(\d+)", base)
    return [int(x) if x.isdigit() else x.lower() for x in parts]

def _expand_image_inputs(inputs: List[str]) -> List[str]:
    out: List[str] = []
    for s in inputs:
        if any(ch in s for ch in "*?[]"):
            out.extend(glob.glob(s))
            continue
        if os.path.isdir(s):
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
                out.extend(glob.glob(os.path.join(s, ext)))
            continue
        if os.path.isfile(s):
            out.append(s)
    out = sorted(list(dict.fromkeys(out)), key=_natural_key)
    if len(out) == 0:
        raise FileNotFoundError("No images found from --image / --image_glob / directories.")
    return out

def _load_image_processor(clip_path: str):
    last_err = None
    if _HAS_AUTO_IMAGE:
        try:
            return AutoImageProcessor.from_pretrained(clip_path)
        except Exception as e:
            last_err = e
    if _HAS_CLIP_IMAGE:
        try:
            return CLIPImageProcessor.from_pretrained(clip_path)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load image processor from {clip_path}: {last_err}")

def load_images_tensor(paths: Union[str, List[str]], clip_path: str) -> torch.Tensor:
    proc = _load_image_processor(clip_path)
    def _one(p):
        with Image.open(p) as im:
            im = im.convert("RGB")
            return proc(images=im, return_tensors="pt")["pixel_values"][0]
    if isinstance(paths, str):
        return _one(paths).unsqueeze(0)          # [1,3,H,W]
    frames = [_one(p) for p in paths]
    px = torch.stack(frames, dim=0)              # [T,3,H,W]
    return px.unsqueeze(0)                       # [1,T,3,H,W]

def build_tokenizer(path: str):
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok

# -----------------------
# Config
# -----------------------

def make_default_cfg(args, T: int = 1):
    H = 1024
    from types import SimpleNamespace as NS
    return NS(
        backbone=NS(
            name_or_path="__ignored__",
            hidden_size=H,
            freeze_layers=0,
            gradient_checkpointing=True,
            torch_dtype="float32",
            device_map=None,
        ),
        vision=NS(
            vision_model=args.clip_path,
            pool="projected",
            embed_dim=H,
            normalize=False,
        ),
        action=NS(
            input_dim=args.a_dim,
            chunk_size=args.k_act,
            hidden_size=H,
            discrete_dim=args.v_act,
            num_classes=args.v_act,
            # 不开启不平衡技巧；推理不需要
            use_class_weights=False,
            use_la=False,
            use_focal=False,
            la_tau=1.0,
            focal_gamma=2.0,
        ),
        world_model=NS(
            hidden_size=H,
            latent_dim=H,
            lstm_hidden_size=H,
            seq_len=args.h_world,
            vocab_size=args.v_wm,
            backbone_hidden=H,
            temperature=1.0,
            label_smoothing=0.0,
        ),
        temporal=NS(
            enabled=bool(T > 1),     # 只要你传了多帧就启用
            num_frames=int(T),
            agg="gru",
            hidden_size=H,
            dropout=0.1,
            use_prev_action=False,   # 你训练阶段没喂 prev actions
        ),
        vocab=NS(action=args.v_act),
    )

# -----------------------
# Model I/O helpers
# -----------------------

def _maybe_extract_state_dict(sd: Any):
    if isinstance(sd, dict):
        for k in ("state_dict", "model", "ema_state_dict", "module"):
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd

def _find_action_logits(out: Dict[str, Any]) -> Optional[torch.Tensor]:
    cand_keys = [
        "action_logits", "logits_action", "act_logits", "logits_actions",
        "actions_logits", "action_logit", "act_logit",
    ]
    if isinstance(out, dict):
        for k in cand_keys:
            v = out.get(k, None)
            if torch.is_tensor(v):
                return v
        for k in ["action", "actions", "act"]:
            v = out.get(k, None)
            if isinstance(v, dict):
                for kk in ["logits", "logit", "pred", "scores"]:
                    t = v.get(kk, None)
                    if torch.is_tensor(t):
                        return t
    if isinstance(out, (list, tuple)):
        for x in out:
            if torch.is_tensor(x) and x.ndim >= 2:
                return x
    return None

# -----------------------
# CLI
# -----------------------

def parse_args():
    p = argparse.ArgumentParser("OceanVLA minimal inference")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--image", nargs="+", required=True)
    p.add_argument("--image_glob", type=str, default="")
    p.add_argument("--text", type=str, default="")
    # tokenizer / clip
    p.add_argument("--tokenizer_path", type=str, default="/root/autodl-tmp/models--google--gemma-2b")
    p.add_argument("--clip_path", type=str, default="/root/autodl-tmp/models--openai--clip-vit-base-patch32")
    # structure（需与训练一致）
    p.add_argument("--k_act", type=int, default=6)
    p.add_argument("--a_dim", type=int, default=6)
    p.add_argument("--v_act", type=int, default=6)
    p.add_argument("--h_world", type=int, default=32)
    p.add_argument("--v_wm", type=int, default=8192)
    # AMP / output
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--output_json", type=str, default="")
    # NEW viewing options
    p.add_argument("--pick_step", type=int, default=-1, help="Only show this step (0..K-1). -1 shows all.")
    p.add_argument("--aggregate", type=str, default="none", choices=["none", "mean", "vote"],
                   help="Aggregate K steps to one prediction.")
    return p.parse_args()

# -----------------------
# Main
# -----------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # images
    img_list = _expand_image_inputs(args.image)
    if args.image_glob:
        img_list += _expand_image_inputs([args.image_glob])
        img_list = sorted(list(dict.fromkeys(img_list)), key=_natural_key)
    img_tensor = load_images_tensor(img_list if len(img_list) > 1 else img_list[0], args.clip_path).to(device)
    T = img_tensor.shape[1] if img_tensor.ndim == 5 else 1
    if T == 1:
        print("[note] You provided 1 frame. If you trained with --temporal_enabled and multiple frames, "
              "consider passing the last few frames to match train-time setup.")

    # text
    tok = build_tokenizer(args.tokenizer_path)
    enc = tok(args.text or "", add_special_tokens=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # model
    cfg = make_default_cfg(args, T=T)
    model = OceanVLA(cfg).to(device).eval()

    # load weights
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = _maybe_extract_state_dict(sd)
    incompatible = model.load_state_dict(sd, strict=False)
    print(f"[load] ckpt={args.ckpt}, missing={len(getattr(incompatible,'missing_keys',[]))}, "
          f"unexpected={len(getattr(incompatible,'unexpected_keys',[]))}")

    # actions: 训练时未用连续动作监督，这里按全零占位即可（模型仍然依赖图像/文本）
    actions = torch.zeros((1, args.k_act, args.a_dim), dtype=torch.float32, device=device)

    # dummy supervision
    act_tgt = torch.full((1, args.k_act), -100, dtype=torch.long, device=device)
    wm_tgt  = torch.full((1, args.h_world), -100, dtype=torch.long, device=device)

    # AMP
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    amp_dtype = dtype_map[args.precision]
    use_amp = (device == "cuda" and args.precision != "fp32")

    # forward
    model_inputs = dict(
        input_ids=input_ids,
        text_attention_mask=attn_mask,
        images=img_tensor,
        actions=actions,
        action_targets=act_tgt,
        wm_targets=wm_tgt,
        lambda_action=0.0,
        lambda_world=0.0,
        lambda_text=0.0,
    )
    model_inputs = _filter_kwargs_for(model.forward, model_inputs)

    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(**model_inputs)
        else:
            out = model(**model_inputs)

    act_logits = _find_action_logits(out)
    if act_logits is None:
        keys = list(out.keys()) if isinstance(out, dict) else type(out)
        raise RuntimeError(f"未找到动作 logits，输出键：{keys}")

    if act_logits.ndim == 2:
        act_logits = act_logits.unsqueeze(1)   # [B,1,V]
    probs = act_logits.softmax(dim=-1).squeeze(0).detach().float().cpu()  # [K,V]
    K, V = probs.shape
    topk = max(1, min(args.topk, V))

    # class names
    if args.v_act == 6:
        cls_names = ["forward", "accelerate", "decelerate", "turn_left", "turn_right", "stop"]
    else:
        cls_names = [f"class_{i}" for i in range(V)]

    n_frames = img_tensor.shape[1] if img_tensor.ndim == 5 else 1
    print(f"\n=== Action prediction (frames={n_frames}, temporal={'ON' if (cfg.temporal.enabled) else 'OFF'}) ===")

    # optional pick_step
    show_steps = range(K) if args.pick_step < 0 else [max(0, min(args.pick_step, K-1))]
    steps_out = []
    for t in show_steps:
        pv = probs[t]
        vals, idx = torch.topk(pv, k=topk, dim=-1)
        vals = vals.tolist(); idx = idx.tolist()
        if topk == 1:
            print(f" step {t:02d}: {cls_names[idx[0]]}  (p={vals[0]:.4f})")
        else:
            pretty = ", ".join([f"{cls_names[i]}:{v:.4f}" for i, v in zip(idx, vals)])
            print(f" step {t:02d}: {pretty}")
        steps_out.append({
            "t": t,
            "argmax_id": idx[0],
            "argmax_name": cls_names[idx[0]],
            "argmax_p": float(vals[0]),
            "topk": [{"id": int(i), "name": cls_names[i], "p": float(v)} for i, v in zip(idx, vals)]
        })

    # aggregate if requested
    if args.aggregate != "none":
        if args.aggregate == "mean":
            agg_p = probs.mean(dim=0)            # [V]
            j = int(torch.argmax(agg_p).item())
            print(f"\n[aggregate=mean] -> {cls_names[j]}  (p={float(agg_p[j]):.4f})")
        elif args.aggregate == "vote":
            votes = probs.argmax(dim=-1).numpy().tolist()
            j = int(np.bincount(votes, minlength=V).argmax())
            print(f"\n[aggregate=vote] -> {cls_names[j]}  (votes={votes.count(j)}/{K})")

    # also print raw argmax per step if showing all
    if args.pick_step < 0:
        pred_ids = probs.argmax(dim=-1).tolist()
        print("\nargmax ids:", pred_ids)
        print("argmax names:", [cls_names[i] for i in pred_ids])

    # optional save
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        payload = {
            "images": img_list,
            "text": args.text,
            "k_act": args.k_act,
            "a_dim": args.a_dim,
            "classes": cls_names,
            "steps": steps_out,
            "temporal_enabled": bool(cfg.temporal.enabled),
            "num_frames": int(cfg.temporal.num_frames),
            "aggregate": args.aggregate,
            "pick_step": args.pick_step,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[saved] {args.output_json}")

if __name__ == "__main__":
    main()
