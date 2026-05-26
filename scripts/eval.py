#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight evaluator for OceanVLA (vision → world/action), no training code changes required.

Key improvements:
- ✅ Apply-time Logit-Adjusted inference (--apply_la_infer / --la_tau_infer)
- ✅ Respect --lambda_action / --lambda_world / --lambda_text
- ✅ Robust Dataset arg names (v_world vs v_wm), signature-safe kwargs
- ✅ Optional ablations: --zero_actions / --no_action_tokens / --shuffle_images
- ✅ Fallback world loss: compute CE on (world_logits, wm_targets) when model doesn't return it
- ✅ Safer device/dtype (AMP optional), TF32 enabled, memory-friendly
- ✅ Actions disabled path: actions_key="__disable_actions__" → model receives actions=None
- ✅ NEW: --emit_predictions 输出逐样本 JSON（前缀 "JSON_RESULT "），便于在线服务解析
"""

import os
import math
import argparse
import inspect
import warnings
import json  # NEW: for JSON_RESULT lines
from types import SimpleNamespace as NS
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Perf knobs (safe on A100/4090)
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# 项目内导入（请从仓库根目录运行：python -m core.scripts.eval_min）
try:
    from core.modeling.ocean_model import OceanVLA
    from core.data.datasets.ocean_dataset import OceanVLADataset
    from core.data.datasets.collate import collate_fn
except Exception as e:
    raise RuntimeError(
        "Failed to import project modules. Run from repo root, e.g.:\n"
        "  python -m core.scripts.eval_min ..."
    ) from e


# ----------------------------- utils ----------------------------- #
IGNORE_INDEX = -100

def _parse_hw_tuple(s: Optional[str]):
    """Parse 'H,W' -> (H, W). Accepts tuple/list or string."""
    if s is None:
        return None
    if isinstance(s, (tuple, list)) and len(s) == 2:
        return int(s[0]), int(s[1])
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid hw tuple: {s} (expect 'H,W')")
    return int(parts[0]), int(parts[1])


def _filter_kwargs_for(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs that are present in callable's signature."""
    try:
        sig = inspect.signature(callable_obj)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except (TypeError, ValueError):
        # Fallback: return all
        return dict(kwargs)


def _get_world_loss_from_out(out: Dict[str, Any]) -> Optional[float]:
    """Try multiple common keys to fetch world-model loss (sum over tokens)."""
    for k in ("loss_w", "world_loss", "loss_world", "loss_wm", "wm_loss"):
        if k in out and out[k] is not None:
            try:
                return float(out[k])
            except Exception:
                pass
    if isinstance(out.get("losses"), dict):
        for k in ("world", "wm", "loss_w", "loss_world"):
            if k in out["losses"]:
                try:
                    return float(out["losses"][k])
                except Exception:
                    pass
    return None


def _get_action_logits(out: Dict[str, Any]) -> Optional[torch.Tensor]:
    """Best-effort 获取动作 logits。"""
    for k in ("action_logits", "logits_action", "act_logits", "logits"):
        v = out.get(k, None)
        if isinstance(v, torch.Tensor):
            return v
    # 兜底：有些实现会把动作部分包在 dict 里
    cand = out.get("action", out.get("actions", None))
    if isinstance(cand, dict):
        for kk in ("logits", "scores", "logit"):
            v = cand.get(kk, None)
            if isinstance(v, torch.Tensor):
                return v
    return None


def _sig_params(callable_obj):
    try:
        return set(inspect.signature(callable_obj).parameters.keys())
    except Exception:
        return set()


def _masked_cross_entropy(logits: torch.Tensor,
                          targets: torch.Tensor,
                          ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """
    CE(logits, targets) averaged over valid (targets != ignore_index).
    Supports [B,N,V] vs [B,N] or [N,V] vs [N].
    """
    assert logits.ndim in (2, 3)
    assert targets.ndim in (1, 2)

    if logits.ndim == 3:
        B, N, V = logits.shape
        logits = logits.reshape(B * N, V)
        targets = targets.reshape(B * N)
    else:
        V = logits.size(-1)

    targets = targets.long()
    mask = (targets != ignore_index)
    if int(mask.sum()) == 0:
        return logits.sum() * 0.0

    loss = F.cross_entropy(logits[mask], targets[mask], reduction="mean")
    return loss


def _get_action_log_prior(model: torch.nn.Module,
                          dataset_prior: Optional[list],
                          device,
                          dtype) -> Optional[torch.Tensor]:
    """
    Get log-prior for actions:
    1) prefer model.act_log_prior buffer (already log-space)
    2) else use dataset._class_prior (prob-space), take log
    """
    if hasattr(model, "act_log_prior") and (getattr(model, "act_log_prior") is not None):
        t = model.act_log_prior.to(device=device, dtype=dtype).view(1, 1, -1)  # [1,1,C]
        return t
    if dataset_prior is not None:
        prior = torch.tensor(dataset_prior, device=device, dtype=dtype).clamp_min(1e-8)
        return prior.log().view(1, 1, -1)
    return None


# ----------------------------- builders ----------------------------- #
def build_tokenizer(args):
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def build_loader(args, tokenizer):
    # 动态识别 Dataset 的参数命名（v_world vs v_wm 等）
    ds_param_names = _sig_params(OceanVLADataset.__init__)

    ds_kwargs = dict(
        root=args.data_root,
        jsonl=args.jsonl,
        tokenizer=tokenizer,
        img_size=args.img_size,
        # —— 动作/世界 —— #
        k=args.k_act,
        a_dim=args.a_dim,
        h_world=args.h_world,
        use_random_wm=False,
        # —— 字段映射 —— #
        image_key=args.image_key,
        text_key=args.text_key,
        actions_key=args.actions_key,
        wm_key=args.wm_key,
        # —— CLIP 预处理 —— #
        use_clip_processor=args.use_clip_processor,
        clip_path=args.clip_path,
        # —— 时序参数 —— #
        temporal_enabled=args.temporal_enabled,
        num_frames=args.num_frames,
        stride=args.stride,
        frame_regex=args.frame_regex,
        # —— world token 下采样 —— #
        wm_src_hw=args.wm_src_hw if args.wm_src_hw else (64, 64),
        wm_tgt_hw=args.wm_tgt_hw if args.wm_tgt_hw else (8, 4),
        wm_pool=args.wm_pool,
        # —— 离散动作 6 类外部标签 —— #
        action6_key=getattr(args, "action6_key", "action_targets_npy"),
        action6_root=(getattr(args, "action6_root", None) or args.data_root),
        action6_template=getattr(args, "action6_template", None),
        logid_key=getattr(args, "logid_key", "meta.log_id"),
        group_index_key=getattr(args, "group_index_key", "meta.group_index"),
        group_size_key=getattr(args, "group_size_key", "meta.group_size"),
    )

    # 兼容数据集把 vocab 维度命名为 v_world 或 v_wm
    if "v_world" in ds_param_names:
        ds_kwargs["v_world"] = args.v_wm
    elif "v_wm" in ds_param_names:
        ds_kwargs["v_wm"] = args.v_wm

    ds = OceanVLADataset(**_filter_kwargs_for(OceanVLADataset.__init__, ds_kwargs))

    # Collate kwargs filtered by signature (避免 collate_fn 不支持的参数报错)
    def _collate(batch):
        kw = dict(
            batch=batch,
            tokenizer=tokenizer,
            k_act=args.k_act,
            h_action=args.h_action,
            h_world=args.h_world,
            add_image_token=True,
        )
        return collate_fn(**_filter_kwargs_for(collate_fn, kw))

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate,
    )
    return ds, loader


def build_model_cfg(args, H=1024):
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
    )


# ----------------------------- main eval ----------------------------- #
def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    # ---- normalize tuple args (argparse 不会对默认值执行 type) ---- #
    args.wm_src_hw = _parse_hw_tuple(args.wm_src_hw) if isinstance(args.wm_src_hw, str) else args.wm_src_hw
    args.wm_tgt_hw = _parse_hw_tuple(args.wm_tgt_hw) if isinstance(args.wm_tgt_hw, str) else args.wm_tgt_hw

    # Tokenizer & data
    tok = build_tokenizer(args)
    ds, loader = build_loader(args, tok)

    # Model
    cfg = build_model_cfg(args)
    model = OceanVLA(cfg).to(device).eval()

    sd = torch.load(args.ckpt, map_location="cpu")
    # 兼容某些保存格式
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    incompatible = model.load_state_dict(sd, strict=False)
    try:
        missing = len(getattr(incompatible, "missing_keys", []))
        unexpected = len(getattr(incompatible, "unexpected_keys", []))
    except Exception:
        # 旧版 PyTorch 直接返回列表
        missing = unexpected = 0
    print(f"[eval] loaded ckpt: {args.ckpt}, missing={missing}, unexpected={unexpected}")

    # Metrics
    tot_w_sum = 0.0
    tot_valid_w = 0
    tot_a_correct = 0
    tot_a_valid = 0

    # Optional per-class stats / confusion
    if args.per_class:
        C = int(args.v_act)
        conf = np.zeros((C, C), dtype=int)  # [gt, pred]
        per_class_correct = np.zeros(C, dtype=int)
        per_class_total = np.zeros(C, dtype=int)
    else:
        conf = per_class_correct = per_class_total = None

    n_batches = 0

    # ====== LA-infer 准备：从模型 buffer 或数据集统计拿先验 ======
    la_log_prior = None  # [1,1,C] in log space
    # 延迟初始化：第一次拿到 action_logits 再创建（知道 dtype）

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    autocast_enabled = bool(args.amp)

    with torch.inference_mode():
        for batch in loader:
            # Move to device
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            # Optional ablations
            if args.zero_actions and "actions" in batch and isinstance(batch["actions"], torch.Tensor):
                batch["actions"] = torch.zeros_like(batch["actions"])
            if args.shuffle_images and "images" in batch and isinstance(batch["images"], torch.Tensor):
                idx = torch.randperm(batch["images"].shape[0], device=device)
                batch["images"] = batch["images"][idx]

            # Prepare model inputs (健壮键名)
            txt_mask = batch.get("text_attention_mask", batch.get("attention_mask", None))
            actions_for_model = batch.get("actions", None)
            # 当 actions_key="__disable_actions__" 时，数据集不会提供连续动作，这里显式传 None
            if args.actions_key == "__disable_actions__":
                actions_for_model = None
            if args.no_action_tokens:
                actions_for_model = None

            model_in = dict(
                input_ids=batch.get("input_ids", None),
                text_attention_mask=txt_mask,
                images=batch.get("images", batch.get("image", None)),
                actions=actions_for_model,
                action_targets=batch.get("action_targets", None),
                wm_targets=batch.get("wm_targets", None),
                actions_prev=batch.get("actions_prev", None),
                # ✅ 使用命令行的 lambda，而不是写死
                lambda_action=float(args.lambda_action),
                lambda_world=float(args.lambda_world),
                lambda_text=float(args.lambda_text),
            )
            model_in = _filter_kwargs_for(model.forward, model_in)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=autocast_enabled):
                out = model(**model_in)

            # ===== World loss =====
            lw_sum = _get_world_loss_from_out(out)
            # Fallback: 若模型没返回 world loss，直接用 logits + targets 算（忽略 label smoothing/温度细节）
            if lw_sum is None:
                wlog = out.get("world_logits", None)
                wtar = batch.get("wm_targets", None)
                if isinstance(wlog, torch.Tensor) and isinstance(wtar, torch.Tensor):
                    ce = _masked_cross_entropy(wlog.float(), wtar)
                    lw_sum = float(ce.item()) * int((wtar != IGNORE_INDEX).sum().item())  # 还原为“sum over tokens”
            if lw_sum is not None:
                valid_w = int((batch.get("wm_targets", torch.empty(0)) != IGNORE_INDEX).sum().item()) \
                          if isinstance(batch.get("wm_targets"), torch.Tensor) else 0
                tot_w_sum += float(lw_sum)
                tot_valid_w += int(valid_w)

            # ===== Action acc & predictions =====
            logits = _get_action_logits(out)
            gt = batch.get("action_targets", None)

            # --- 评测端 LA 矫正（避免“全猜类0”） ---
            pred = None
            if isinstance(logits, torch.Tensor):
                if args.apply_la_infer:
                    if la_log_prior is None:
                        la_log_prior = _get_action_log_prior(
                            model,
                            getattr(getattr(loader, "dataset", None), "_class_prior", None),
                            device=logits.device,
                            dtype=logits.dtype,
                        )
                    if la_log_prior is not None:
                        tau = float(args.la_tau_infer)
                        logits = logits - tau * la_log_prior  # [B,K,C] 支持
                pred = logits.argmax(-1)  # [B,K] 或 [B]

            # —— 累计准确率（当 gt 存在时）——
            if isinstance(pred, torch.Tensor) and isinstance(gt, torch.Tensor):
                mask = (gt != IGNORE_INDEX)
                # 兼容 [B,K] / [B]
                if pred.ndim == 2 and gt.ndim == 2 and pred.shape[:2] == gt.shape[:2]:
                    tot_a_correct += (pred.eq(gt) & mask).sum().item()
                    tot_a_valid += mask.sum().item()
                elif pred.ndim == 2 and gt.ndim == 1:
                    # 取 K=1 的场景（或只对第一个 token 评测）
                    tot_a_correct += (pred[:, 0].eq(gt) & (gt != IGNORE_INDEX)).sum().item()
                    tot_a_valid += int((gt != IGNORE_INDEX).sum().item())
                elif pred.ndim == 1 and gt.ndim == 1:
                    tot_a_correct += (pred.eq(gt) & (gt != IGNORE_INDEX)).sum().item()
                    tot_a_valid += int((gt != IGNORE_INDEX).sum().item())

                if args.per_class and conf is not None:
                    # 展开为标量对
                    if pred.ndim == 2:  # [B,K]
                        g = gt.view(-1)
                        p = pred.view(-1)
                        m = (gt != IGNORE_INDEX).view(-1)
                    else:  # [B]
                        g = gt.view(-1)
                        p = pred.view(-1)
                        m = (gt != IGNORE_INDEX).view(-1)
                    for gi, pi, mi in zip(g.tolist(), p.tolist(), m.tolist()):
                        if not mi:
                            continue
                        if 0 <= gi < conf.shape[0] and 0 <= pi < conf.shape[1]:
                            conf[gi, pi] += 1
                            per_class_total[gi] += 1
                            if gi == pi:
                                per_class_correct[gi] += 1

            # —— 逐样本预测输出（供在线服务解析）——
            if args.emit_predictions:
                # 尝试从 batch 中拿图像路径（不同实现可能字段名不同）
                img_paths = batch.get("paths", batch.get("image_paths", batch.get("images_paths", None)))

                # 统一成 [B] 的 top1 预测（优先取第一个 token）
                top1_vec = None
                if isinstance(pred, torch.Tensor):
                    if pred.ndim == 2:
                        # [B,K] -> 取第一个 token
                        if pred.size(1) > 0:
                            top1_vec = pred[:, 0]
                        else:
                            top1_vec = pred.view(pred.size(0))
                    else:
                        top1_vec = pred

                # 推断 batch 大小 B
                if isinstance(top1_vec, torch.Tensor):
                    B = top1_vec.size(0)
                elif isinstance(gt, torch.Tensor):
                    B = gt.size(0)
                elif isinstance(img_paths, (list, tuple)):
                    B = len(img_paths)
                else:
                    B = 1

                for b in range(B):
                    item = {"index": int(b)}
                    # 预测
                    if isinstance(top1_vec, torch.Tensor) and b < top1_vec.size(0):
                        try:
                            item["top1"] = int(top1_vec[b].item())
                        except Exception:
                            pass
                    # 标注（可无）
                    if isinstance(gt, torch.Tensor) and b < gt.size(0):
                        try:
                            if gt.ndim == 2 and gt.size(1) > 0:
                                item["gt"] = int(gt[b, 0].item()) if int(gt[b, 0].item()) != IGNORE_INDEX else None
                            else:
                                item["gt"] = int(gt[b].item()) if int(gt[b].item()) != IGNORE_INDEX else None
                        except Exception:
                            pass
                    # 路径（可无）
                    if isinstance(img_paths, (list, tuple)) and b < len(img_paths):
                        pth = img_paths[b]
                        if isinstance(pth, (list, tuple)) and len(pth) > 0:
                            pth = pth[0]
                        item["image"] = str(pth)

                    print("JSON_RESULT " + json.dumps(item, ensure_ascii=False), flush=True)

            n_batches += 1
            if args.num_batches > 0 and n_batches >= args.num_batches:
                break

            # 适度清理显存碎片
            if device == "cuda" and (n_batches % 50 == 0):
                torch.cuda.empty_cache()

    if tot_valid_w > 0:
        w_tok = tot_w_sum / max(1, tot_valid_w)
        ppl = math.exp(w_tok)
        print(f"[eval] items={len(ds)}, batches={n_batches}, world_token_nats={w_tok:.4f}, world_ppl={ppl:.3f}")
    else:
        print(f"[eval] items={len(ds)}, batches={n_batches}, world_loss: N/A (no valid tokens)")

    if tot_a_valid > 0:
        acc = 100.0 * tot_a_correct / max(1, tot_a_valid)
        print(f"[eval] action_top1={acc:.2f}%  (valid={tot_a_valid})")
    else:
        print("[eval] action logits not exposed by model or no valid labels; 只统计了 world loss。")

    if args.per_class and per_class_total is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_acc = np.divide(per_class_correct, np.maximum(1, per_class_total))
        print("[eval] per-class acc (%):", np.round(per_class_acc * 100, 2).tolist())
        print("[eval] confusion matrix [gt, pred]:\n", conf)


# ----------------------------- argparse ----------------------------- #
def parse_args():
    p = argparse.ArgumentParser("OceanVLA quick evaluator")
    # Data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=0)
    # Tokenizer & CLIP
    p.add_argument("--tokenizer_path", type=str, default="/root/autodl-tmp/models--google--gemma-2b")
    p.add_argument("--use_clip_processor", action="store_true")
    p.add_argument("--clip_path", type=str, default="/root/autodl-tmp/models--openai--clip-vit-base-patch32")
    # Shapes / vocab
    p.add_argument("--k_act", type=int, default=6)
    p.add_argument("--a_dim", type=int, default=6)
    p.add_argument("--h_action", type=int, default=6)  # reserved by collate
    p.add_argument("--h_world", type=int, default=32)
    p.add_argument("--v_act", type=int, default=6)
    p.add_argument("--v_wm", type=int, default=8192)
    # Collate mapping
    p.add_argument("--image_key", type=str, default="image")
    p.add_argument("--text_key", type=str, default="text")
    p.add_argument("--actions_key", type=str, default="actions")
    p.add_argument("--wm_key", type=str, default="wm_targets")
    # Temporal
    p.add_argument("--temporal_enabled", action="store_true")
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--frame_regex", type=str, default=r"frame[_\-]?(\d+)")
    # World downsample (默认 None，稍后统一解析为 tuple)
    p.add_argument("--wm_src_hw", default=None)
    p.add_argument("--wm_tgt_hw", default=None)
    p.add_argument("--wm_pool", type=str, default="mode", choices=["mode", "center"])
    # Batch / loop
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_batches", type=int, default=100, help="Max number of batches to evaluate; set <=0 for all")
    # Checkpoint
    p.add_argument("--ckpt", type=str, required=True, help="Path to model state_dict (.pt)")
    # Loss weights
    p.add_argument("--lambda_action", type=float, default=1.0, help="Weight for action loss (if computed)")
    p.add_argument("--lambda_world", type=float, default=1.5, help="Weight for world loss (if computed)")
    p.add_argument("--lambda_text", type=float, default=0.0, help="Weight for text loss (if computed)")
    # Ablations
    p.add_argument("--zero_actions", action="store_true", help="Zero the action inputs to test pure visual drive")
    p.add_argument("--no_action_tokens", action="store_true", help="Drop action tokens (if model supports actions=None)")
    p.add_argument("--shuffle_images", action="store_true", help="Shuffle images across batch dimension")
    # Reporting
    p.add_argument("--per_class", action="store_true", help="Report per-class accuracy & confusion matrix")
    # Inference-time LA correction
    p.add_argument("--apply_la_infer", action="store_true",
                   help="Apply logit-adjusted correction at inference (mitigate prior collapse)")
    p.add_argument("--la_tau_infer", type=float, default=1.0,
                   help="Temperature τ for inference-time LA (logits -= τ*log_prior)")
    # AMP for eval
    p.add_argument("--amp", action="store_true", help="Enable autocast (CUDA) for evaluation speed/memory")
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    # NEW: emit per-sample JSON predictions
    p.add_argument("--emit_predictions", action="store_true",
                   help="Emit per-sample JSON lines (prefix 'JSON_RESULT ') for online service parsing")
    return p.parse_args()


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
