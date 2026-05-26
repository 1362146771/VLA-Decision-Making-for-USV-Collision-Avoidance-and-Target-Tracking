#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OceanVLA – robust evaluator with top-1/top-3/top-5 accuracy
"""

import os
import math
import argparse
import inspect
import warnings
from types import SimpleNamespace as NS
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

IGNORE_INDEX = -100

# TF32 perf (safe on Ampere+)
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Project imports (run from repo root)
try:
    from core.modeling.ocean_model import OceanVLA
    from core.data.datasets.ocean_dataset import OceanVLADataset
    from core.data.datasets.collate import collate_fn
except Exception as e:
    raise RuntimeError(
        "Failed to import project modules. Run from repo root, e.g.:\n"
        "  python -m core.scripts.eval_min ..."
    ) from e

# ----------------------------- helpers ----------------------------- #

def _parse_hw_tuple(s: Optional[str]):
    if s is None:
        return None
    if isinstance(s, (tuple, list)) and len(s) == 2:
        return int(s[0]), int(s[1])
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid hw tuple: {s} (expect 'H,W')")
    return int(parts[0]), int(parts[1])


def _sig_params(callable_obj):
    try:
        return set(inspect.signature(callable_obj).parameters.keys())
    except Exception:
        return set()


def _filter_kwargs_for(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except (TypeError, ValueError):
        return dict(kwargs)


def _masked_cross_entropy(logits: torch.Tensor,
                          targets: torch.Tensor,
                          ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """CE(logits, targets) on valid indices only. Supports [B,N,V]/[B,N] or [N,V]/[N]."""
    assert logits.ndim in (2, 3)
    assert targets.ndim in (1, 2)
    if logits.ndim == 3:
        B, N, V = logits.shape
        logits = logits.reshape(B * N, V)
        targets = targets.reshape(B * N)
    targets = targets.long()
    mask = (targets != ignore_index)
    if int(mask.sum()) == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], targets[mask], reduction="mean")


def _topk_accuracy(logits: torch.Tensor, 
                   targets: torch.Tensor, 
                   k: int = 1,
                   ignore_index: int = IGNORE_INDEX) -> Tuple[int, int]:
    """
    计算 top-k 准确率
    返回: (correct_count, valid_count)
    """
    if logits is None or targets is None:
        return 0, 0
    
    # 展平
    if logits.ndim == 3:
        B, N, V = logits.shape
        logits = logits.reshape(B * N, V)
        targets = targets.reshape(B * N)
    
    targets = targets.long()
    mask = (targets != ignore_index)
    
    if mask.sum() == 0:
        return 0, 0
    
    valid_logits = logits[mask]  # [M, V]
    valid_targets = targets[mask]  # [M]
    
    # 获取 top-k 预测
    _, topk_pred = valid_logits.topk(k=min(k, valid_logits.shape[-1]), dim=-1)  # [M, k]
    
    # 检查真实标签是否在 top-k 中
    correct = topk_pred.eq(valid_targets.unsqueeze(-1)).any(dim=-1)
    
    return int(correct.sum().item()), int(mask.sum().item())


def _get_world_loss_from_out(out: Dict[str, Any]) -> Optional[float]:
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
    for k in ("action_logits", "logits_action", "act_logits", "logits"):
        v = out.get(k, None)
        if isinstance(v, torch.Tensor):
            return v
    cand = out.get("action", out.get("actions", None))
    if isinstance(cand, dict):
        for kk in ("logits", "scores", "logit"):
            v = cand.get(kk, None)
            if isinstance(v, torch.Tensor):
                return v
    return None


def _logits_stats(t: torch.Tensor) -> Tuple[float, float, float]:
    """(mean, std, mean row-variance) for [B,K] or [...,K]"""
    if t is None or not isinstance(t, torch.Tensor):
        return (float("nan"), float("nan"), float("nan"))
    x = t.detach().float()
    m = float(x.mean().item())
    s = float(x.std().item())
    rowvar = float(x.var(dim=-1).mean().item()) if x.ndim >= 2 else float("nan")
    return m, s, rowvar


def _safe_log_prior(prior: Optional[list], K: int, peak_thr: float = 0.9) -> Optional[list]:
    if not prior:
        return None
    if len(prior) != K:
        warnings.warn(f"[prior] length mismatch: got {len(prior)} need {K} -> ignore")
        return None
    s = sum(prior)
    if s <= 0:
        return None
    prior = [max(0.0, float(p)) / s for p in prior]
    if max(prior) > peak_thr:
        warnings.warn(f"[prior] too peaked {prior} -> using uniform prior")
        return [1.0 / K] * K
    return prior


# ----------------------------- builders ----------------------------- #

def build_tokenizer(args):
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def build_loader(args, tokenizer):
    ds_param_names = _sig_params(OceanVLADataset.__init__)

    ds_kwargs = dict(
        root=args.data_root,
        jsonl=args.jsonl,
        tokenizer=tokenizer,
        img_size=args.img_size,
        # action/world
        k=args.k_act,
        a_dim=args.a_dim,
        h_world=args.h_world,
        use_random_wm=False,
        # field mapping
        image_key=args.image_key,
        text_key=args.text_key,
        actions_key=args.actions_key,
        wm_key=args.wm_key,
        # CLIP preprocess
        use_clip_processor=args.use_clip_processor,
        clip_path=args.clip_path,
        # temporal
        temporal_enabled=args.temporal_enabled,
        num_frames=args.num_frames,
        stride=args.stride,
        frame_regex=args.frame_regex,
        # world downsample
        wm_src_hw=args.wm_src_hw if args.wm_src_hw else (64, 64),
        wm_tgt_hw=args.wm_tgt_hw if args.wm_tgt_hw else (8, 4),
        wm_pool=args.wm_pool,
        # six-class external labels
        action6_key=getattr(args, "action6_key", "action_targets_npy"),
        action6_root=(getattr(args, "action6_root", None) or args.data_root),
        action6_template=getattr(args, "action6_template", None),
        logid_key=getattr(args, "logid_key", "meta.log_id"),
        group_index_key=getattr(args, "group_index_key", "meta.group_index"),
        group_size_key=getattr(args, "group_size_key", "meta.group_size"),
    )

    if "v_world" in ds_param_names:
        ds_kwargs["v_world"] = args.v_wm
    elif "v_wm" in ds_param_names:
        ds_kwargs["v_wm"] = args.v_wm

    ds = OceanVLADataset(**_filter_kwargs_for(OceanVLADataset.__init__, ds_kwargs))

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
            pool="avg",
            embed_dim=H,
            normalize=True,
            unfreeze_layers=int(getattr(args, "unfreeze_clip_layers", 0)),
        ),
        temporal=NS(
            enabled=bool(args.temporal_enabled),
            num_frames=int(args.num_frames),
            hidden_size=H,
            agg="gru",
            dropout=0.1,
            use_prev_action=False,
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

    # normalize tuple args
    args.wm_src_hw = _parse_hw_tuple(args.wm_src_hw) if isinstance(args.wm_src_hw, str) else args.wm_src_hw
    args.wm_tgt_hw = _parse_hw_tuple(args.wm_tgt_hw) if isinstance(args.wm_tgt_hw, str) else args.wm_tgt_hw

    # Tokenizer & data
    tok = build_tokenizer(args)
    ds, loader = build_loader(args, tok)

    # Model
    cfg = build_model_cfg(args)
    model = OceanVLA(cfg).to(device).eval()

    # Load checkpoint (strict or lenient)
    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    incompatible = model.load_state_dict(sd, strict=bool(args.ckpt_strict))

    def _len_safe(x):
        try:
            return len(x)
        except Exception:
            return 0

    miss = _len_safe(getattr(incompatible, 'missing_keys', []))
    unex = _len_safe(getattr(incompatible, 'unexpected_keys', []))
    print(f"[eval] loaded ckpt: {args.ckpt}, missing={miss}, unexpected={unex}")

    if (miss or unex) and args.fail_on_incompatible:
        raise RuntimeError(f"Checkpoint incompatible (missing={miss}, unexpected={unex}).")

    # Optionally inspect a few mismatched keys
    if args.print_incompat and (miss or unex):
        mk = list(getattr(incompatible, 'missing_keys', []))[:10]
        uk = list(getattr(incompatible, 'unexpected_keys', []))[:10]
        if mk:
            print("[eval] missing keys (first 10):", mk)
        if uk:
            print("[eval] unexpected keys (first 10):", uk)

    # --- Estimate prior safely for LA (optional) ---
    dataset_prior = None
    if args.la_infer in ("add", "sub", "auto"):
        try:
            K = int(args.v_act)
            cnt = np.zeros(K, dtype=np.int64)
            for it in getattr(ds, "items", []):
                rel = it.get("action_targets_npy", None)
                if not rel:
                    continue
                p = rel if os.path.isabs(rel) else os.path.join(args.data_root, rel)
                if not os.path.isfile(p):
                    continue
                arr = np.load(p, allow_pickle=False).reshape(-1)
                arr = arr[(arr >= 0) & (arr < K)]
                if arr.size:
                    cnt += np.bincount(arr, minlength=K)
            if cnt.sum() > 0:
                dataset_prior = (cnt / cnt.sum()).astype(float).tolist()
                print("[eval] dataset prior (npy) = " + ", ".join(f"{i}:{dataset_prior[i]:.4f}" for i in range(K)))
            dataset_prior = _safe_log_prior(dataset_prior, K)
        except Exception as e:
            warnings.warn(f"[eval] prior estimation failed: {e}")

    # Metrics accumulators
    tot_w_sum = 0.0
    tot_valid_w = 0
    
    # Top-1, Top-3, Top-5 准确率累加器
    tot_a1_correct = 0
    tot_a3_correct = 0
    tot_a5_correct = 0
    tot_a_valid = 0

    C = int(args.v_act)
    conf = np.zeros((C, C), dtype=int) if args.per_class else None
    per_class_correct = np.zeros(C, dtype=int) if args.per_class else None
    per_class_total = np.zeros(C, dtype=int) if args.per_class else None

    # Constant-pred detection
    global_pred_hist = np.zeros(C, dtype=np.int64)

    n_batches = 0
    la_log_prior = None  # [1,1,C] in log space

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    autocast_enabled = bool(args.amp)

    with torch.inference_mode():
        for batch in loader:
            # move tensors
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            # Optional ablations
            if args.zero_actions and isinstance(batch.get("actions"), torch.Tensor):
                batch["actions"] = torch.zeros_like(batch["actions"])
            if args.shuffle_images and isinstance(batch.get("images"), torch.Tensor):
                idx = torch.randperm(batch["images"].shape[0], device=device)
                batch["images"] = batch["images"][idx]

            # Sanity: image variance
            if isinstance(batch.get("images"), torch.Tensor) and args.print_inputs_stats:
                img_std = float(batch["images"].float().std().item())
                print(f"[dbg] images std={img_std:.5f}")

            # Prepare inputs
            txt_mask = batch.get("text_attention_mask", batch.get("attention_mask", None))
            actions_for_model = batch.get("actions", None)
            if args.actions_key == "__disable_actions__" or args.no_action_tokens:
                actions_for_model = None

            # Map image tensor
            img = batch.get("images", batch.get("image", None))
            allowed = set(inspect.signature(model.forward).parameters.keys())
            img_arg = None
            for kname in ("images", "image", "pixel_values", "vision_inputs"):
                if kname in allowed:
                    img_arg = kname
                    break
            model_in = dict(
                input_ids=batch.get("input_ids", None),
                text_attention_mask=txt_mask,
                actions=actions_for_model,
                action_targets=batch.get("action_targets", None),
                wm_targets=batch.get("wm_targets", None),
                actions_prev=batch.get("actions_prev", None),
                lambda_action=float(args.lambda_action),
                lambda_world=float(args.lambda_world),
                lambda_text=float(args.lambda_text),
            )
            if img is not None and img_arg is not None:
                model_in[img_arg] = img
            else:
                warnings.warn(
                    f"[eval] image tensor present={img is not None}, but model.forward expects none"
                )

            model_in = _filter_kwargs_for(model.forward, model_in)
            if args.print_inputs_stats:
                print("[dbg] forward accepted keys:", sorted(list(model_in.keys())))
                if img_arg:
                    print(f"[dbg] using image arg: {img_arg}")

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=autocast_enabled):
                out = model(**model_in)

            # World loss
            lw_sum = _get_world_loss_from_out(out)
            if lw_sum is None:
                wlog = out.get("world_logits", None)
                wtar = batch.get("wm_targets", None)
                if isinstance(wlog, torch.Tensor) and isinstance(wtar, torch.Tensor):
                    ce = _masked_cross_entropy(wlog.float(), wtar)
                    lw_sum = float(ce.item()) * int((wtar != IGNORE_INDEX).sum().item())
            if lw_sum is not None:
                valid_w = int((batch.get("wm_targets", torch.empty(0)) != IGNORE_INDEX).sum().item()) \
                          if isinstance(batch.get("wm_targets"), torch.Tensor) else 0
                tot_w_sum += float(lw_sum)
                tot_valid_w += int(valid_w)

            # Action logits / accuracy
            logits = _get_action_logits(out)
            gt = batch.get("action_targets", None)

            if isinstance(logits, torch.Tensor) and isinstance(gt, torch.Tensor):
                # LA setup (lazy)
                if la_log_prior is None and args.la_infer != "off":
                    if hasattr(model, "act_log_prior") and getattr(model, "act_log_prior") is not None:
                        la_log_prior = model.act_log_prior.detach().to(device=logits.device, dtype=logits.dtype).view(1, 1, -1)
                    elif dataset_prior is not None:
                        prior = torch.tensor(dataset_prior, device=logits.device, dtype=logits.dtype)
                        la_log_prior = prior.log().view(1, 1, -1)

                mode = args.la_infer
                if args.apply_la_infer and mode == "off":
                    mode = "add"
                if mode == "auto":
                    use_la_trained = bool(getattr(model, "act_use_la", False))
                    mode = "add" if (use_la_trained and la_log_prior is not None) else "off"
                if mode in ("add", "sub") and not bool(getattr(model, "act_use_la", False)):
                    mode = "off"

                if la_log_prior is not None and mode in ("add", "sub"):
                    tau = float(args.la_tau_infer)
                    if la_log_prior.shape[-1] != logits.shape[-1]:
                        warnings.warn(f"[la_infer] prior size mismatch -> skip")
                    else:
                        logits = logits + (tau * la_log_prior if mode == "add" else -tau * la_log_prior)

                # Diagnostics
                if args.print_logits_stats:
                    m, s, rowvar = _logits_stats(logits)
                    logits_flat = logits.reshape(-1, logits.shape[-1])
                    per_class_mean = logits_flat.float().mean(dim=0).cpu().numpy()
                    print(f"[dbg] logits mean={m:.4f} std={s:.4f} rowvar={rowvar:.6f}")
                    print("[dbg] per-class mean logits=" + ", ".join(f"{i}:{x:.3f}" for i, x in enumerate(per_class_mean)))

                # 计算 Top-1, Top-3, Top-5
                top1_cor, valid = _topk_accuracy(logits, gt, k=1, ignore_index=IGNORE_INDEX)
                top3_cor, _ = _topk_accuracy(logits, gt, k=3, ignore_index=IGNORE_INDEX)
                top5_cor, _ = _topk_accuracy(logits, gt, k=5, ignore_index=IGNORE_INDEX)
                
                tot_a1_correct += top1_cor
                tot_a3_correct += top3_cor
                tot_a5_correct += top5_cor
                tot_a_valid += valid

                # Top-1 predictions for histogram
                pred = logits.argmax(-1)
                mask = (gt != IGNORE_INDEX)

                # Global hist (constant-pred detector)
                pred_flat = pred.view(-1).detach().cpu().numpy()
                for v in pred_flat:
                    if 0 <= int(v) < C:
                        global_pred_hist[int(v)] += 1

                if args.per_class and conf is not None:
                    g = gt.view(-1).detach().cpu().numpy()
                    p = pred.view(-1).detach().cpu().numpy()
                    msk = mask.view(-1).detach().cpu().numpy().astype(bool)
                    for gi, pi, mi in zip(g, p, msk):
                        if mi and 0 <= gi < C and 0 <= pi < C:
                            conf[gi, pi] += 1
                            per_class_total[gi] += 1
                            if gi == pi:
                                per_class_correct[gi] += 1

                if args.print_pred_hist:
                    hist = np.bincount(pred_flat, minlength=C).astype(float)
                    frac = (hist / max(1.0, hist.sum())).round(3).tolist()
                    print(f"[eval_debug] pred_hist={frac}")

                if args.print_recall:
                    y = gt.view(-1)
                    p = pred.view(-1)
                    msk = (y != IGNORE_INDEX)
                    y, p = y[msk], p[msk]
                    recs = []
                    for c in range(C):
                        denom = int((y == c).sum().item())
                        rec = float(((p[y == c] == c).float().mean().item())) if denom > 0 else float("nan")
                        recs.append(None if math.isnan(rec) else round(rec, 3))
                    print(f"[recall] per-class={recs}")

            n_batches += 1
            if args.num_batches > 0 and n_batches >= args.num_batches:
                break

            if device == "cuda" and (n_batches % 50 == 0):
                torch.cuda.empty_cache()

    # ==================== Report ====================
    print("\n" + "="*60)
    print("[EVALUATION RESULTS]")
    print("="*60)
    
    # World model results
    if tot_valid_w > 0:
        w_tok = tot_w_sum / max(1, tot_valid_w)
        ppl = math.exp(w_tok)
        print(f"World Model:")
        print(f"  - Valid tokens: {tot_valid_w}")
        print(f"  - Token loss (nats): {w_tok:.4f}")
        print(f"  - Perplexity: {ppl:.3f}")
    else:
        print(f"World Model: N/A (no valid tokens)")

    print()
    
    # Action prediction results
    if tot_a_valid > 0:
        acc1 = 100.0 * tot_a1_correct / tot_a_valid
        acc3 = 100.0 * tot_a3_correct / tot_a_valid
        acc5 = 100.0 * tot_a5_correct / tot_a_valid
        
        print(f"Action Prediction:")
        print(f"  - Total batches: {n_batches}")
        print(f"  - Total samples: {len(ds)}")
        print(f"  - Valid predictions: {tot_a_valid}")
        print(f"  - Top-1 Accuracy: {acc1:.2f}%  ({tot_a1_correct}/{tot_a_valid})")
        print(f"  - Top-3 Accuracy: {acc3:.2f}%  ({tot_a3_correct}/{tot_a_valid})")
        print(f"  - Top-5 Accuracy: {acc5:.2f}%  ({tot_a5_correct}/{tot_a_valid})")
    else:
        print("Action Prediction: N/A (no valid labels)")

    print()

    # Per-class statistics
    if args.per_class and per_class_total is not None:
        print(f"Per-Class Statistics:")
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_acc = np.divide(per_class_correct, np.maximum(1, per_class_total))
        print(f"  - Per-class accuracy (%): {np.round(per_class_acc * 100, 2).tolist()}")
        print(f"  - Per-class samples: {per_class_total.tolist()}")
        print(f"\nConfusion Matrix [ground_truth, prediction]:")
        print(conf)

    print()

    # Constant prediction detector
    total_preds = int(global_pred_hist.sum())
    if total_preds > 0:
        frac = (global_pred_hist / max(1, total_preds))
        top_c = int(frac.argmax())
        top_f = float(frac[top_c])
        print(f"Prediction Distribution:")
        print(f"  - Total predictions: {total_preds}")
        print(f"  - Histogram: {dict((i, int(global_pred_hist[i])) for i in range(C))}")
        print(f"  - Most frequent class: {top_c} ({top_f*100:.2f}%)")
        
        if top_f >= args.constant_alert_threshold:
            print("\n" + "!"*60)
            warnings.warn(
                f"[ALERT] Predictions are {top_f*100:.2f}% on class {top_c} -> likely constant output!\n"
                f"Suggested actions:\n"
                f"  1. Check temporal config matches training\n"
                f"  2. Verify no missing/unexpected keys in checkpoint\n"
                f"  3. Try ablation: --shuffle_images or --no_action_tokens\n"
                f"  4. Ensure feature flow is correct"
            )
            print("!"*60)
    
    print("="*60 + "\n")


# ----------------------------- argparse ----------------------------- #

def parse_args():
    p = argparse.ArgumentParser("OceanVLA robust evaluator with top-k accuracy")
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
    p.add_argument("--h_action", type=int, default=6)
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
    # World downsample
    p.add_argument("--wm_src_hw", default=None)
    p.add_argument("--wm_tgt_hw", default=None)
    p.add_argument("--wm_pool", type=str, default="mode", choices=["mode", "center"])
    # Batch / loop
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_batches", type=int, default=100, help="Max batches; -1 for all")
    # Checkpoint
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--ckpt_strict", action="store_true")
    p.add_argument("--fail_on_incompatible", action="store_true")
    p.add_argument("--print_incompat", action="store_true")
    # Loss weights
    p.add_argument("--lambda_action", type=float, default=1.0)
    p.add_argument("--lambda_world", type=float, default=1.5)
    p.add_argument("--lambda_text", type=float, default=0.0)
    # Ablations
    p.add_argument("--zero_actions", action="store_true")
    p.add_argument("--no_action_tokens", action="store_true")
    p.add_argument("--shuffle_images", action="store_true")
    p.add_argument("--print_inputs_stats", action="store_true")
    # Reporting
    p.add_argument("--per_class", action="store_true")
    p.add_argument("--print_pred_hist", action="store_true")
    p.add_argument("--print_recall", action="store_true")
    p.add_argument("--print_logits_stats", action="store_true")
    p.add_argument("--constant_alert_threshold", type=float, default=0.98)
    # Inference-time LA
    p.add_argument("--la_infer", type=str, default="off", choices=["off", "add", "sub", "auto"])
    p.add_argument("--la_tau_infer", type=float, default=1.0)
    p.add_argument("--apply_la_infer", action="store_true")
    # AMP
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    return p.parse_args()


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()