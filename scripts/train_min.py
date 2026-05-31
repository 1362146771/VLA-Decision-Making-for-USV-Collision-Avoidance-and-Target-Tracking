#!/usr/bin/env python3
# core/scripts/train_min.py
"""
OceanVLA three-stage training script (Section 4.5).

Stage 1 — Ocean World Model pretraining  (20 epochs, lr=3e-4, Adam, batch=64)
Stage 2 — Maritime CLIP fine-tuning      (10 epochs, lr=1e-5, on 15 K maritime pairs)
Stage 3 — Transformer Policy training    (50 epochs, lr=3e-4, AdamW, frozen WM)

Hardware: 4× NVIDIA PRO6000 GPUs, mixed-precision FP16
Total training time: ≈36 hours (20h WM + 6h CLIP + 10h policy) — Section 4.5

Focal loss (γ=2.0) and class-balanced sampling address the severe action imbalance
in the collision-avoidance dataset (FORWARD: 58.2%, STOP: 2.4%).

Usage:
    # Stage 3 policy training (most common)
    python -m core.scripts.train_min \\
        --stage policy \\
        --data_root /path/to/data \\
        --jsonl train.jsonl \\
        --tokenizer_path /path/to/tokenizer \\
        --model_cfg core/configs/model/ocean_vla.yaml \\
        --batch_size 32 --epochs 50 --lr 3e-4

    # Stage 1 world model pretraining
    python -m core.scripts.train_min \\
        --stage world_model \\
        --epochs 20 --lr 3e-4 --batch_size 64
"""

import argparse
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.modeling.ocean_model import OceanVLA, OceanWorldModel
from core.training.optimization import build_optimizer
from transformers import AutoTokenizer
from core.data.datasets.collate import collate_fn
from core.data.datasets.ocean_dataset import OceanVLADataset


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _parse_hw_tuple(s):
    if not s:
        return None
    if isinstance(s, (tuple, list)) and len(s) == 2:
        return int(s[0]), int(s[1])
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid hw tuple: {s!r} (expected 'H,W')")
    return int(parts[0]), int(parts[1])


def _parse_floats_csv(s):
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    try:
        return [float(x) for x in parts]
    except Exception:
        raise argparse.ArgumentTypeError(f"Cannot parse class_prior: {s!r}")


def _to_float(x, default=0.0):
    if x is None:
        return float(default)
    if torch.is_tensor(x):
        return float(x.detach().item())
    try:
        return float(x)
    except Exception:
        return float(default)


def _filter_kwargs_for(fn, kwargs):
    import inspect
    try:
        allowed = set(inspect.signature(fn).parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)


def sanity_check_first_batch(loader, num_classes=6, h_world=32):
    """Quick sanity check on the first batch: shapes, label coverage, VQ token range."""
    b = next(iter(loader))
    img   = b.get("images", b.get("image", None))
    a     = b["action_targets"]
    w     = b["wm_targets"]
    labels = b.get("labels", None)
    text_ntok = int((labels != -100).sum().item()) if labels is not None else 0

    valid_mask = a != -100
    valid_cnt  = int(valid_mask.sum().item())
    a_valid    = a[valid_mask].view(-1)
    if a_valid.numel() > 0:
        uniq = sorted(a_valid.unique().tolist())
        hist = torch.bincount(a_valid, minlength=num_classes).tolist()
        amin, amax = int(a_valid.min()), int(a_valid.max())
    else:
        uniq, hist, amin, amax = "none", [], None, None

    w_valid = int((w != -100).sum())
    print(
        "[sanity]",
        "images", tuple(img.shape) if img is not None else None,
        "| action_targets", tuple(a.shape), f"valid={valid_cnt}",
        f"unique={uniq}", f"min/max={amin}/{amax}",
        f"hist(0..{num_classes-1})={hist}",
        "| wm", tuple(w.shape), f"valid={w_valid}/{w.numel()}",
        f"| text_tokens={text_ntok}",
    )


# ---------------------------------------------------------------------------
# Dummy dataset (pipeline smoke test)
# ---------------------------------------------------------------------------

class DummyMultimodalDS(Dataset):
    """Synthetic dataset for pipeline smoke testing without real data."""

    def __init__(self, N=128, T_txt=32, K=6, A_dim=4,
                 H_action=6, H_world=32, V_act=6, V_wm=8192, img_size=224):
        self.N = N
        self.T_txt   = T_txt
        self.K       = K
        self.A_dim   = A_dim
        self.H_action = H_action
        self.H_world  = H_world
        self.V_act    = V_act
        self.V_wm     = V_wm
        self.img_size = img_size

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return dict(
            input_ids      = torch.randint(5, 10000, (self.T_txt,), dtype=torch.long),
            labels         = torch.full((self.T_txt,), -100, dtype=torch.long),
            image          = torch.randn(3, self.img_size, self.img_size),
            action_targets = torch.randint(0, self.V_act, (self.H_action,), dtype=torch.long),
            wm_targets     = torch.randint(0, self.V_wm,  (self.H_world,),  dtype=torch.long),
            actions        = torch.zeros(self.K, self.A_dim),
        )


# ---------------------------------------------------------------------------
# Tokenizer / DataLoader builders
# ---------------------------------------------------------------------------

def build_tokenizer(args):
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _major_class_from_item(it, args, data_root):
    """Return the most frequent action class in a sample's .npy label file, or None."""
    rel = it.get(args.action6_key, None) if args.action6_key else None
    if not rel:
        rel = it.get("action_targets_npy", None)
    if not rel:
        return None
    p = rel if os.path.isabs(rel) else os.path.join(data_root, rel)
    if not os.path.isfile(p):
        return None
    try:
        arr = np.load(p, allow_pickle=False).reshape(-1)
        vals, cnts = np.unique(arr, return_counts=True)
        return int(vals[np.argmax(cnts)])
    except Exception:
        return None


def build_dataloader(args):
    tokenizer = build_tokenizer(args)

    if args.dataset == "dummy":
        ds = DummyMultimodalDS(
            N=args.num_samples,
            T_txt=args.t_txt,
            K=args.k_act,
            A_dim=args.a_dim,
            H_action=args.h_action,
            H_world=args.h_world,
            V_act=args.v_act,
            V_wm=args.v_wm,
            img_size=args.img_size,
        )
    else:
        ds_kwargs = dict(
            root=args.data_root,
            jsonl=args.jsonl,
            tokenizer=tokenizer,
            img_size=args.img_size,
            k=args.k_act,
            a_dim=args.a_dim,
            h_world=args.h_world,
            v_world=args.v_wm,
            use_random_wm=args.use_random_wm,
            temporal_enabled=args.temporal,
            num_frames=args.num_frames,
            stride=args.stride,
            text_supervision=args.text_supervision,
            min_text_tokens=args.min_text_tokens,
            use_clip_processor=not args.no_clip_processor,
            clip_path=args.clip_path,
        )
        if args.wm_src_hw:
            ds_kwargs["wm_src_hw"] = args.wm_src_hw
        if args.wm_tgt_hw:
            ds_kwargs["wm_tgt_hw"] = args.wm_tgt_hw
        if args.action6_key:
            ds_kwargs["action6_key"] = args.action6_key

        ds_kwargs = _filter_kwargs_for(OceanVLADataset.__init__, ds_kwargs)
        ds = OceanVLADataset(**ds_kwargs)

    # Class-balanced sampling (Section 4.5): equal representation per action class
    sampler = None
    if args.weighted_sampling and args.dataset != "dummy" and hasattr(ds, "items"):
        class_ids = [_major_class_from_item(it, args, args.data_root) for it in ds.items]
        valid = [(i, c) for i, c in enumerate(class_ids) if c is not None]
        if valid:
            counter = Counter(c for _, c in valid)
            weight_per_class = {c: 1.0 / max(cnt, 1) for c, cnt in counter.items()}
            sample_weights = [weight_per_class.get(class_ids[i], 1e-6) for i in range(len(ds))]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(ds),
                replacement=True,
            )

    _collate = lambda b: collate_fn(
        b, tokenizer,
        k_act=args.k_act,
        h_world=args.h_world,
        add_image_token=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
        drop_last=True,
    )
    return loader, tokenizer


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(args):
    """Instantiate OceanVLA from YAML config or CLI overrides."""
    from types import SimpleNamespace as NS

    if args.model_cfg and os.path.isfile(args.model_cfg):
        import yaml
        with open(args.model_cfg) as f:
            cfg_dict = yaml.safe_load(f)

        def _to_ns(d):
            if isinstance(d, dict):
                return NS(**{k: _to_ns(v) for k, v in d.items()})
            return d

        config = _to_ns(cfg_dict)
    else:
        # Minimal config from CLI arguments
        config = NS(
            backbone=NS(name_or_path=args.backbone_path, torch_dtype="bfloat16",
                        gradient_checkpointing=True, freeze_layers=0, device_map=None),
            vision=NS(vision_model=args.clip_path, pool="projected",
                      out_dim=None, normalize=False, unfreeze_layers=12),
            temporal=NS(enabled=args.temporal, num_frames=args.num_frames,
                        agg="gru", hidden_size=768, dropout=0.1, use_prev_action=False),
            action=NS(input_dim=args.a_dim, chunk_size=args.k_act, hidden_size=768,
                      discrete_dim=args.v_act, mlp_hidden=256, dropout=0.0,
                      use_class_weights=True, use_focal=True, focal_gamma=2.0,
                      label_smoothing=0.0, class_prior=args.class_prior,
                      loss_last_only=False),
            world_model=NS(hidden_size=768, latent_dim=256, pred_horizon=4,
                           seq_len=args.h_world, vocab_size=args.v_wm,
                           temperature=1.0, label_smoothing=0.0,
                           lambda_dyn=1.0, lambda_risk=0.5),
        )

    model = OceanVLA(config)
    return model


# ---------------------------------------------------------------------------
# Optimizer builder
# ---------------------------------------------------------------------------

def build_optimizer_for_stage(model, args, stage="policy"):
    """
    Build AdamW optimizer with separate learning rates for different parameter groups.

    Stage 1 (world_model): train only world model parameters
    Stage 2 (clip):        train only CLIP encoder parameters at low lr
    Stage 3 (policy):      train everything except frozen world model
    """
    if stage == "world_model":
        params = [p for p in model.ocean_world_model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=args.lr, weight_decay=0.0)

    if stage == "clip":
        params = [p for p in model.vision_encoder.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=1e-5, weight_decay=args.weight_decay)

    # Default: Stage 3 policy training (AdamW, lr=3e-4, cosine annealing)
    # Freeze world model weights
    for p in model.ocean_world_model.parameters():
        p.requires_grad_(False)

    clip_params = set(model.vision_encoder.parameters())
    groups = [
        {"params": [p for p in model.parameters()
                    if p.requires_grad and p not in clip_params],
         "lr": args.lr},
        {"params": [p for p in clip_params if p.requires_grad],
         "lr": args.clip_lr or args.lr},
    ]
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="OceanVLA Training Script")

    # Data
    p.add_argument("--dataset",      default="dummy",   choices=["dummy", "ocean"])
    p.add_argument("--data_root",    default="data")
    p.add_argument("--jsonl",        default="train.jsonl")
    p.add_argument("--num_samples",  type=int, default=256,  help="Dummy dataset size")

    # Model
    p.add_argument("--model_cfg",    default="")
    p.add_argument("--backbone_path", default="/root/autodl-tmp/models--google--gemma-2b")
    p.add_argument("--clip_path",    default="/root/autodl-tmp/models--openai--clip-vit-large-patch14")
    p.add_argument("--tokenizer_path", default="/root/autodl-tmp/models--google--gemma-2b")
    p.add_argument("--checkpoint",   default="")

    # Stage
    p.add_argument("--stage", default="policy",
                   choices=["world_model", "clip", "policy"],
                   help="Training stage (Section 4.5)")

    # Data dimensions
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--t_txt",       type=int, default=32)
    p.add_argument("--k_act",       type=int, default=6,    help="Action chunk size K")
    p.add_argument("--a_dim",       type=int, default=6,    help="Action input dimension")
    p.add_argument("--h_action",    type=int, default=6,    help="Discrete action context length")
    p.add_argument("--h_world",     type=int, default=32,   help="World token sequence length N")
    p.add_argument("--v_act",       type=int, default=6,    help="Action vocabulary size |A|")
    p.add_argument("--v_wm",        type=int, default=8192, help="World VQ vocabulary size")
    p.add_argument("--wm_src_hw",   type=_parse_hw_tuple, default=None)
    p.add_argument("--wm_tgt_hw",   type=_parse_hw_tuple, default=None)

    # Training hyperparameters (Section 4.5)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--max_steps",    type=int,   default=0)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--clip_lr",      type=float, default=None, help="CLIP encoder learning rate")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int,   default=100)
    p.add_argument("--lambda_action", type=float, default=1.0)
    p.add_argument("--lambda_world",  type=float, default=1.0)
    p.add_argument("--class_prior",  type=_parse_floats_csv, default=None,
                   help="Comma-separated prior probabilities per class, e.g. 0.582,0.024,...")

    # Loss configuration
    p.add_argument("--no_focal",         action="store_true")
    p.add_argument("--focal_gamma",      type=float, default=2.0)
    p.add_argument("--label_smoothing",  type=float, default=0.0)
    p.add_argument("--weighted_sampling", action="store_true",
                   help="Enable class-balanced weighted sampling (Section 4.5)")

    # Temporal / augmentation
    p.add_argument("--temporal",       action="store_true")
    p.add_argument("--num_frames",     type=int, default=3, help="Memory bank size M")
    p.add_argument("--stride",         type=int, default=1)
    p.add_argument("--text_supervision", action="store_true")
    p.add_argument("--min_text_tokens", type=int, default=4)

    # Infrastructure
    p.add_argument("--no_clip_processor", action="store_true")
    p.add_argument("--use_random_wm",     action="store_true")
    p.add_argument("--action6_key",       default="action_targets_npy")
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--log_every",    type=int, default=20)
    p.add_argument("--save_every",   type=int, default=500)
    p.add_argument("--save_dir",     default="checkpoints")
    p.add_argument("--amp",          action="store_true", default=True)
    p.add_argument("--no_amp",       action="store_false", dest="amp")
    p.add_argument("--sanity_check", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  stage={args.stage}  dataset={args.dataset}")

    os.makedirs(args.save_dir, exist_ok=True)

    # --- Data ---
    loader, tokenizer = build_dataloader(args)
    if args.sanity_check:
        sanity_check_first_batch(loader, num_classes=args.v_act, h_world=args.h_world)

    # --- Model ---
    model = build_model(args)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[train] Checkpoint loaded: {args.checkpoint}")
        print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
    model = model.to(device)

    # --- Optimizer ---
    optimizer = build_optimizer_for_stage(model, args, stage=args.stage)

    # --- Scheduler: cosine with warmup (Section 4.5) ---
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * len(loader)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- AMP scaler ---
    use_amp = args.amp and torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # --- Training loop ---
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(loader):
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                fwd_kwargs = {
                    "input_ids":           batch["input_ids"],
                    "text_attention_mask": batch["text_attention_mask"],
                    "images":              batch["images"],
                    "actions":             batch["actions"],
                    "action_targets":      batch["action_targets"],
                    "wm_targets":          batch["wm_targets"],
                    "lambda_action":       args.lambda_action,
                    "lambda_world":        args.lambda_world,
                }
                if "actions_prev" in batch:
                    fwd_kwargs["actions_prev"] = batch["actions_prev"]

                out = model(**_filter_kwargs_for(model.forward, fwd_kwargs))
                loss = out["loss"]

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            global_step += 1

            if global_step % args.log_every == 0:
                loss_a = _to_float(out.get("loss_a"))
                loss_w = _to_float(out.get("loss_w"))
                lr_now = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                print(
                    f"[train] epoch={epoch+1}/{args.epochs} "
                    f"step={global_step}/{total_steps} "
                    f"loss={_to_float(loss):.4f} "
                    f"loss_a={loss_a:.4f} loss_w={loss_w:.4f} "
                    f"lr={lr_now:.2e} "
                    f"elapsed={elapsed/60:.1f}min"
                )

            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.save_dir, f"step_{global_step:06d}.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                }, ckpt_path)
                print(f"[train] Checkpoint saved: {ckpt_path}")

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Save final checkpoint
    final_path = os.path.join(args.save_dir, "final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "global_step": global_step,
        "args": vars(args),
    }, final_path)
    print(f"[train] Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
