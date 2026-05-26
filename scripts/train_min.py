# core/scripts/train_min.py (drop-in fixed)
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ===== 采样/统计依赖 =====
import numpy as np
from collections import Counter
from torch.utils.data import WeightedRandomSampler

# 允许从仓库根目录运行：python -m core.scripts.train_min
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.modeling.ocean_model import OceanVLA
from core.training.optimization import build_optimizer  # 预留，不强依赖

# 数据与工具
from transformers import AutoTokenizer
from core.data.datasets.collate import collate_fn
from core.data.datasets.ocean_dataset import OceanVLADataset


# ---------------------------
# 实用函数
# ---------------------------
def _parse_hw_tuple(s: str | None):
    """将 'H,W' 字符串解析为 (int(H), int(W))。传 None/空串返回 None。"""
    if not s:
        return None
    if isinstance(s, (tuple, list)) and len(s) == 2:
        return int(s[0]), int(s[1])
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid hw tuple: {s} (expect 'H,W')")
    return int(parts[0]), int(parts[1])


def _parse_floats_csv(s: str | None):
    """解析逗号分隔浮点；传 None/空串返回 None。允许空格。"""
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    try:
        return [float(x) for x in parts]
    except Exception:
        raise argparse.ArgumentTypeError(f"--class_prior 解析失败: {s}")


def _to_float(x, default=0.0):
    """把 (None / tensor / 标量) 安全转成 float。"""
    if x is None:
        return float(default)
    if torch.is_tensor(x):
        return float(x.detach().item())
    try:
        return float(x)
    except Exception:
        return float(default)


def _filter_kwargs_for(callable_obj, kwargs: dict) -> dict:
    """只保留 callable 的签名里存在的关键字参数，避免 forward 不接受时报错。"""
    try:
        import inspect
        sig = inspect.signature(callable_obj)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)


def sanity_check_first_batch(loader, num_classes=6, h_world=64):
    """拉一个 batch 做体检：看图像形状、动作标签是否非全 -100、类别范围、以及 VQ 长度与文本有效 token。"""
    import torch
    b = next(iter(loader))

    img = b.get("images", b.get("image", None))
    a = b["action_targets"]
    w = b["wm_targets"]
    labels = b.get("labels", None)
    text_ntok = int((labels != -100).sum().item()) if labels is not None else 0

    valid_mask = (a != -100)
    valid_cnt = int(valid_mask.sum().item())
    a_valid = a[valid_mask].view(-1)
    if a_valid.numel() > 0:
        uniq = sorted(a_valid.unique().tolist())
        hist = torch.bincount(a_valid, minlength=num_classes).tolist()
        amin = int(a_valid.min().item()); amax = int(a_valid.max().item())
    else:
        uniq, hist, amin, amax = "none", [], None, None

    w_valid = int((w != -100).sum().item())
    w_total = int(w.numel())

    print(
        "[sanity]",
        "images", tuple(img.shape) if img is not None else None,
        "| actions(targets)", tuple(a.shape), f"valid={valid_cnt}",
        f"uniq={uniq}", f"min/max={amin}/{amax}", f"hist(0..{num_classes-1})={hist}",
        "| wm", tuple(w.shape), f"valid={w_valid}/{w_total}",
        f"| text_ntok={text_ntok}"
    )


# ---------------------------
# Dummy 数据集（打通链路）
# ---------------------------
class DummyMultimodalDS(Dataset):
    def __init__(self, N=128, T_txt=32, K=6, A_dim=4, H_action=6, H_world=64,
                 V_act=6, V_wm=8192, img_size=224):
        super().__init__()
        self.N = N
        self.T_txt = T_txt
        self.K = K
        self.A_dim = A_dim
        self.H_action = H_action
        self.H_world = H_world
        self.V_act = V_act
        self.V_wm = V_wm
        self.img_size = img_size

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        input_ids = torch.randint(5, 10000, (self.T_txt,), dtype=torch.long)
        labels = torch.full((self.T_txt,), -100, dtype=torch.long)
        image = torch.randn(3, self.img_size, self.img_size)
        action_targets = torch.randint(0, self.V_act, (self.H_action,), dtype=torch.long)
        wm_targets = torch.randint(0, self.V_wm, (self.H_world,), dtype=torch.long)
        return dict(
            input_ids=input_ids,
            labels=labels,
            image=image,
            action_targets=action_targets,
            wm_targets=wm_targets,
        )


def build_tokenizer(args):
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _major_class_from_item(it, args, data_root):
    """读取样本对应的 .targets.npy，返回该样本里出现最多的类别（0..5），失败返回 None。"""
    rel = it.get(args.action6_key, None) if args.action6_key else None
    if (not rel) and args.action6_key != "action_targets_npy":
        rel = it.get("action_targets_npy", None)
    if not rel:
        return None
    p = rel if os.path.isabs(rel) else os.path.join(data_root, rel)
    if not os.path.isfile(p):
        return None
    try:
        arr = np.load(p, allow_pickle=False).reshape(-1)
        if arr.size == 0:
            return None
        vals, cnts = np.unique(arr, return_counts=True)
        return int(vals[int(np.argmax(cnts))])
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
        sampler = None
        ds._class_prior = None

    elif args.dataset == "real":
        ds = OceanVLADataset(
            root=args.data_root,
            jsonl=args.jsonl,
            tokenizer=tokenizer,
            img_size=args.img_size,
            k=args.k_act,
            a_dim=args.a_dim,
            h_world=args.h_world,
            v_world=args.v_wm,
            use_random_wm=args.use_random_wm,
            image_key=args.image_key,
            text_key=args.text_key,
            actions_key="__disable_actions__",
            wm_key=args.wm_key,
            use_clip_processor=args.use_clip_processor,
            clip_path=args.clip_path,
            temporal_enabled=args.temporal_enabled,
            num_frames=args.num_frames,
            stride=args.stride,
            frame_regex=args.frame_regex,
            wm_src_hw=args.wm_src_hw if args.wm_src_hw else (64, 64),
            wm_tgt_hw=args.wm_tgt_hw,
            wm_pool=args.wm_pool,
            action6_key=args.action6_key,
            action6_root=args.action6_root if args.action6_root else args.data_root,
            action6_template=args.action6_template,
            logid_key=args.logid_key,
            group_index_key=args.group_index_key,
            group_size_key=args.group_size_key,
            text_supervision=args.text_supervision,
            min_text_tokens=args.min_text_tokens,
        )

        K = int(getattr(args, "v_act", 6))
        token_counts = np.zeros(K, dtype=np.int64)

        def _accum_token_counts_from_item(it, args, data_root):
            rel = it.get(args.action6_key, None) if args.action6_key else None
            if (not rel) and args.action6_key != "action_targets_npy":
                rel = it.get("action_targets_npy", None)
            if not rel:
                return
            p = rel if os.path.isabs(rel) else os.path.join(data_root, rel)
            if not os.path.isfile(p):
                return
            try:
                arr = np.load(p, allow_pickle=False).reshape(-1)
                arr = arr[(arr >= 0) & (arr < K)]
                if arr.size:
                    token_counts[:] += np.bincount(arr, minlength=K)
            except Exception:
                pass

        for it in getattr(ds, "items", []):
            _accum_token_counts_from_item(it, args, args.data_root)

        total_tokens = int(token_counts.sum())
        if total_tokens > 0:
            prior_list = (token_counts / total_tokens).astype(float).tolist()
        else:
            prior_list = [1.0 / K] * K
        ds._class_prior = prior_list
        print(f"[stats] token-level class prior = {{"
              + ", ".join([f"{i}:{prior_list[i]:.4f}" for i in range(K)]) + "}}")

        classes = []
        for it in getattr(ds, "items", []):
            c = _major_class_from_item(it, args, args.data_root)
            classes.append(0 if c is None else int(c))
        freq = Counter(classes)
        sampler = None
        if args.weighted_sampling:
            min_freq = max(1, min([v for k, v in freq.items() if v > 0] + [1]))
            weights = [1.0 / (freq.get(c, min_freq)) for c in classes]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            print("[sampler] enable WeightedRandomSampler(replacement=True)")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda batch: collate_fn(
            batch,
            tokenizer=tokenizer,
            k_act=args.k_act,
            h_world=args.h_world,
            add_image_token=True,
        ),
    )
    return loader


def parse_args():
    p = argparse.ArgumentParser()
    # 数据选择
    p.add_argument("--dataset", type=str, default="dummy", choices=["dummy", "real"])
    p.add_argument("--num_samples", type=int, default=256)
    p.add_argument("--data_root", type=str, default="/root/autodl-tmp/数据")
    p.add_argument("--jsonl", type=str, default="samples.jsonl")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)

    # tokenizer
    p.add_argument("--tokenizer_path", type=str,
                   default="/root/autodl-tmp/models--google--gemma-2b")

    # 序列长度/码表
    p.add_argument("--t_txt", type=int, default=32)
    p.add_argument("--k_act", type=int, default=6)
    p.add_argument("--a_dim", type=int, default=6)
    p.add_argument("--h_action", type=int, default=6)
    p.add_argument("--h_world", type=int, default=64)
    p.add_argument("--v_act", type=int, default=6)
    p.add_argument("--v_wm", type=int, default=8192)

    p.add_argument("--use_random_wm", action="store_true")

    # 字段映射 & CLIP 预处理
    p.add_argument("--image_key", type=str, default="image")
    p.add_argument("--text_key", type=str, default="text")
    p.add_argument("--actions_key", type=str, default="__disable_actions__")
    p.add_argument("--wm_key", type=str, default="wm_targets")
    p.add_argument("--use_clip_processor", action="store_true")
    p.add_argument("--clip_path", type=str,
                   default="/root/autodl-tmp/models--openai--clip-vit-base-patch32")

    # 时序参数
    p.add_argument("--temporal_enabled", action="store_true")
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--frame_regex", type=str, default=r"frame[_\-]?(\d+)")

    # world token 下采样参数
    p.add_argument("--wm_src_hw", type=_parse_hw_tuple, default="64,64")
    p.add_argument("--wm_tgt_hw", type=_parse_hw_tuple, default="8,8")
    p.add_argument("--wm_pool", type=str, default="mode", choices=["mode", "center"])

    # 离散动作 6 类标签来源
    p.add_argument("--action6_key", type=str, default="action_targets_npy")
    p.add_argument("--action6_root", type=str, default="")
    p.add_argument("--action6_template", type=str, default="")
    p.add_argument("--logid_key", type=str, default="meta.log_id")
    p.add_argument("--group_index_key", type=str, default="meta.group_index")
    p.add_argument("--group_size_key", type=str, default="meta.group_size")

    # 训练
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_backbone", type=float, default=3e-5)
    p.add_argument("--lr_heads", type=float, default=3e-4)
    p.add_argument("--lr_vision", type=float, default=1e-5,
                   help="CLIP 视觉编码器的学习率（解冻的层）")
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="开启混合精度")

    # LR Scheduler
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--min_lr_mult", type=float, default=0.1)

    # world 温度退火
    p.add_argument("--world_temp_start", type=float, default=2.0)
    p.add_argument("--world_temp_end", type=float, default=1.0)
    p.add_argument("--world_temp_anneal", type=int, default=5000)

    # 保存
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=100)

    # precision
    p.add_argument("--precision", type=str, default="bf16", choices=["fp16", "bf16"])

    # 冻结骨干/解冻最后 N 层
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--unfreeze_last", type=int, default=0)

    # ★ CLIP 视觉编码器解冻
    p.add_argument("--unfreeze_clip_layers", type=int, default=0,
                   help="解冻 CLIP ViT 的最后 N 层 (0=全冻结, 4=解冻后4层)")

    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_layers_last", type=int, default=4)
    p.add_argument("--lora_targets", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # world 初始化温度/label smoothing
    p.add_argument("--world_temperature", type=float, default=1.0)
    p.add_argument("--world_label_smoothing", type=float, default=0.0)
    p.add_argument("--init_from", type=str, default="")
    p.add_argument("--freeze_heads", action="store_true")
    p.add_argument("--freeze_heads_steps", type=int, default=0)

    # 文本损失权重
    p.add_argument("--lambda_text", type=float, default=1.0)

    # 动作/世界损失权重
    p.add_argument("--lambda_action", type=float, default=1.0)
    p.add_argument("--lambda_world", type=float, default=1.0)

    # 文本监督开关
    p.add_argument("--text_supervision", action="store_true", default=True)
    p.add_argument("--min_text_tokens", type=int, default=4)

    # 采样与类别不平衡修正
    p.add_argument("--weighted_sampling", action="store_true")
    p.add_argument("--use_la", action="store_true")
    p.add_argument("--la_tau", type=float, default=1.0)
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--class_prior", type=str, default="")
    p.add_argument("--use_focal", action="store_true")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--prior_reg_lambda", type=float, default=0.0)

    p.add_argument("--loss_last_only", action="store_true")

    return p.parse_args()


# ---------------------------
# Backbone 解冻/LoRA 工具
# ---------------------------
def _get_hf_model_from_backbone(backbone):
    for name in ["hf_model", "model", "backbone", "base_model", "transformer"]:
        if hasattr(backbone, name):
            return getattr(backbone, name)
    return backbone


def _get_decoder_layers(hf):
    cand = [
        ("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
        ("model.decoder.layers", lambda m: getattr(getattr(m, "model", None), "decoder", None) and getattr(getattr(m, "model", None).decoder, "layers", None)),
        ("layers", lambda m: getattr(m, "layers", None)),
        ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None)),
    ]
    for name, fn in cand:
        layers = fn(hf)
        if layers is not None:
            return layers
    raise RuntimeError("Unable to locate decoder layers on HF model")


def unfreeze_last_n_layers(backbone, n_last: int):
    hf = _get_hf_model_from_backbone(backbone)
    layers = _get_decoder_layers(hf)
    total = len(layers)
    keep = max(0, total - n_last)
    for p in hf.parameters():
        p.requires_grad_(False)
    for li in range(keep, total):
        for p in layers[li].parameters():
            p.requires_grad_(True)
    for name in ["model.norm", "ln_f", "final_layernorm", "norm"]:
        mod = getattr(getattr(hf, "model", hf), name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad_(True)
    return total, n_last


class LoRALinear(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)

        self.r = int(r)
        self.scaling = float(alpha) / max(1, self.r)

        dev = linear.weight.device
        dt  = linear.weight.dtype

        self.lora_A = torch.nn.Linear(
            linear.in_features, self.r, bias=False, device=dev, dtype=dt
        )
        self.lora_B = torch.nn.Linear(
            self.r, linear.out_features, bias=False, device=dev, dtype=dt
        )
        self.dropout = (
            torch.nn.Dropout(dropout) if dropout and dropout > 0 else torch.nn.Identity()
        )

        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        y = self.linear(x)
        if self.r > 0:
            y = y + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return y


def apply_lora_to_last_layers(backbone, layers_last=4, target_names=(), r=16, alpha=32, dropout=0.05):
    hf = _get_hf_model_from_backbone(backbone)
    layers = _get_decoder_layers(hf)
    total = len(layers)
    start = max(0, total - int(layers_last))

    for p in hf.parameters():
        p.requires_grad_(False)

    replaced = 0
    for li in range(start, total):
        block = layers[li]
        for tgt in target_names:
            if not hasattr(block, "self_attn") and not hasattr(block, "mlp"):
                continue
            attn = getattr(block, "self_attn", None)
            if attn is not None and hasattr(attn, tgt):
                mod = getattr(attn, tgt)
                if isinstance(mod, torch.nn.Linear):
                    lora = LoRALinear(mod, r=r, alpha=alpha, dropout=dropout)
                    setattr(attn, tgt, lora)
                    replaced += 1
            mlp = getattr(block, "mlp", None)
            if mlp is not None and hasattr(mlp, tgt):
                mod = getattr(mlp, tgt)
                if isinstance(mod, torch.nn.Linear):
                    lora = LoRALinear(mod, r=r, alpha=alpha, dropout=dropout)
                    setattr(mlp, tgt, lora)
                    replaced += 1

    for n, p in hf.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad_(True)
    return total, layers_last, replaced


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # ---------------------------
    # Data & Model
    # ---------------------------
    loader = build_dataloader(args)
    sanity_check_first_batch(loader, num_classes=getattr(args, "v_act", 6), h_world=getattr(args, "h_world", 64))

    from types import SimpleNamespace as NS
    def make_default_cfg(args):
        H = 1024
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
                temperature=float(args.world_temp_start),
                label_smoothing=float(args.world_label_smoothing),
            ),
        )

    cfg = make_default_cfg(args)

    # 挂载动作不平衡配置
    prior_cli = _parse_floats_csv(args.class_prior) if (args.class_prior and args.class_prior.lower() != "auto") else None
    prior_auto = getattr(getattr(loader, "dataset", None), "_class_prior", None) if (args.class_prior and args.class_prior.lower() == "auto") else None
    if prior_cli is not None and len(prior_cli) != 6:
        raise ValueError("--class_prior 需要 6 个数字，对应 6 类")
    prior_use = prior_cli if (prior_cli is not None) else prior_auto

    cfg.action.use_la = bool(args.use_la)
    cfg.action.la_tau = float(args.la_tau)
    cfg.action.use_class_weights = bool(args.use_class_weights)
    cfg.action.use_focal = bool(args.use_focal)
    cfg.action.focal_gamma = float(args.focal_gamma)
    cfg.action.prior_reg_lambda = float(args.prior_reg_lambda)
    cfg.action.loss_last_only = bool(args.loss_last_only)
    if prior_use is not None:
        cfg.action.class_prior = prior_use
        cfg.action.log_prior = [math.log(max(1e-8, p)) for p in prior_use]
        print(f"[imbalance] using class_prior = {prior_use}")
    else:
        print("[imbalance] class_prior not provided; imbalance losses will use defaults if enabled.")

    model = OceanVLA(cfg).to(device)

    with torch.no_grad():
        try:
            model.gate_world_ctx.copy_(torch.tensor(0.3, device=device))
        except Exception:
            pass

    # warm-start
    if args.init_from and os.path.isfile(args.init_from):
        try:
            sd = torch.load(args.init_from, map_location="cpu")
            incompatible = model.load_state_dict(sd, strict=False)
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            print(f"[init] loaded: {args.init_from}, missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"[init][warn] failed to load init_from={args.init_from}: {e}")

    # 冻结/解冻策略
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    if args.unfreeze_last and args.unfreeze_last > 0:
        total, nlast = unfreeze_last_n_layers(model.backbone, args.unfreeze_last)
        print(f"[debug] unfreeze_last={nlast}/{total} decoder layers")

    if args.use_lora:
        target_names = [s.strip() for s in args.lora_targets.split(",") if s.strip()]
        total, lastL, replaced = apply_lora_to_last_layers(
            model.backbone,
            layers_last=args.lora_layers_last,
            target_names=target_names,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout
        )
        print(f"[debug] LoRA injected on last {lastL}/{total} layers, replaced linear modules={replaced}")

    if args.freeze_heads:
        for m in [model.action_head, model.world_head]:
            for p in m.parameters():
                p.requires_grad_(False)
        print("[debug] freeze_heads=True -> action_head/world_head parameters are frozen")

    # 统计可训练参数
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    print(f"[debug] trainable_params={trainable_params:,} / total_params={total_params:,} "
          f"({trainable_params/total_params*100:.2f}%), backbone_requires_grad="
          f"{any(p.requires_grad for p in model.backbone.parameters())}")

    # 统计 vision_encoder 可训练参数
    vision_trainable = sum(p.numel() for p in model.vision_encoder.parameters() if p.requires_grad)
    print(f"[debug] vision_encoder trainable params: {vision_trainable:,}")

    model.train()

    # ---------------------------
    # Optimizer + Scheduler
    # ---------------------------
    def build_param_groups(model, lr_backbone=3e-5, lr_heads=3e-4, lr_vision=1e-5, weight_decay=1e-2):
        """
        构建参数分组，确保每个参数只出现在一个组中
        """
        no_decay_keys = ("bias", "LayerNorm.weight", "layer_norm.weight",
                         "ln_", "norm.weight", "norm.bias", "embed", "embedding")
        
        # 用集合追踪已分配的参数，避免重复
        seen_ids = set()
        
        # 收集各类参数
        backbone_params = []
        vision_params = []
        heads_params = []  # action_head + world_head
        other_params = []  # 其他所有参数
        
        # 1. Vision encoder 参数（优先级最高）
        for n, p in model.vision_encoder.named_parameters():
            if p.requires_grad and id(p) not in seen_ids:
                vision_params.append((f"vision_encoder.{n}", p))
                seen_ids.add(id(p))
        
        # 2. Action head 和 World head 参数
        for n, p in model.action_head.named_parameters():
            if p.requires_grad and id(p) not in seen_ids:
                heads_params.append((f"action_head.{n}", p))
                seen_ids.add(id(p))
        
        for n, p in model.world_head.named_parameters():
            if p.requires_grad and id(p) not in seen_ids:
                heads_params.append((f"world_head.{n}", p))
                seen_ids.add(id(p))
        
        # 3. Backbone (LLM) 参数
        for n, p in model.backbone.named_parameters():
            if p.requires_grad and id(p) not in seen_ids:
                backbone_params.append((f"backbone.{n}", p))
                seen_ids.add(id(p))
        
        # 4. 其他所有参数
        for n, p in model.named_parameters():
            if p.requires_grad and id(p) not in seen_ids:
                other_params.append((n, p))
                seen_ids.add(id(p))
        
        # 分离 decay / no_decay
        def split_decay(named_params):
            with_decay, without_decay = [], []
            for n, p in named_params:
                if any(k in n for k in no_decay_keys):
                    without_decay.append(p)
                else:
                    with_decay.append(p)
            return with_decay, without_decay
        
        b_decay, b_no_decay = split_decay(backbone_params)
        v_decay, v_no_decay = split_decay(vision_params)
        h_decay, h_no_decay = split_decay(heads_params)
        o_decay, o_no_decay = split_decay(other_params)
        
        # 构建参数组（过滤空组）
        param_groups = []
        if b_decay:
            param_groups.append({"params": b_decay, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"})
        if b_no_decay:
            param_groups.append({"params": b_no_decay, "lr": lr_backbone, "weight_decay": 0.0, "name": "backbone_no_decay"})
        if o_decay:
            param_groups.append({"params": o_decay, "lr": lr_heads, "weight_decay": weight_decay, "name": "other_decay"})
        if o_no_decay:
            param_groups.append({"params": o_no_decay, "lr": lr_heads, "weight_decay": 0.0, "name": "other_no_decay"})
        if h_decay:
            param_groups.append({"params": h_decay, "lr": lr_heads, "weight_decay": weight_decay, "name": "heads_decay"})
        if h_no_decay:
            param_groups.append({"params": h_no_decay, "lr": lr_heads, "weight_decay": 0.0, "name": "heads_no_decay"})
        if v_decay:
            param_groups.append({"params": v_decay, "lr": lr_vision, "weight_decay": weight_decay, "name": "vision_decay"})
        if v_no_decay:
            param_groups.append({"params": v_no_decay, "lr": lr_vision, "weight_decay": 0.0, "name": "vision_no_decay"})
        
        # 打印分组信息
        for g in param_groups:
            n_params = sum(p.numel() for p in g["params"])
            print(f"[param_group] {g.get('name', 'unnamed'):20s}: {n_params:>12,} params, lr={g['lr']:.2e}")
        
        return param_groups

    optim = torch.optim.AdamW(
        build_param_groups(
            model, 
            lr_backbone=args.lr_backbone, 
            lr_heads=args.lr_heads, 
            lr_vision=args.lr_vision,
            weight_decay=args.weight_decay
        )
    )

    # 打印可训练参数统计
    num_trainable = sum(p.numel() for g in optim.param_groups for p in g["params"])
    num_total = sum(p.numel() for p in model.parameters())
    print(f"[debug] optimizer trainable_params={num_trainable:,} / total_params={num_total:,}")

    # Scheduler: warmup + cosine
    warmup = int(max(0, args.warmup_steps))
    steps_per_epoch = max(1, len(loader))
    inferred_total = int(args.epochs) * steps_per_epoch
    total_steps = int(args.max_steps) if args.max_steps and args.max_steps > 0 else inferred_total
    total_steps = max(total_steps, warmup + 1)
    min_mult = float(args.min_lr_mult)
    print(f"[sched] steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, warmup={warmup}")

    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        prog = (step - warmup) / float(max(1, total_steps - warmup))
        return min_mult + (1.0 - min_mult) * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # 头部阶段性冻结
    heads_warm_steps = int(args.freeze_heads_steps) if not args.freeze_heads else 0

    # AMP
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    amp_dtype = dtype_map[args.precision]
    use_scaler = bool(args.amp and args.precision == "fp16")
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    model.train()
    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            global_step += 1
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)

            # world logits 温度退火
            with torch.no_grad():
                T0, T1 = float(args.world_temp_start), float(args.world_temp_end)
                anneal = max(1, int(args.world_temp_anneal))
                frac = min(1.0, global_step / float(anneal))
                model.world_logit_temperature = T0 + (T1 - T0) * frac
                if hasattr(model, "world_label_smoothing"):
                    model.world_label_smoothing = float(args.world_label_smoothing)

            # WM 目标有效性检查
            wm = batch.get("wm_targets", None)
            if wm is not None and global_step % 200 == 0:
                valid = int((wm != -100).sum().item())
                total = int(wm.numel())
                ratio = valid / max(1, total)
                print(f"[check] valid_wm_tokens: {valid}/{total} = {ratio:.2%}")

            # 准备文本 labels
            labels = batch.get("labels", None)
            if labels is None:
                labels = batch["input_ids"].clone()
                labels[:, 0] = -100

            imgs = batch.get("images", batch.get("image", None))

            # 文本监督动态权重
            lambda_text_eff = float(args.lambda_text) if args.text_supervision else 0.0
            if lambda_text_eff > 0:
                valid_tok = int((labels != -100).sum().item())
                if valid_tok < int(args.min_text_tokens):
                    lambda_text_eff = 0.0

            # 前向
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=args.amp):
                lam_a = float(getattr(args, "lambda_action", 1.0))
                lam_w = float(getattr(args, "lambda_world", 1.0))

                model_inputs = dict(
                    input_ids=batch["input_ids"],
                    text_attention_mask=batch["text_attention_mask"],
                    images=imgs,
                    actions=None,
                    action_targets=batch["action_targets"],
                    wm_targets=batch["wm_targets"],
                    actions_prev=None,
                    lambda_action=lam_a,
                    lambda_world=lam_w,
                    labels=labels,
                    lambda_text=lambda_text_eff,
                )
                out = model(**_filter_kwargs_for(model.forward, model_inputs))
                loss = out["loss"]

            # 诊断
            if global_step % 50 == 0:
                valid_ratio = (model_inputs["action_targets"] != -100).float().mean().item()
                print(f"[debug] valid_action_ratio={valid_ratio:.3f}")

            # 预测直方图
            def _extract_action_logits_raw(out_dict):
                if isinstance(out_dict.get("action_logits", None), torch.Tensor):
                    return out_dict["action_logits"]
                act = out_dict.get("action", None)
                if isinstance(act, dict) and isinstance(act.get("logits", None), torch.Tensor):
                    return act["logits"]
                return None

            if global_step % 50 == 0:
                with torch.no_grad():
                    raw_logits = _extract_action_logits_raw(out)
                    if isinstance(raw_logits, torch.Tensor):
                        logits = raw_logits.detach().float()
                        V = logits.size(-1)
                        pred = logits.argmax(dim=-1).reshape(-1)
                        hist = torch.bincount(pred, minlength=V).float()
                        pred_hist = (hist / hist.sum().clamp_min(1)).cpu().numpy()
                        print(f"[debug] pred_hist_raw={np.round(pred_hist, 3).tolist()}")

            # 反传
            optim.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()
            scheduler.step()

            if global_step % 10 == 0:
                la_raw = _to_float(out.get("loss_a"), 0.0)
                lw_raw = _to_float(out.get("loss_w"), 0.0)
                lt_raw = _to_float(out.get("loss_t"), 0.0)
                loss_val = _to_float(loss, 0.0)
                
                # 获取各组学习率
                lr_info = {}
                for g in optim.param_groups:
                    name = g.get("name", "unnamed")
                    if "backbone" in name and "backbone" not in lr_info:
                        lr_info["backbone"] = g["lr"]
                    elif "vision" in name and "vision" not in lr_info:
                        lr_info["vision"] = g["lr"]
                    elif "heads" in name and "heads" not in lr_info:
                        lr_info["heads"] = g["lr"]
                    elif "other" in name and "other" not in lr_info:
                        lr_info["other"] = g["lr"]
                
                print(
                    f"[e{epoch} s{global_step}] "
                    f"loss={loss_val:.4f} "
                    f"(a={la_raw:.4f}x{lam_a}, w={lw_raw:.4f}x{lam_w})  "
                    f"LRs(back={lr_info.get('backbone', 0):.2e}, vis={lr_info.get('vision', 0):.2e}, heads={lr_info.get('heads', 0):.2e})"
                )

            if args.save_every > 0 and global_step % args.save_every == 0:
                ckpt = os.path.join(args.save_dir, f"oceanvla_step{global_step}.pt")
                torch.save(model.state_dict(), ckpt)
                print(f"Saved checkpoint -> {ckpt}")

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # 保存最终权重
    final_ckpt = os.path.join(args.save_dir, "oceanvla_final.pt")
    os.makedirs(os.path.dirname(final_ckpt), exist_ok=True)
    torch.save(model.state_dict(), final_ckpt)
    dt = time.time() - t0
    print(f"Done. Saved final checkpoint -> {final_ckpt} (elapsed {dt/3600:.2f} h)")


if __name__ == "__main__":
    main()
