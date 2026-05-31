# core/data/datasets/collate.py
"""
Collate function for OceanVLA DataLoader.

Assembles heterogeneous per-sample dicts into a batch dict with:
  - Padded text token sequences (right-pad with tokenizer pad_token_id)
  - Stacked images: [B, 3, H, W] (single-frame) or [B, T, 3, H, W] (temporal)
  - Stacked action tensors, targets, and world-model targets
  - A full attention mask covering [text | image_token | action_tokens | world_tokens]

This mask corresponds to the sequence layout used in OceanVLA.forward():
  Z_input = [text] ++ [1 image token] ++ [K action tokens] ++ [N world tokens]
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pad_1d_right(x: torch.Tensor, L: int, pad_val: int) -> torch.Tensor:
    """Right-pad a 1-D tensor to length L."""
    if x.size(0) == L:
        return x
    return F.pad(x, (0, L - x.size(0)), value=pad_val)


def _stack_text(tokenizer, input_ids_list, labels_list):
    """
    Right-pad text sequences to the maximum length in the batch.

    Returns:
        input_ids:  [B, L_max]
        labels:     [B, L_max]  (pad positions filled with IGNORE_INDEX)
        text_mask:  [B, L_max]  (1 for real tokens, 0 for padding)
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0

    L = max(t.size(0) for t in input_ids_list)
    input_ids = torch.stack(
        [_pad_1d_right(t, L, pad_id) for t in input_ids_list], dim=0
    )
    labels = torch.stack(
        [_pad_1d_right(lbl, L, IGNORE_INDEX) for lbl in labels_list], dim=0
    )
    text_mask = (input_ids != pad_id).long()
    return input_ids, labels, text_mask


def _stack_images(batch: List[Dict[str, Any]]):
    """
    Handle three cases:
      1. All single-frame: 'image' [3, H, W]   → output [B, 3, H, W], seq_len=None
      2. All temporal:     'images' [T, 3, H, W] → output [B, T, 3, H, W], seq_len=T
      3. Mixed:            unsqueeze single-frames to T=1, pad shorter sequences to max T
    """
    has_seq  = all("images" in it for it in batch)
    none_seq = all("images" not in it for it in batch)

    if none_seq:
        images = torch.stack([it["image"] for it in batch], dim=0)
        return images, None

    if has_seq:
        T_list = [it["images"].shape[0] for it in batch]
        if len(set(T_list)) != 1:
            raise ValueError(
                f"All temporal samples in a batch must have the same T, got {T_list}"
            )
        images = torch.stack([it["images"] for it in batch], dim=0)
        return images, T_list[0]

    # Mixed: normalize everything to temporal format
    images_seq, T_list = [], []
    for it in batch:
        if "images" in it:
            images_seq.append(it["images"])
            T_list.append(it["images"].shape[0])
        else:
            images_seq.append(it["image"].unsqueeze(0))
            T_list.append(1)

    if len(set(T_list)) != 1:
        # Pad shorter sequences by repeating the last frame
        Tmax = max(T_list)
        padded = []
        for img in images_seq:
            if img.shape[0] < Tmax:
                pad = img[-1:].repeat(Tmax - img.shape[0], 1, 1, 1)
                img = torch.cat([img, pad], dim=0)
            padded.append(img)
        images_seq = padded
        T = Tmax
    else:
        T = T_list[0]

    images = torch.stack(images_seq, dim=0)
    return images, T


def _stack_optional_actions_prev(
    batch: List[Dict[str, Any]],
) -> Optional[torch.Tensor]:
    """
    Stack 'actions_prev' tensors if all samples contain them; otherwise return None.
    Supports both 1-D (discrete) and 2-D (continuous / one-hot) action formats.
    """
    if not all("actions_prev" in it for it in batch):
        return None

    ex = batch[0]["actions_prev"]
    if not isinstance(ex, torch.Tensor):
        ex = torch.as_tensor(ex)

    tensors = []
    for it in batch:
        t = it["actions_prev"]
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        tensors.append(t)

    stacked = torch.stack(tensors, dim=0)
    if stacked.dtype.is_floating_point:
        return stacked.float()
    return stacked.long()


# ---------------------------------------------------------------------------
# Public collate_fn
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer,
    k_act: Optional[int] = None,
    h_world: Optional[int] = None,
    add_image_token: bool = True,
) -> Dict[str, Any]:
    """
    Collate a list of per-sample dicts into a batch dict.

    Attention mask layout:
      [text tokens (L)] + [image token (1)] + [action tokens (K)] + [world tokens (N)]

    Args:
        batch:           List of dicts from OceanVLADataset.__getitem__.
        tokenizer:       HuggingFace tokenizer (needed for pad_token_id).
        k_act:           Fallback action chunk size if not inferable from tensors.
        h_world:         Fallback world token count if not inferable from tensors.
        add_image_token: Whether to include the 1-token image slot in the mask.
    Returns:
        Batch dict with keys:
          input_ids, labels, text_attention_mask, images,
          actions, action_targets, wm_targets, [actions_prev]
    """
    # Text
    input_ids, labels, text_mask = _stack_text(
        tokenizer,
        [it["input_ids"] for it in batch],
        [it["labels"] for it in batch],
    )

    # Images
    images, _seq_len = _stack_images(batch)

    # Tensors
    actions        = torch.stack([it["actions"]        for it in batch], dim=0)
    action_targets = torch.stack([it["action_targets"] for it in batch], dim=0)
    wm_targets     = torch.stack([it["wm_targets"]     for it in batch], dim=0)
    actions_prev   = _stack_optional_actions_prev(batch)

    # Sequence lengths
    K = actions.size(1) if actions.ndim == 3 else int(k_act or 0)
    N = wm_targets.size(1) if wm_targets.ndim == 2 else int(h_world or 0)
    if K <= 0 or N <= 0:
        raise ValueError(
            f"Invalid sequence lengths: K={K}, N={N}. "
            "Check the shapes of 'actions' and 'wm_targets' or provide k_act/h_world."
        )

    B = input_ids.size(0)
    img_tok   = 1 if add_image_token else 0
    extra_len = img_tok + K + N
    extra_mask = torch.ones((B, extra_len), dtype=torch.long)
    attention_mask = torch.cat([text_mask, extra_mask], dim=1)

    out: Dict[str, Any] = {
        "input_ids":          input_ids,         # [B, L]
        "labels":             labels,             # [B, L]
        "text_attention_mask": attention_mask,    # [B, L + 1 + K + N]
        "images":             images,             # [B, 3, H, W] or [B, T, 3, H, W]
        "actions":            actions,            # [B, K, A_dim]
        "action_targets":     action_targets,     # [B, K]
        "wm_targets":         wm_targets,         # [B, N]
    }
    if actions_prev is not None:
        out["actions_prev"] = actions_prev

    return out
