# core/data/datasets/ocean_dataset.py
"""
OceanVLADataset: PyTorch Dataset for OceanVLA training and evaluation.

Reads samples from a JSONL file. Each line contains paths to:
  - An RGB image (onboard camera frame)
  - A natural-language task instruction
  - Continuous action vectors
  - World model VQ token targets
  - Discrete 6-class action labels (0=FORWARD, 1=STOP, 2=TURNLEFT,
                                     3=TURNRIGHT, 4=ACCELERATE, 5=DECELERATE)

Temporal mode (temporal_enabled=True):
  Returns 'images' [T, 3, H, W] by looking back T frames in the same log.
  Frames are padded with the first available frame when history is insufficient.
  Single-frame mode returns 'image' [3, H, W] for backward compatibility.

Data statistics (Section 5.1.3):
  Collision avoidance: 25,416 frames
    FORWARD: 58.2%, TURNLEFT: 12.8%, TURNRIGHT: 11.5%,
    DECELERATE: 9.3%, ACCELERATE: 5.8%, STOP: 2.4%
  Target tracking: 17,955 frames (more balanced distribution)
  Total: 43,371 labeled frames; split 70/15/15 at episode level.
"""

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as TV

from transformers import AutoTokenizer  # Used by callers; keep import for API consistency

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Grid inference utilities
# ---------------------------------------------------------------------------

def _infer_src_hw_from_len(L: int) -> Tuple[int, int]:
    """
    Infer a near-square (h, w) grid such that h * w == L.
    Falls back to (L, 1) if no factor pair is found.
    """
    if L <= 0:
        return (1, 1)
    r = int(math.sqrt(L))
    for h in range(r, 0, -1):
        if L % h == 0:
            return (h, L // h)
    return (L, 1)


def _tgt_hw_from_hworld(N: int) -> Tuple[int, int]:
    """Infer a near-square target grid (tgt_h, tgt_w) satisfying tgt_h * tgt_w == N."""
    if N <= 0:
        return (1, 1)
    r = int(math.sqrt(N))
    for h in range(r, 0, -1):
        if N % h == 0:
            return (h, N // h)
    return (N, 1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OceanVLADataset(Dataset):
    """
    General-purpose VLA dataset for maritime USV control.

    Supports both single-frame and temporal (multi-frame) input modes.
    Text supervision can be enabled to add autoregressive language targets.

    Key design choices (aligned with Section 5.1.3):
      - Episode-level train/val/test split (no frame leakage between sets)
      - 12 environmental conditions: 3 sea states × 4 illumination conditions
      - Caption diversity via synonym replacement and structural randomization
    """

    def __init__(
        self,
        root: str,
        jsonl: str,
        tokenizer,
        img_size: int = 224,
        wm_src_hw: Tuple[int, int] = (64, 64),
        wm_tgt_hw: Optional[Tuple[int, int]] = None,
        wm_pool: str = "mode",        # "mode" (majority vote) | "center" (center pixel)

        # Action / world model hyperparameters
        k: int = 4,                   # Action chunk size K
        a_dim: int = 7,               # Continuous action dimension
        h_world: int = 1024,          # World token sequence length
        v_world: int = 16384,         # World vocabulary size (upper bound)
        use_random_wm: bool = False,  # Whether to use random placeholder WM targets

        # Field name mapping (adjust if your JSONL uses different keys)
        image_key: str = "image",
        text_key: str = "text",
        actions_key: Optional[str] = "actions",
        wm_key: Optional[str] = "wm_targets",

        # 6-class discrete action label configuration
        action6_key: Optional[str] = "action_targets_npy",
        action6_root: Optional[str] = None,
        action6_template: Optional[str] = None,
        logid_key: str = "meta.log_id",
        group_index_key: str = "meta.group_index",
        group_size_key: str = "meta.group_size",

        # Image preprocessing
        use_clip_processor: bool = True,
        clip_path: str = "/root/autodl-tmp/models--openai--clip-vit-base-patch32",

        # Temporal configuration
        temporal_enabled: bool = False,
        num_frames: int = 1,              # T: number of frames per sample (including current)
        stride: int = 1,                  # Temporal stride between frames
        frame_regex: str = r"frame[_\-]?(\d+)",

        # Text supervision
        text_supervision: bool = True,
        min_text_tokens: int = 4,
    ):
        super().__init__()
        self.root = root
        self.tok = tokenizer

        self.k = int(k)
        self.a_dim = int(a_dim)
        self.h_world = int(h_world)
        self.v_world = int(v_world)
        self.use_random_wm = bool(use_random_wm)

        self.wm_src_hw = tuple(wm_src_hw)
        self.wm_tgt_hw = tuple(wm_tgt_hw) if wm_tgt_hw is not None else None
        self.wm_pool   = str(wm_pool)

        self.image_key   = image_key
        self.text_key    = text_key
        self.actions_key = actions_key
        self.wm_key      = wm_key

        self.action6_key      = action6_key
        self.action6_root     = action6_root or root
        self.action6_template = action6_template
        self.logid_key        = logid_key
        self.group_index_key  = group_index_key
        self.group_size_key   = group_size_key

        self.use_clip_processor = bool(use_clip_processor)
        self.clip_path = clip_path
        self.img_size  = img_size

        self.clip_processor = None
        self.tfm = None

        self.temporal_enabled = bool(temporal_enabled)
        self.T       = int(max(1, num_frames))
        self.stride  = int(max(1, stride))
        self.frame_pat = re.compile(frame_regex)

        self.text_supervision = bool(text_supervision)
        self.min_text_tokens  = int(min_text_tokens)

        if self.use_clip_processor:
            try:
                from transformers import AutoImageProcessor
                self.clip_processor = AutoImageProcessor.from_pretrained(self.clip_path)
                print(f"[OceanVLADataset] Loaded CLIP processor from: {self.clip_path}")
            except Exception as e1:
                print(f"[OceanVLADataset] CLIP processor failed ({e1}), falling back to manual transforms.")
                self.clip_processor = None

        if self.clip_processor is None:
            # Standard ImageNet normalization (CLIP uses the same statistics)
            self.tfm = TV.Compose([
                TV.Resize((img_size, img_size)),
                TV.ToTensor(),
                TV.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            ])

        # Load JSONL index
        jsonl_path = jsonl if os.path.isabs(jsonl) else os.path.join(root, jsonl)
        self.items: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    import json
                    self.items.append(json.loads(line))

        # Build frame-index lookup for temporal sampling
        self._frame_index: Dict[str, List[int]] = {}
        if self.temporal_enabled:
            self._build_frame_index()

        print(f"[OceanVLADataset] Loaded {len(self.items)} samples from {jsonl_path}")

    # ------------------------------------------------------------------

    def _build_frame_index(self):
        """
        Build a mapping from log_id to sorted list of sample indices for
        efficient temporal neighborhood lookup.
        """
        import json
        for i, item in enumerate(self.items):
            lid = self._get_nested(item, self.logid_key, default=f"__log_{i}")
            self._frame_index.setdefault(str(lid), []).append(i)
        for lid in self._frame_index:
            self._frame_index[lid].sort(key=lambda idx: self._parse_frame_num(self.items[idx]))

    def _parse_frame_num(self, item: Dict) -> int:
        img_path = item.get(self.image_key, "")
        m = self.frame_pat.search(os.path.basename(str(img_path)))
        return int(m.group(1)) if m else 0

    def _get_nested(self, d: Dict, key: str, default=None):
        """Dot-notation nested dictionary access."""
        parts = key.split(".")
        cur = d
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur

    def _load_image(self, img_path: str) -> torch.Tensor:
        """Load and preprocess a single RGB image to [3, H, W]."""
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.root, img_path)
        img = Image.open(img_path).convert("RGB")
        if self.clip_processor is not None:
            processed = self.clip_processor(images=img, return_tensors="pt")
            return processed["pixel_values"][0]  # [3, H, W]
        return self.tfm(img)  # [3, H, W]

    def _load_action6(self, item: Dict) -> torch.Tensor:
        """
        Load 6-class discrete action labels from a .npy file.
        Returns a tensor of shape [K] with values in {0..5} or IGNORE_INDEX.
        """
        rel = None
        if self.action6_key:
            rel = item.get(self.action6_key, None)
            if rel is None:
                rel = item.get("action_targets_npy", None)

        if rel:
            path = rel if os.path.isabs(rel) else os.path.join(self.action6_root, rel)
            if os.path.isfile(path):
                try:
                    arr = np.load(path, allow_pickle=False).reshape(-1)
                    arr = arr[:self.k]
                    tgt = torch.full((self.k,), IGNORE_INDEX, dtype=torch.long)
                    tgt[:len(arr)] = torch.from_numpy(arr.astype(np.int64))
                    return tgt
                except Exception:
                    pass

        return torch.full((self.k,), IGNORE_INDEX, dtype=torch.long)

    def _load_wm_targets(self, item: Dict) -> torch.Tensor:
        """Load world model VQ token targets. Returns IGNORE_INDEX fill if unavailable."""
        rel = item.get(self.wm_key, None) if self.wm_key else None
        if rel:
            path = rel if os.path.isabs(rel) else os.path.join(self.root, rel)
            if os.path.isfile(path):
                try:
                    arr = np.load(path, allow_pickle=False).reshape(-1)
                    # Pool from source grid to target size
                    tgt = self._pool_wm(arr)
                    return torch.from_numpy(tgt.astype(np.int64))
                except Exception:
                    pass

        if self.use_random_wm:
            return torch.randint(0, self.v_world, (self.h_world,))
        return torch.full((self.h_world,), IGNORE_INDEX, dtype=torch.long)

    def _pool_wm(self, arr: np.ndarray) -> np.ndarray:
        """Pool a flat VQ token array from src grid to target size h_world."""
        src_h, src_w = self.wm_src_hw
        if self.wm_tgt_hw is not None:
            tgt_h, tgt_w = self.wm_tgt_hw
        else:
            tgt_h, tgt_w = _tgt_hw_from_hworld(self.h_world)

        arr_2d = arr.reshape(src_h, src_w)
        block_h = src_h // tgt_h
        block_w = src_w // tgt_w
        if block_h < 1:
            block_h = 1
        if block_w < 1:
            block_w = 1

        out = np.zeros((tgt_h, tgt_w), dtype=np.int64)
        for i in range(tgt_h):
            for j in range(tgt_w):
                block = arr_2d[i * block_h:(i + 1) * block_h,
                               j * block_w:(j + 1) * block_w].reshape(-1)
                if self.wm_pool == "center":
                    out[i, j] = block[len(block) // 2]
                else:  # mode (majority vote)
                    vals, cnts = np.unique(block, return_counts=True)
                    out[i, j] = vals[np.argmax(cnts)]
        return out.reshape(-1)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        # ---- Image(s) ----
        img_path = item.get(self.image_key, "")
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.root, img_path)

        current_frame = self._load_image(img_path)  # [3, H, W]

        if self.temporal_enabled and self.T > 1:
            # Build T-frame temporal sequence ending at the current frame
            frames = self._load_temporal_frames(item, idx, current_frame)
            result_images = frames  # [T, 3, H, W]
        else:
            result_images = None  # single-frame mode

        # ---- Text instruction ----
        text = item.get(self.text_key, "")
        encoded = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        input_ids = encoded["input_ids"][0]   # [L]

        if self.text_supervision and input_ids.size(0) >= self.min_text_tokens:
            labels = input_ids.clone()
        else:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # ---- Continuous actions (context) ----
        acts_raw = item.get(self.actions_key, None) if self.actions_key else None
        if acts_raw is not None:
            arr = np.array(acts_raw, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1).repeat(self.k, axis=0)
            arr = arr[:self.k, :self.a_dim]
            if arr.shape[0] < self.k:
                pad = np.zeros((self.k - arr.shape[0], self.a_dim), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            actions = torch.from_numpy(arr)
        else:
            actions = torch.zeros(self.k, self.a_dim)

        # ---- Discrete action targets (0..5) ----
        action_targets = self._load_action6(item)  # [K]

        # ---- World model targets ----
        wm_targets = self._load_wm_targets(item)  # [h_world]

        out: Dict[str, Any] = {
            "input_ids":      input_ids,
            "labels":         labels,
            "image":          current_frame,   # [3, H, W] — always present
            "actions":        actions,         # [K, A_dim]
            "action_targets": action_targets,  # [K]
            "wm_targets":     wm_targets,      # [h_world]
        }
        if result_images is not None:
            out["images"] = result_images      # [T, 3, H, W]

        return out

    def _load_temporal_frames(
        self,
        item: Dict,
        idx: int,
        current_frame: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a T-frame sequence [T, 3, H, W] ending at the current frame.
        Earlier frames are filled from the same log; missing history uses the
        first available frame (left-pad strategy).
        """
        log_id = str(self._get_nested(item, self.logid_key, default=f"__log_{idx}"))
        log_indices = self._frame_index.get(log_id, [idx])

        try:
            pos = log_indices.index(idx)
        except ValueError:
            pos = 0

        frames = []
        for step in range(self.T - 1, -1, -1):
            src_pos = pos - step * self.stride
            if src_pos < 0:
                src_pos = 0
            src_idx = log_indices[src_pos]
            src_item = self.items[src_idx]
            src_path = src_item.get(self.image_key, "")
            if not os.path.isabs(src_path):
                src_path = os.path.join(self.root, src_path)
            try:
                frames.append(self._load_image(src_path))
            except Exception:
                frames.append(current_frame)

        return torch.stack(frames, dim=0)  # [T, 3, H, W]
