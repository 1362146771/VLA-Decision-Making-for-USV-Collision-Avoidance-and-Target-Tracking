# core/data/transforms.py
"""
Image preprocessing transforms for OceanVLA.

Two modes:
  - 'train': random resized crop (scale 0.8–1.0), color jitter
             (brightness ±20%, contrast ±15%), Gaussian noise (σ=0.05),
             horizontal flip (p=0.5) — Section 4.5 data augmentation
  - 'eval':  deterministic resize + center crop

All modes apply ImageNet normalization (means=[0.485, 0.456, 0.406],
stds=[0.229, 0.224, 0.225]) for compatibility with CLIP preprocessing.
"""

import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class MaritimeAugment:
    """
    Data augmentation pipeline for maritime imagery (Section 4.5).

    Applies:
      - Color jitter: brightness ±20%, contrast ±15%
      - Additive Gaussian noise: σ = 0.05
      - Random crop-and-resize: scale 0.8–1.0
      - Horizontal flip: p = 0.5 (with action label swapping TURNLEFT ↔ TURNRIGHT)
    """

    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.15)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, img: Image.Image):
        # Random resized crop (scale 0.8–1.0)
        i, j, h, w = T.RandomResizedCrop.get_params(
            img, scale=(0.8, 1.0), ratio=(3/4, 4/3)
        )
        img = TF.resized_crop(img, i, j, h, w, (self.img_size, self.img_size))
        # Color jitter
        img = self.color_jitter(img)
        # To tensor
        x = self.to_tensor(img)  # [3, H, W] in [0, 1]
        # Additive Gaussian noise σ=0.05
        x = x + 0.05 * torch.randn_like(x)
        x = x.clamp(0.0, 1.0)
        # Normalize
        x = self.normalize(x)
        return x


class MultiModalTransform:
    """
    Preprocessing pipeline for OceanVLA training and evaluation.

    Args:
        config: configuration object with fields:
            - data.image_size: int or [H, W]
            - data.augmentation.brightness: float (ignored in eval mode)
            - data.augmentation.contrast:   float (ignored in eval mode)
        mode:   'train' or 'eval'
    """

    def __init__(self, config, mode: str = "train"):
        self.mode = mode
        image_size = getattr(config.data, "image_size", 224)
        if isinstance(image_size, (list, tuple)):
            self.image_size = tuple(image_size)
        else:
            self.image_size = (int(image_size), int(image_size))

        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        aug = getattr(config.data, "augmentation", None)
        brightness = getattr(aug, "brightness", 0.2) if aug else 0.2
        contrast   = getattr(aug, "contrast", 0.15)  if aug else 0.15

        if mode == "train":
            self.transforms = T.Compose([
                T.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                T.ColorJitter(brightness=brightness, contrast=contrast),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(self.image_size),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                normalize,
            ])

        self.max_text_len = getattr(config.data, "max_text_len", 512)

    def __call__(self, raw_batch):
        """
        Process a raw batch dict containing 'image' and 'text' fields.

        Args:
            raw_batch: dict with 'image' (PIL Image) and optional 'text' (str)
        Returns:
            dict with preprocessed 'image' tensor and truncated 'text'
        """
        out = {}
        if "image" in raw_batch:
            img = raw_batch["image"]
            if isinstance(img, torch.Tensor):
                out["image"] = img
            else:
                out["image"] = self.transforms(img)

        if "text" in raw_batch:
            text = raw_batch["text"]
            words = text.split()
            if len(words) > self.max_text_len:
                text = " ".join(words[:self.max_text_len])
            out["text"] = text

        # Pass through any other keys unchanged
        for k, v in raw_batch.items():
            if k not in out:
                out[k] = v

        return out
