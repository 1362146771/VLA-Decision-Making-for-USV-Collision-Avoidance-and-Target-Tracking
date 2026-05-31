# core/modeling/vision_encoders.py
"""
Maritime-Adapted CLIP Vision-Language Encoder (Section 4.2).

Base architecture: CLIP ViT-L/14
  - Vision encoder: 24 layers, D_v=1024, N=256 patch tokens (14×14 patches on 224×224 input)
  - Text encoder:   12 layers, D_t=512
  - Total vision parameters: 307 M

Maritime adaptation:
  - Fine-tuned on 15 K maritime image-text pairs (7 K public + 8 K synthetic)
  - Contrastive loss (Eq. 3) with temperature τ=0.07
  - First 12 layers frozen; final 12 layers fine-tuned at lr=1e-5 for 10 epochs

Pooling modes:
  'projected' — use CLIPModel.get_image_features() (post-projection, D=projection_dim)
  'cls'       — CLS token from vision_model (D=hidden_size)
  'avg'       — average of patch tokens, excluding CLS (D=hidden_size)
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import CLIPModel
except Exception:
    CLIPModel = None


class CLIPVisionTower(nn.Module):
    """
    CLIP vision tower that returns a single pooled feature vector [B, D].

    In OceanVLA the full patch token sequence is extracted separately inside
    OceanVLA._encode_vision_with_patches(); this module handles the pooled
    (global) representation used as the image token in the LLM sequence.

    Args:
        name:      HuggingFace model ID or local path.
                   Paper uses 'openai/clip-vit-large-patch14' (ViT-L/14, 307 M params).
        pool:      Pooling strategy ('projected' | 'cls' | 'avg').
        out_dim:   If not None and differs from pooled dimension, adds a linear projection.
        normalize: If True, L2-normalize the output (useful for contrastive tasks).
    """

    def __init__(
        self,
        name: str = "openai/clip-vit-large-patch14",
        pool: Literal["projected", "cls", "avg"] = "projected",
        out_dim: Optional[int] = None,
        normalize: bool = False,
    ):
        super().__init__()
        if CLIPModel is None:
            raise ImportError(
                "transformers is not installed. Please run: pip install transformers"
            )

        self.clip = CLIPModel.from_pretrained(name)
        self.pool = pool
        self.normalize = normalize

        # Determine native output dimension before optional projection
        d_proj = getattr(self.clip.config, "projection_dim", None)
        d_hid  = getattr(self.clip.vision_model.config, "hidden_size", None)

        if pool == "projected":
            if d_proj is None:
                raise ValueError(
                    "CLIP config does not contain projection_dim. "
                    "Use pool='cls' or pool='avg' instead."
                )
            in_dim = d_proj
        else:
            if d_hid is None:
                raise ValueError("Cannot determine hidden_size from vision_model.config.")
            in_dim = d_hid

        self.project: Optional[nn.Linear] = None
        if out_dim is not None and out_dim != in_dim:
            self.project = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.xavier_uniform_(self.project.weight)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: FloatTensor [B, 3, H, W], pre-processed with CLIP normalization.
        Returns:
            feat:   FloatTensor [B, D] where D = out_dim or the native pooled dimension.
        """
        if self.pool == "projected":
            try:
                feat = self.clip.get_image_features(pixel_values=images)
            except TypeError:
                feat = self.clip.get_image_features(images)
        else:
            try:
                out = self.clip.vision_model(
                    pixel_values=images,
                    output_hidden_states=False,
                    return_dict=True,
                )
            except TypeError:
                out = self.clip.vision_model(
                    pixel_values=images,
                    output_hidden_states=False,
                )

            if hasattr(out, "last_hidden_state"):
                hs = out.last_hidden_state   # [B, 1+N, hidden]
            elif isinstance(out, tuple):
                hs = out[0]
            else:
                raise RuntimeError(
                    f"Unexpected output type from vision_model: {type(out)}"
                )

            if self.pool == "cls":
                feat = hs[:, 0, :]           # CLS token
            elif self.pool == "avg":
                feat = hs[:, 1:, :].mean(dim=1)  # mean over patch tokens
            else:
                raise ValueError(f"Unknown pool mode: {self.pool!r}")

        if self.project is not None:
            feat = self.project(feat)

        if self.normalize:
            feat = F.normalize(feat, dim=-1)

        return feat  # [B, D]


# ---------------------------------------------------------------------------
# Maritime CLIP contrastive fine-tuning loss (Eq. 3)
# ---------------------------------------------------------------------------

def maritime_clip_loss(
    vision_embeds: torch.Tensor,   # [B, D] L2-normalized image embeddings
    text_embeds: torch.Tensor,     # [B, D] L2-normalized text embeddings
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric contrastive loss used for maritime domain adaptation (Eq. 3).

    L_CLIP = -1/(2B) * sum_i [ log(exp(S_ii/τ) / sum_j exp(S_ij/τ))
                               + log(exp(S_ii/τ) / sum_j exp(S_ji/τ)) ]

    where S_ij = cos(f_vision(I_i), f_text(x_j)) and τ=0.07.

    Args:
        vision_embeds: per-image embeddings, already L2-normalized.
        text_embeds:   per-caption embeddings, already L2-normalized.
        temperature:   learnable temperature parameter (initialized to 0.07).
    """
    B = vision_embeds.size(0)
    # S_ij = cosine similarity matrix [B, B]
    S = (vision_embeds @ text_embeds.T) / temperature

    labels = torch.arange(B, device=S.device)
    loss_i2t = F.cross_entropy(S, labels)          # image-to-text
    loss_t2i = F.cross_entropy(S.T, labels)        # text-to-image
    return (loss_i2t + loss_t2i) / 2.0
