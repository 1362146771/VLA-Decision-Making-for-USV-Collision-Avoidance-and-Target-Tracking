# core/modeling/heads/world_prefix.py
"""
WorldPrefix: projects the combined image + action context into N world-model tokens
that are prepended to the world-token teacher-forcing sequence.

This module provides the world model context vector H (Section 4.4):
  world_ctx = WorldPrefix(hist_feat, act_tokens)  — [B, N, H]

The LSTM compresses [img_feat; act_tokens] into a fixed-length recurrent state,
which is then expanded (via a channel-wise linear projection) to N tokens with
learnable positional embeddings.
"""

import torch
import torch.nn as nn


class WorldPrefix(nn.Module):
    """
    Args:
        hidden:  Transformer hidden dimension H.
        wm_dim:  World model latent dimension D_h (256 in the paper).
        N:       Number of world-model context tokens (seq_len from config).
    """

    def __init__(self, hidden: int, wm_dim: int, N: int):
        super().__init__()
        self.N = N
        # Compress [img_feat; act_tokens] sequence via LSTM
        self.core = nn.LSTM(
            input_size=hidden,
            hidden_size=wm_dim,
            num_layers=1,
            batch_first=True,
        )
        self.to_hidden = nn.Linear(wm_dim, hidden)

        # Expand the aggregated representation from 1 token to N tokens
        self.len_proj = nn.Linear(1, N, bias=False)
        self.pos_world = nn.Parameter(torch.randn(1, N, hidden) * 0.01)

    def forward(
        self,
        img_feat: torch.Tensor,    # [B, H]  — pooled visual feature
        act_tokens: torch.Tensor,  # [B, K, H] — action embeddings
    ) -> torch.Tensor:
        """
        Returns:
            world_ctx: [B, N, H]  — context tokens for the world model sequence
        """
        B, K, H = act_tokens.shape
        # Concatenate image feature and action tokens along sequence dim
        seq = torch.cat([img_feat.unsqueeze(1), act_tokens], dim=1)  # [B, 1+K, H]
        z, _ = self.core(seq)       # [B, 1+K, wm_dim]
        z = self.to_hidden(z)       # [B, 1+K, H]

        # Aggregate along time then expand to N tokens
        zT  = z.transpose(1, 2)                    # [B, H, 1+K]
        z1  = zT.mean(dim=-1, keepdim=True)        # [B, H, 1]
        zN  = self.len_proj(z1)                    # [B, H, N]
        zN  = zN.transpose(1, 2) + self.pos_world  # [B, N, H]
        return zN
