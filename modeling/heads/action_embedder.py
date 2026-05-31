# core/modeling/heads/action_embedder.py
"""
ActionEmbedder: encodes continuous action chunks into token embeddings
with learnable positional embeddings.

Used to embed the action history / context into the LLM sequence
before the transformer policy forward pass.

Input:  actions [B, K, A_dim]  — continuous action vectors (or one-hot)
Output: [B, K, H]              — projected and position-encoded embeddings
"""

import torch
import torch.nn as nn


class ActionEmbedder(nn.Module):
    """
    Two-layer SiLU MLP followed by learnable positional embeddings.

    Args:
        in_dim: Input action dimension A_dim (e.g., 6 for discrete or 7 for continuous).
        hidden: Output dimension H (must match transformer hidden dim).
        K:      Action chunk size (number of action steps per forward pass).
    """

    def __init__(self, in_dim: int, hidden: int, K: int):
        super().__init__()
        self.K = K
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.pos = nn.Parameter(torch.randn(1, K, hidden))

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: [B, K, A_dim]
        Returns:
            [B, K, H]
        """
        if actions.dim() != 3 or actions.size(1) != self.K:
            raise ValueError(
                f"ActionEmbedder: expected [B, {self.K}, A_dim], got {actions.shape}"
            )
        x = self.proj(actions)  # [B, K, H]
        return x + self.pos
