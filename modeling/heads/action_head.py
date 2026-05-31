# core/modeling/heads/action_head.py
"""
MLP Action Head (Section 4.4, Eq. 14).

Maps the [ACTION] token hidden state h_act ∈ R^d to a probability distribution
over the six discrete maritime maneuvers:

  p(a_t | I_{≤t}, x) = softmax(W2 * ReLU(W1 * h_act + b1) + b2)

where:
  W1 ∈ R^{d × d'}, b1 ∈ R^{d'}, d' = 256
  W2 ∈ R^{d' × |A|}, b2 ∈ R^{|A|}, |A| = 6

Action space |A| = 6:
  0: FORWARD      1: STOP
  2: TURNLEFT     3: TURNRIGHT
  4: ACCELERATE   5: DECELERATE

At inference, greedy decoding selects the action with the highest probability (Eq. 15):
  a_t = argmax_{a ∈ A} p(a | I_{≤t}, x)
"""

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    """
    Two-layer MLP action head.

    Input:  x [B, K, H]  — hidden states from the transformer policy
    Output: logits [B, K, V]  where V = discrete action vocabulary size (6)

    Config requirements:
        cfg.hidden_size  — must match backbone hidden dim H (set by OceanVLA.__init__)
        cfg.discrete_dim — action vocabulary size V (should be 6)
        cfg.mlp_hidden   — optional intermediate dimension (default: H, paper uses 256)
        cfg.dropout      — optional dropout probability (default: 0)
    """

    def __init__(self, cfg):
        super().__init__()
        H = getattr(cfg, "hidden_size", None)
        V = getattr(cfg, "discrete_dim", None)
        if H is None:
            raise ValueError("ActionHead: cfg.hidden_size must be set.")
        if V is None:
            raise ValueError("ActionHead: cfg.discrete_dim must be set.")

        # Paper specifies d' = 256 (intermediate dimension)
        mlp_hidden = int(getattr(cfg, "mlp_hidden", 256))
        dropout_p  = float(getattr(cfg, "dropout", 0.0))

        self.hidden_size = H
        self.vocab_size  = V

        # Eq. 14: h_act → ReLU → logits
        layers = [
            nn.Linear(H, mlp_hidden),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(mlp_hidden, V))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, K, H]  — token hidden states from transformer policy
        Returns:
            logits: [B, K, V]  — unnormalized action log-probabilities
        """
        B, K, H = x.shape
        assert H == self.hidden_size, (
            f"ActionHead: input dim {H} != cfg.hidden_size {self.hidden_size}"
        )
        y = self.mlp(x.reshape(B * K, H))      # [B*K, V]
        return y.reshape(B, K, self.vocab_size)  # [B, K, V]
