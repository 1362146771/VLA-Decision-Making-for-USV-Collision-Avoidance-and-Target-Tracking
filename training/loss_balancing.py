# core/training/loss_balancing.py
"""
AdaptiveLoss: learnable multi-task loss weighting via uncertainty.

Implements homoscedastic uncertainty weighting for combining the
action prediction loss and world model loss:

  L_total = sum_i [ exp(-log_var_i) * L_i + log_var_i ]

Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses
           for Scene Geometry and Semantics", CVPR 2018.
"""

import torch
import torch.nn as nn


class AdaptiveLoss(nn.Module):
    """
    Learnable uncertainty-based loss balancer for multi-task training.

    Args:
        tasks: list of task names, e.g. ['action', 'world']
    """

    def __init__(self, tasks):
        super().__init__()
        self.task_names = tasks
        # log(sigma^2) — one per task, learned jointly
        self.log_vars = nn.Parameter(torch.zeros(len(tasks)))

    def forward(self, losses):
        """
        Args:
            losses: list of scalar loss tensors, one per task
        Returns:
            Weighted total loss (scalar)
        """
        total_loss = losses[0].new_zeros(1).squeeze()
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss = total_loss + precision * loss + self.log_vars[i]
        return total_loss

    def get_weights(self):
        """Return current task weight (precision) for logging."""
        return {
            name: float(torch.exp(-var).item())
            for name, var in zip(self.task_names, self.log_vars)
        }
