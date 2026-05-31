# core/training/grad_coordination.py
"""
PCGrad: Gradient surgery for multi-task learning.

Projects conflicting task gradients to reduce interference between
the action loss and world model loss during training.

Usage:
    pc = PCGrad(optimizer)
    loss_dict = {"action": loss_a, "world": loss_w}
    pc.pc_backward(loss_dict, model.parameters())
    optimizer.step()

Reference: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.
"""

import torch


class PCGrad:
    """
    Minimal PCGrad implementation: mutually projects task gradients
    to reduce conflict between the action and world model training objectives.
    """

    def __init__(self, optimizer):
        self._optim = optimizer

    def zero_grad(self, set_to_none: bool = True):
        self._optim.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _project_conflict(self, g_i, g_j):
        """
        Project g_i onto the orthogonal complement of g_j when their dot product
        is negative (i.e., when the gradients conflict).

        g_i := g_i - (<g_i, g_j> / ||g_j||^2) * g_j   if <g_i, g_j> < 0
        """
        dot = 0.0
        for gi, gj in zip(g_i, g_j):
            if gi is None or gj is None:
                continue
            dot += torch.sum(gi * gj)
        if dot >= 0:  # No conflict; return unchanged
            return g_i

        norm2 = 0.0
        for gj in g_j:
            if gj is None:
                continue
            norm2 += torch.sum(gj * gj)
        if norm2 <= 0:
            return g_i

        coef = dot / norm2
        return [
            (gi - coef * gj if (gi is not None and gj is not None) else gi)
            for gi, gj in zip(g_i, g_j)
        ]

    def pc_backward(self, loss_dict, params):
        """
        Compute PCGrad update:
          1. Compute per-task gradient snapshots.
          2. Mutually project conflicting gradients.
          3. Write the projected gradient back into param.grad.

        Args:
            loss_dict: dict mapping task name → scalar loss tensor
            params:    iterable of model parameters (must require grad)
        """
        params = list(params)
        task_names = list(loss_dict.keys())
        grads = []

        # Step 1: snapshot per-task gradients
        for name in task_names:
            self._optim.zero_grad(set_to_none=True)
            loss_dict[name].backward(retain_graph=True)
            snap = [
                p.grad.detach().clone() if p.grad is not None else None
                for p in params
            ]
            grads.append(snap)

        # Step 2: project conflicts (pairwise from first task outward)
        g_ref = grads[0]
        for k in range(1, len(grads)):
            g_ref = self._project_conflict(g_ref, grads[k])

        # Step 3: write projected gradients back
        self._optim.zero_grad(set_to_none=True)
        for p, g in zip(params, g_ref):
            if p.requires_grad and g is not None:
                p.grad = g
