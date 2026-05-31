# core/training/schedulers.py
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math

def _warmup_lambda(current_step: int, num_warmup_steps: int):
    if num_warmup_steps <= 0:
        return 1.0
    return min(1.0, float(current_step) / float(max(1, num_warmup_steps)))

def _linear_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return _warmup_lambda(current_step, num_warmup_steps)
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 1.0 - progress)

def _cosine_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int, cycles: float = 0.5):
    if current_step < num_warmup_steps:
        return _warmup_lambda(current_step, num_warmup_steps)
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * cycles * progress)))

def get_scheduler(
    optimizer: Optimizer,
    name: str = "cosine",
    num_warmup_steps: int = 1000,
    num_training_steps: int = 100000,
    cycles: float = 0.5,
) -> LambdaLR:
    name = (name or "cosine").lower()
    if name == "constant":
        lr_lambda = lambda step: 1.0
    elif name == "linear":
        lr_lambda = lambda step: _linear_lambda(step, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    elif name == "cosine":
        lr_lambda = lambda step: _cosine_lambda(step, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, cycles=cycles)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
    return LambdaLR(optimizer, lr_lambda=lr_lambda)
