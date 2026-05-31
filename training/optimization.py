# core/training/optimization.py
import torch
from torch.optim import AdamW

def build_optimizer(params, lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
    return AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
