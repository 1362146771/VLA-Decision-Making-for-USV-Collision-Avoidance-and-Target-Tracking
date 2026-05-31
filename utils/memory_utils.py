# core/utils/memory_utils.py
"""Utilities for GPU memory optimization during training."""

import torch
from torch.utils.checkpoint import checkpoint


class MemoryOptimizer:
    """Helper class for reducing GPU memory usage during training."""

    @staticmethod
    def enable_gradient_checkpointing(model):
        """
        Enable gradient checkpointing to trade compute for memory.
        Falls back to module-level checkpointing when the model does not
        expose gradient_checkpointing_enable().
        """
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            def make_checkpointable(module):
                module.forward = checkpoint(module.forward)
            model.apply(make_checkpointable)

    @staticmethod
    def optimize_attention_memory(model, seq_length: int):
        """
        Enable memory-efficient attention (FlashAttention or chunked attention)
        on any sub-module that exposes the relevant interface.
        """
        for module in model.modules():
            if hasattr(module, "attention"):
                if hasattr(module.attention, "enable_flash"):
                    module.attention.enable_flash(seq_length)
                if hasattr(module.attention, "set_chunk_size"):
                    module.attention.set_chunk_size(256)

    @staticmethod
    def clear_cache():
        """Free unused GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
