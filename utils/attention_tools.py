# core/utils/attention_tools.py
"""Utilities for attention weight visualization and analysis."""

import torch


class AttentionVisualizer:
    """Collects and formats attention maps from transformer layers."""

    def __init__(self, model):
        self.model = model
        self._hooks = []
        self._attn_maps = {}

    def register_hooks(self, layer_names=None):
        """Register forward hooks on attention layers to collect attention maps."""
        for name, module in self.model.named_modules():
            if layer_names and name not in layer_names:
                continue
            if hasattr(module, "attention") or "attn" in name.lower():
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                # MultiheadAttention returns (output, attn_weights)
                if len(output) > 1 and output[1] is not None:
                    self._attn_maps[name] = output[1].detach().cpu()
            elif isinstance(output, torch.Tensor):
                self._attn_maps[name] = output.detach().cpu()
        return hook

    def get_attention_maps(self):
        """Return collected attention maps (dict: layer_name -> tensor)."""
        return dict(self._attn_maps)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._attn_maps.clear()
