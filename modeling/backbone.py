# core/modeling/backbone.py
"""
LLM Backbone wrapper for OceanVLA.

The transformer policy in the paper uses a 12-layer decoder-only GPT-style transformer
with 12 attention heads, hidden dimension d=768, totaling 85 M parameters (Section 4.4).

This wrapper provides a unified interface for any HuggingFace causal LM backbone,
exposing:
  - forward(inputs_embeds, attention_mask) -> last_hidden_state
  - get_input_embeddings()
  - hidden_size property

Usage:
    cfg = BackboneConfig(name_or_path="path/to/model", torch_dtype="bfloat16")
    backbone = get_backbone(cfg)
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

try:
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
except Exception:
    AutoConfig = None
    AutoModelForCausalLM = None
    PreTrainedModel = object


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BackboneConfig:
    """
    Configuration for the LLM backbone.

    Replace name_or_path with your local checkpoint or a HuggingFace model ID,
    e.g. 'google/gemma-2-2b' or '/path/to/local/weights'.
    """
    name_or_path: str = "openlm-research/open_llama_3b"
    torch_dtype: Union[str, torch.dtype, None] = "auto"
    device_map: Optional[Union[str, Dict[str, Any]]] = None
    gradient_checkpointing: bool = False
    freeze_layers: int = 0     # Freeze the first N transformer layers
    use_cache: bool = False    # Disable KV-cache during training
    output_hidden_states: bool = False
    compile: bool = False      # torch.compile (PyTorch >= 2.0)


def _resolve_dtype(dtype_cfg) -> Optional[torch.dtype]:
    if dtype_cfg is None or dtype_cfg == "auto":
        return None
    if isinstance(dtype_cfg, torch.dtype):
        return dtype_cfg
    if isinstance(dtype_cfg, str):
        s = dtype_cfg.lower()
        if s in ("bf16", "bfloat16"):
            return torch.bfloat16
        if s in ("fp16", "float16", "half"):
            return torch.float16
        if s in ("fp32", "float32", "float"):
            return torch.float32
    return None


# ---------------------------------------------------------------------------
# Backbone wrapper
# ---------------------------------------------------------------------------

class LLMBackbone(nn.Module):
    """
    Thin wrapper around HuggingFace AutoModelForCausalLM that:
      - Exposes forward(inputs_embeds, attention_mask) → last_hidden_state
      - Exposes get_input_embeddings()
      - Optionally freezes the first N transformer layers
    """

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        if AutoConfig is None or AutoModelForCausalLM is None:
            raise ImportError(
                "transformers is not installed. "
                "Please run: pip install transformers>=4.41"
            )

        dtype = _resolve_dtype(cfg.torch_dtype)
        hcfg = AutoConfig.from_pretrained(cfg.name_or_path)

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            cfg.name_or_path,
            config=hcfg,
            torch_dtype=dtype,
            device_map=cfg.device_map,
        )
        self.use_cache = bool(cfg.use_cache)

        # Resolve hidden dimension (field name varies across architectures)
        self.hidden_size = getattr(
            self.model.config, "hidden_size",
            getattr(self.model.config, "n_embd", None),
        )
        if self.hidden_size is None:
            raise ValueError(
                "Cannot determine hidden_size from model config. "
                "Check the model architecture."
            )

        if cfg.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        if cfg.freeze_layers and cfg.freeze_layers > 0:
            self._freeze_prefix_layers(cfg.freeze_layers)

        if cfg.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def _freeze_prefix_layers(self, n: int):
        """
        Freeze the first n transformer blocks.
        Supports common naming conventions: model.layers (LLaMA) and transformer.h (GPT-2).
        """
        layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h

        if layers is None:
            return  # Unknown architecture; skip silently

        for i, block in enumerate(layers):
            if i < n:
                for p in block.parameters():
                    p.requires_grad = False

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass returning an object with .last_hidden_state [B, L, D].

        Routes through the inner transformer directly (skipping the LM head)
        when possible to avoid unnecessary computation during policy training.
        """
        backbone = None
        if hasattr(self.model, "model"):
            backbone = self.model.model
        elif hasattr(self.model, "transformer"):
            backbone = self.model.transformer

        if backbone is not None:
            out = backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            return SimpleNamespace(last_hidden_state=out.last_hidden_state)

        # Fallback: run the full CausalLM and extract the last hidden state
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return SimpleNamespace(last_hidden_state=out.hidden_states[-1])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_backbone(cfg_dict_or_obj: Union[BackboneConfig, Dict[str, Any]]) -> LLMBackbone:
    """
    Instantiate an LLMBackbone from a BackboneConfig or a plain dict.

    Example:
        backbone = get_backbone({"name_or_path": "google/gemma-2-2b",
                                  "gradient_checkpointing": True})
    """
    if isinstance(cfg_dict_or_obj, dict):
        cfg = BackboneConfig(**cfg_dict_or_obj)
    else:
        cfg = cfg_dict_or_obj
    return LLMBackbone(cfg)
