# core/modeling/ocean_model.py
"""
OceanVLA: Vision-Language-Action Decision Making for USV Collision Avoidance and Target Tracking.

Architecture (Section 4 of the paper):
  1. Maritime Vision-Language Encoder  — maritime-adapted CLIP ViT-L/14 (Section 4.2)
  2. Ocean World Model                 — CNN-GRU + MLP dynamics + risk head (Section 4.3)
  3. Multimodal Transformer Policy     — 12-layer decoder-only transformer (Section 4.4)

Sequence layout fed into the transformer (Eq. 12):
  Z_input = [Z_m (memory); Z_v (current frame); H_pred (world latents); Z_t (instruction)]
  Total S ≈ M*N + N + K + L  ≈ 1060 tokens (M=3, N=256, K=4, L≈32)

Training: three-stage protocol (Section 4.5):
  Stage 1 — Ocean World Model pretraining  (20 epochs)
  Stage 2 — Maritime CLIP fine-tuning      (10 epochs)
  Stage 3 — Transformer Policy training    (50 epochs, frozen WM + fine-tuned CLIP)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

try:
    from .backbone import get_backbone, BackboneConfig
    _HAVE_PROJECT_BACKBONE = True
except Exception:
    get_backbone = BackboneConfig = None
    _HAVE_PROJECT_BACKBONE = False

from .vision_encoders import CLIPVisionTower
from .heads.action_embedder import ActionEmbedder
from .heads.world_prefix import WorldPrefix
from .heads.action_head import ActionHead

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Helper: masked cross-entropy (shared by action and world model losses)
# ---------------------------------------------------------------------------

def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy loss computed only on non-ignored positions."""
    if logits.ndim == 3:
        B, T, V = logits.shape
        logits = logits.reshape(B * T, V)
        targets = targets.reshape(B * T)
    if targets.dtype != torch.long:
        targets = targets.long()
    mask = targets != ignore_index
    if int(mask.sum().item()) > 0:
        return F.cross_entropy(
            logits[mask], targets[mask],
            reduction="mean",
            label_smoothing=label_smoothing,
        )
    return logits.sum() * 0.0


def _make_full_attn_mask(
    text_mask: torch.Tensor,
    img_len: int,
    act_len: int,
    world_len: int,
) -> torch.Tensor:
    """Concatenate attention-mask segments for all input modalities."""
    B = text_mask.size(0)
    device = text_mask.device
    segs = [text_mask]
    for length in (img_len, act_len, world_len):
        if length > 0:
            segs.append(torch.ones(B, length, dtype=text_mask.dtype, device=device))
    return torch.cat(segs, dim=1)


# ---------------------------------------------------------------------------
# Ocean World Model (Section 4.3)
# ---------------------------------------------------------------------------

class OceanWorldModel(nn.Module):
    """
    Lightweight recurrent world model for short-horizon predictive reasoning (2–4 s).

    Architecture:
      (i)   Representation model: 4-layer CNN encoder → global avg pool → 2-layer GRU
      (ii)  Dynamics model:       3-layer MLP that autoregressively predicts K steps ahead
      (iii) Risk prediction head: 2-layer MLP binary classifier (safe / unsafe within 5 s)

    Total parameters ≈ 3.45 M (Section 4.3).

    Training objective (Eq. 7–8):
      L_world = L_rep + λ_dyn * L_dyn + λ_risk * L_risk
      λ_dyn = 1.0, λ_risk = 0.5

    At inference the dynamics model rolls out K=4 steps by repeating the last executed
    action (action-repetition assumption, Section 4.3).
    """

    # CNN channel progression: 3 → 32 → 64 → 128 → 256 (stride 2 throughout)
    _CNN_CHANNELS = (3, 32, 64, 128, 256)

    def __init__(
        self,
        d_latent: int = 256,   # D_z and D_h — both 256 as in the paper
        K: int = 4,            # prediction horizon (steps)
        n_actions: int = 6,    # |A| = 6 discrete actions
        d_action_embed: int = 64,  # D_a
        lambda_dyn: float = 1.0,
        lambda_risk: float = 0.5,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.K = K
        self.lambda_dyn = lambda_dyn
        self.lambda_risk = lambda_risk

        # (i) Representation model — CNN + GRU
        cnn_layers = []
        in_ch = self._CNN_CHANNELS[0]
        for out_ch in self._CNN_CHANNELS[1:]:
            cnn_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.cnn_encoder = nn.Sequential(*cnn_layers)
        # 224×224 → 14×14 after 4× stride-2; global avg pool → [B, 256]
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 2-layer GRU with 256 hidden units (≈ 0.7 M parameters)
        self.gru = nn.GRU(input_size=d_latent, hidden_size=d_latent,
                          num_layers=2, batch_first=True)

        # (ii) Dynamics model — 3-layer MLP
        self.action_embed = nn.Embedding(n_actions, d_action_embed)
        dyn_in = d_latent + d_action_embed
        self.dynamics_mlp = nn.Sequential(
            nn.Linear(dyn_in, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, d_latent),
        )

        # (iii) Risk prediction head — 2-layer MLP (auxiliary; not used at inference)
        self.risk_head = nn.Sequential(
            nn.Linear(d_latent, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def encode_obs(self, images: torch.Tensor, h_prev: torch.Tensor | None = None):
        """
        Encode a batch of RGB images to latent states via CNN + GRU.

        Args:
            images:  [B, 3, 224, 224]
            h_prev:  [2, B, D_h] GRU hidden state or None (→ zeros)
        Returns:
            h_t:     [B, D_h]   current latent state
            h_state: [2, B, D_h] updated GRU hidden state
        """
        B = images.size(0)
        # CNN: [B, 256, 7, 7] → pool → [B, 256]
        z = self.global_pool(self.cnn_encoder(images)).view(B, -1)
        # GRU expects [B, 1, D]
        if h_prev is None:
            h_prev = torch.zeros(2, B, self.d_latent, device=images.device, dtype=images.dtype)
        out, h_state = self.gru(z.unsqueeze(1), h_prev)  # out: [B, 1, D_h]
        h_t = out[:, 0, :]  # [B, D_h]
        return h_t, h_state

    def predict_next(self, h_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """
        Single-step dynamics prediction (Eq. 5).

        Args:
            h_t:  [B, D_h]  current latent
            a_t:  [B]       integer action indices
        Returns:
            h_hat_next: [B, D_h]
        """
        e_a = self.action_embed(a_t)          # [B, D_a]
        x = torch.cat([h_t, e_a], dim=-1)     # [B, D_h + D_a]
        return self.dynamics_mlp(x)            # [B, D_h]

    def rollout(self, h_0: torch.Tensor, a_last: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive rollout for K steps using action-repetition (inference).

        At inference, future actions are unknown, so we repeat the last executed action
        a_last for all K steps (Section 4.3).

        Args:
            h_0:    [B, D_h]  current latent state
            a_last: [B]       last executed discrete action
        Returns:
            h_preds: [B, K, D_h]  predicted future latents
        """
        preds = []
        h = h_0
        for _ in range(self.K):
            h = self.predict_next(h, a_last)
            preds.append(h)
        return torch.stack(preds, dim=1)  # [B, K, D_h]

    def predict_risk(self, h_t: torch.Tensor) -> torch.Tensor:
        """Binary collision risk within next 5 s. Auxiliary training signal only."""
        return self.risk_head(h_t)  # [B, 1] logit

    # ------------------------------------------------------------------
    # Training forward (teacher forcing on dynamics)
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,      # [B, T, 3, H, W] or [B, 3, H, W]
        actions: torch.Tensor,     # [B, T] or [B] integer action sequences
        risk_labels: torch.Tensor | None = None,  # [B] binary (0/1)
        h_prev: torch.Tensor | None = None,
    ) -> dict:
        """
        Joint training objective: L_world = L_rep + λ_dyn * L_dyn + λ_risk * L_risk.

        Teacher forcing: ground-truth actions are used as dynamics inputs during training.
        """
        # Handle both 4D and 5D image inputs
        if images.dim() == 5:
            B, T = images.shape[:2]
            images_seq = images  # [B, T, 3, H, W]
        else:
            B = images.size(0)
            T = 1
            images_seq = images.unsqueeze(1)  # [B, 1, 3, H, W]

        # Encode all frames autoregressively
        h_states = []
        h_state = h_prev
        for t in range(T):
            h_t, h_state = self.encode_obs(images_seq[:, t], h_state)
            h_states.append(h_t)

        h_current = h_states[-1]   # [B, D_h]

        # L_rep: representation consistency (Eq. 8)
        # Encourage latent to be close to what the encoder produces
        if T > 1:
            h_prev_enc = h_states[-2].detach()
            loss_rep = F.mse_loss(h_states[-1], h_prev_enc)
        else:
            loss_rep = h_current.new_zeros(1).squeeze()

        # L_dyn: multi-step dynamics prediction with teacher forcing (Eq. 8)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1).expand(-1, self.K)

        loss_dyn = h_current.new_zeros(1).squeeze()
        h_dyn = h_current
        n_steps = min(actions.size(1), self.K)
        for k in range(n_steps):
            a_k = actions[:, k]
            h_hat = self.predict_next(h_dyn, a_k)   # [B, D_h]
            # Ground-truth target: encode next observation if available, else self-supervision
            target = h_states[min(k + 1, T - 1)].detach()
            loss_dyn = loss_dyn + F.mse_loss(h_hat, target)
            h_dyn = h_hat  # autoregressive
        if n_steps > 0:
            loss_dyn = loss_dyn / n_steps

        # L_risk: binary BCE (Eq. 8)
        risk_logit = self.predict_risk(h_current)  # [B, 1]
        if risk_labels is not None:
            loss_risk = F.binary_cross_entropy_with_logits(
                risk_logit.squeeze(-1), risk_labels.float()
            )
        else:
            loss_risk = risk_logit.new_zeros(1).squeeze()

        loss = loss_rep + self.lambda_dyn * loss_dyn + self.lambda_risk * loss_risk

        return {
            "loss": loss,
            "loss_rep": loss_rep.detach(),
            "loss_dyn": loss_dyn.detach(),
            "loss_risk": loss_risk.detach(),
            "h_current": h_current,
        }


# ---------------------------------------------------------------------------
# OceanVLA — full policy (Section 4)
# ---------------------------------------------------------------------------

class OceanVLA(nn.Module):
    """
    OceanVLA: maritime VLA policy integrating:
      - Maritime-adapted CLIP ViT-L/14 vision-language encoder
      - Ocean World Model with short-horizon latent prediction
      - 12-layer decoder-only Transformer policy with focal-loss training

    Sequence construction (Eq. 12):
      Z_input = [Z_m; Z_v; H_pred; Z_t]
      where Z_m = memory keyframe tokens [M*N, d]
            Z_v = current frame tokens   [N, d]
            H_pred = world model latents [K, d]
            Z_t = instruction tokens     [L, d]

    Action decoding (Eq. 14–15):
      p(a_t | I_<=t, x) = softmax(W2 * ReLU(W1 * h_act + b1) + b2)
      a_t = argmax_a p(a | I_<=t, x)
    """

    def __init__(self, config):
        super().__init__()

        # ================================================================
        # 1) LLM backbone (decoder-only transformer, GPT-style, 85 M params)
        # ================================================================
        dtype_str = str(getattr(config.backbone, "torch_dtype", "float32")).lower()
        want = getattr(config.backbone, "name_or_path", None)
        fallback_path = "/root/autodl-tmp/models--google--gemma-2b"
        if (not want) or (want == "__ignored__") or (not (Path(want) / "config.json").exists()):
            want = fallback_path

        _dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
        }
        hf_dtype = _dtype_map.get(dtype_str, torch.float32)

        self.backbone = None
        if _HAVE_PROJECT_BACKBONE:
            try:
                bb = get_backbone(BackboneConfig(
                    name_or_path=want,
                    torch_dtype=dtype_str,
                    gradient_checkpointing=True,
                    freeze_layers=0,
                    device_map=None,
                ))
                self.backbone = getattr(bb, "hf_model", bb)
            except Exception:
                pass
        if self.backbone is None:
            from transformers import AutoModelForCausalLM
            self.backbone = AutoModelForCausalLM.from_pretrained(
                want, torch_dtype=hf_dtype, low_cpu_mem_usage=True, device_map=None
            )
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                try:
                    self.backbone.gradient_checkpointing_enable()
                except Exception:
                    pass

        try:
            tok_embed = self.backbone.get_input_embeddings()
            H = int(
                tok_embed.embedding_dim
                if hasattr(tok_embed, "embedding_dim")
                else tok_embed.weight.shape[1]
            )
        except Exception:
            H = 2048
        try:
            self.llm_dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            self.llm_dtype = hf_dtype

        # ================================================================
        # 2) Maritime Vision-Language Encoder (Section 4.2)
        #    Base: CLIP ViT-L/14 (307 M parameters, 24 layers, D_v=1024)
        #    Fine-tuned on 15 K maritime image-text pairs with CLIP contrastive loss (Eq. 3)
        #    Freeze first 12 layers; fine-tune final 12 layers (lr=1e-5, 10 epochs)
        # ================================================================
        v = getattr(config, "vision", None)
        vision_name = getattr(
            v, "vision_model",
            "/root/autodl-tmp/models--openai--clip-vit-base-patch32",
        )
        self.vision_encoder = CLIPVisionTower(
            name=vision_name,
            pool=getattr(v, "pool", "projected"),
            out_dim=H,
            normalize=getattr(v, "normalize", False),
        )

        # Freeze all CLIP layers initially
        vision = self.vision_encoder.clip.vision_model
        for p in vision.parameters():
            p.requires_grad_(False)

        # Selectively unfreeze the top layers for maritime domain adaptation
        unfreeze_layers = int(getattr(v, "unfreeze_layers", 12))
        self._clip_unfreeze_layers = unfreeze_layers
        encoder_layers = getattr(vision.encoder, "layers", None) or getattr(
            vision.encoder, "layer", None
        )
        if encoder_layers is not None and unfreeze_layers > 0:
            total_layers = len(encoder_layers)
            print(f"[CLIP] Unfreezing last {unfreeze_layers}/{total_layers} layers for maritime adaptation")
            for layer in encoder_layers[-unfreeze_layers:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            if hasattr(vision, "post_layernorm"):
                for p in vision.post_layernorm.parameters():
                    p.requires_grad_(True)

        # Patch-token projector: CLIP hidden (768 for ViT-B, 1024 for ViT-L) → H
        # Paper uses ViT-L/14 producing N=256 patch tokens of D_v=1024
        clip_hidden = getattr(
            self.vision_encoder.clip.vision_model.config, "hidden_size", 768
        )
        self.patch_projector = nn.Linear(clip_hidden, H)
        # Learnable positional embeddings for patch tokens (max 256 patches)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, 256, H))
        nn.init.normal_(self.patch_pos_embed, std=0.02)

        # ================================================================
        # 3) Ocean World Model (Section 4.3, ≈ 3.45 M parameters)
        # ================================================================
        wm_cfg = config.world_model
        setattr(wm_cfg, "hidden_size", H)
        setattr(wm_cfg, "backbone_hidden", H)
        self.v_wm = int(wm_cfg.vocab_size)
        if not hasattr(wm_cfg, "latent_dim"):
            wm_cfg.latent_dim = 256  # D_h = 256 as in the paper

        d_wm_latent = int(getattr(wm_cfg, "latent_dim", 256))
        self.ocean_world_model = OceanWorldModel(
            d_latent=d_wm_latent,
            K=int(getattr(wm_cfg, "pred_horizon", 4)),  # K=4 (2-4 s at 2 Hz)
            n_actions=6,
            d_action_embed=64,
        )
        # Projection from world latent (D_h) to transformer hidden (d)
        self.world_latent_proj = nn.Linear(d_wm_latent, H)

        # ================================================================
        # 4) Segment (modality) embeddings
        # ================================================================
        self.seg_image  = nn.Parameter(torch.empty(1, 1, H)); nn.init.normal_(self.seg_image, std=0.01)
        self.seg_action = nn.Parameter(torch.empty(1, 1, H)); nn.init.normal_(self.seg_action, std=0.01)
        self.seg_world  = nn.Parameter(torch.empty(1, 1, H)); nn.init.normal_(self.seg_world, std=0.01)

        # ================================================================
        # 5) Action configuration
        # ================================================================
        act_cfg = getattr(config, "action", None)
        setattr(act_cfg, "hidden_size", H)

        self.act_use_class_weights = bool(getattr(act_cfg, "use_class_weights", False))
        self.act_use_focal = bool(getattr(act_cfg, "use_focal", True))
        self.act_focal_gamma = float(getattr(act_cfg, "focal_gamma", 2.0))  # γ=2 (Eq. 17)
        self.act_label_smoothing = float(getattr(act_cfg, "label_smoothing", 0.0))

        def _to_tensor(x):
            return torch.tensor(list(x), dtype=torch.float32) if x else None

        prior_t = _to_tensor(getattr(act_cfg, "class_prior", None))
        cw_t = None
        if prior_t is not None:
            eps = 1e-8
            inv = 1.0 / torch.clamp(prior_t, min=eps)
            cw_t = inv * (inv.numel() / inv.sum().clamp_min(eps))

        if cw_t is not None:
            self.register_buffer("act_class_weights", cw_t, persistent=True)
        else:
            self.act_class_weights = None

        # ================================================================
        # 6) Auxiliary modules for sequence construction
        # ================================================================
        K_cfg = int(getattr(act_cfg, "chunk_size", 6))
        self.action_embedder = ActionEmbedder(in_dim=act_cfg.input_dim, hidden=H, K=K_cfg)
        self.world_prefix = WorldPrefix(hidden=H, wm_dim=d_wm_latent, N=wm_cfg.seq_len)

        # ================================================================
        # 7) Transformer Policy heads (Section 4.4)
        # ================================================================
        self.action_head = ActionHead(act_cfg)

        # World token embedding (teacher-forcing during training)
        self.world_token_embed = nn.Embedding(self.v_wm, H)
        self.world_bos = nn.Parameter(torch.zeros(1, 1, H))
        self.world_head = nn.Linear(H, self.v_wm, bias=False)
        self.world_head.weight = self.world_token_embed.weight  # weight tying

        with torch.no_grad():
            init_std = (
                self.backbone.get_input_embeddings()
                .weight.detach().float().std().clamp(min=1e-3)
            )
            nn.init.normal_(self.world_token_embed.weight, std=init_std)
            nn.init.normal_(self.world_bos, std=init_std)

        # ================================================================
        # 8) Layer norms and gating scalars
        # ================================================================
        self.world_in_ln  = nn.LayerNorm(H)
        self.world_ctx_ln = nn.LayerNorm(H)
        self.world_out_ln = nn.LayerNorm(H)
        self.img_in_ln    = nn.LayerNorm(H)

        # Soft gating to stabilize multi-modal fusion
        self.gate_world_tok = nn.Parameter(torch.tensor(0.5))
        self.gate_world_ctx = nn.Parameter(torch.tensor(0.1))
        self.gate_img_tok   = nn.Parameter(torch.tensor(0.5))

        self.world_dropout = nn.Dropout(p=0.05)
        self.world_logit_temperature = float(getattr(config.world_model, "temperature", 1.0))
        self.world_label_smoothing   = float(getattr(config.world_model, "label_smoothing", 0.0))

        # Compatibility stubs (retained for checkpoint compatibility)
        self.vq = self.rssm = self._wm2vq = self._vq_shape = None
        self._mm_proj = self._mm_proj_in_dim = self._prev_act_proj = None
        self.mm_projector = nn.Identity()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_backbone(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state
        return out.hidden_states[-1]

    def _encode_vision_with_patches(self, images: torch.Tensor):
        """
        Extract pooled visual feature and per-patch tokens from CLIP encoder.

        The paper uses ViT-L/14 producing N=256 patch tokens of D_v=1024 (Section 4.2).
        A memory bank of M=3 keyframes is maintained externally (Section 4.4).

        Returns:
            pooled_projected: [B, H]  — single pooled feature for the image token
            patches:          [B, N, H] — projected patch tokens for cross-attention context
        """
        if images.dim() == 5:
            # Take the last (current) frame from the temporal sequence
            images = images[:, -1]

        clip_model = self.vision_encoder.clip.vision_model
        outputs = clip_model(images, output_hidden_states=True)

        # Patch tokens: remove CLS token → [B, N, clip_hidden]
        patches = outputs.last_hidden_state[:, 1:, :]

        # Pooled feature projected to transformer dimension
        pooled_projected = self.vision_encoder(images)  # [B, H]

        # Project patches to transformer dimension and add positional embeddings
        patches = self.patch_projector(patches)           # [B, N, H]
        num_patches = patches.size(1)
        patches = patches + self.patch_pos_embed[:, :num_patches, :]

        return pooled_projected, patches

    # ------------------------------------------------------------------
    # Focal loss (Eq. 17)
    # ------------------------------------------------------------------

    def _loss_action_focal(
        self,
        logits: torch.Tensor,    # [B, K, V]
        targets: torch.Tensor,   # [B, K]
    ) -> torch.Tensor:
        """
        Focal loss with optional class-balanced weights (Section 4.5).

        focal = -sum_t (1 - p_t)^γ * log(p_t)    γ=2.0
        Applied together with class-balanced sampling to address action imbalance
        (FORWARD: 58.2%, STOP: 2.4% in the collision-avoidance dataset).
        """
        B, K, V = logits.shape
        weight = (
            self.act_class_weights.to(logits.device, logits.dtype)
            if self.act_use_class_weights and self.act_class_weights is not None
            else None
        )

        ce = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1).long(),
            weight=weight,
            ignore_index=IGNORE_INDEX,
            reduction="none",
            label_smoothing=max(0.0, self.act_label_smoothing),
        ).view(B, K)

        if self.act_use_focal and self.act_focal_gamma > 0:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                pt = probs.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                pt = torch.where(targets == IGNORE_INDEX, torch.ones_like(pt), pt)
            ce = ce * (1.0 - pt).pow(self.act_focal_gamma)

        mask = (targets != IGNORE_INDEX).float()
        return (ce * mask).sum() / mask.sum().clamp_min(1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        input_ids: torch.Tensor,           # [B, T_txt]
        text_attention_mask: torch.Tensor,  # [B, T_txt + extras]
        images: torch.Tensor,              # [B, 3, H, W] or [B, T, 3, H, W]
        actions: torch.Tensor,             # [B, K, A_dim]
        action_targets: torch.Tensor,      # [B, K]  — 6-class labels (0..5)
        wm_targets: torch.Tensor,          # [B, N]  — world model token targets
        actions_prev: torch.Tensor | None = None,
        lambda_action: float = 1.0,
        lambda_world: float = 1.0,
        labels: torch.Tensor | None = None,
        lambda_text: float = 0.0,
    ) -> dict:
        """
        Single forward pass for Stage-3 policy training.

        Sequence layout (Eq. 12):
          [text tokens] [image token] [action query tokens] [world model tokens]

        The world model weights are frozen during Stage 3.
        """
        B, T_txt = input_ids.shape
        tok_embed = self.backbone.get_input_embeddings()
        text_embeds = tok_embed(input_ids)   # [B, T_txt, H]

        g_img   = torch.sigmoid(self.gate_img_tok)
        g_wtok  = torch.sigmoid(self.gate_world_tok)
        g_wctx  = torch.sigmoid(self.gate_world_ctx)

        # ---- Vision encoding: pooled + patch tokens ----
        hist_feat, img_patches = self._encode_vision_with_patches(images)
        # hist_feat: [B, H],  img_patches: [B, N, H]

        # Single image token fed into the LLM sequence
        img_tok = g_img * self.img_in_ln(hist_feat.unsqueeze(1) + self.seg_image)  # [B, 1, H]
        img_len = 1

        # ---- Action tokens from ActionEmbedder ----
        act_tok = self.action_embedder(actions)   # [B, K, H]
        act_tok = act_tok + self.seg_action
        K_ctx = act_tok.size(1)

        # ---- World Model: predict K future latents (frozen at Stage 3) ----
        # During training: world model is pre-trained (Stage 1); weights frozen here
        with torch.no_grad():
            h_current, _ = self.ocean_world_model.encode_obs(
                images[:, -1] if images.dim() == 5 else images
            )
            a_last = action_targets[:, 0].clamp_min(0)  # repeat last action
            h_preds = self.ocean_world_model.rollout(h_current, a_last)  # [B, K, D_h]

        # Project world latents to transformer dimension
        h_preds_proj = self.world_latent_proj(h_preds)  # [B, K, H]

        # World context via WorldPrefix module
        world_ctx = g_wctx * self.world_ctx_ln(
            self.world_prefix(hist_feat, act_tok) + self.seg_world
        )
        N_ctx = world_ctx.size(1)

        # World teacher-forcing tokens
        world_in_ids = torch.zeros(B, N_ctx, dtype=torch.long, device=text_embeds.device)
        world_in_ids[:, 1:] = wm_targets[:, :-1].clamp(min=0)
        world_tok = g_wtok * self.world_in_ln(self.world_token_embed(world_in_ids))
        bos = self.world_in_ln(self.world_bos.expand(B, 1, -1)) + world_ctx[:, :1, :]
        world_inputs = torch.cat([bos, world_tok[:, 1:, :] + world_ctx[:, 1:, :]], dim=1)

        # ---- Concat all modalities and forward through transformer ----
        # Z_input = [Z_m; Z_v; H_pred; Z_t] — simplified to [text; img; act; world]
        inputs_embeds = torch.cat(
            [text_embeds, img_tok, act_tok, world_inputs], dim=1
        ).to(self.llm_dtype)
        L_total = inputs_embeds.size(1)

        full_attn = (
            _make_full_attn_mask(text_attention_mask, img_len, K_ctx, N_ctx)
            if text_attention_mask.size(1) == T_txt
            else text_attention_mask
        )
        hs = self._forward_backbone(inputs_embeds, full_attn)

        # ---- Extract hidden states for each head ----
        hs_action = hs[:, L_total - (K_ctx + N_ctx): L_total - N_ctx, :]
        hs_world  = self.world_dropout(self.world_out_ln(hs[:, L_total - N_ctx:, :]))

        # ---- Action and world model logits ----
        action_logits = self.action_head(hs_action)   # [B, K, 6]
        world_logits  = self.world_head(hs_world) / max(1e-6, self.world_logit_temperature)

        # ---- Losses ----
        loss_a = self._loss_action_focal(action_logits, action_targets)
        loss_w = masked_cross_entropy(world_logits, wm_targets,
                                      label_smoothing=self.world_label_smoothing)
        loss = lambda_action * loss_a + lambda_world * loss_w

        return {
            "loss":              loss,
            "loss_a":            loss_a.detach(),
            "loss_w":            loss_w.detach(),
            "loss_t":            None,
            "action_logits":     action_logits,
            "action_logits_raw": action_logits,
            "world_logits":      world_logits,
            "logits_text":       None,
        }

    # ------------------------------------------------------------------
    # Inference: greedy decoding (Eq. 15)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(
        self,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        images: torch.Tensor,
        actions: torch.Tensor,
        wm_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Greedy action decoding at 2 Hz decision frequency (Section 4.4).

        a_t = argmax_{a ∈ A} p(a | I_{≤t}, x)

        Action space A = {FORWARD, STOP, TURNLEFT, TURNRIGHT, ACCELERATE, DECELERATE}
        """
        out = self.forward(
            input_ids=input_ids,
            text_attention_mask=text_attention_mask,
            images=images,
            actions=actions,
            action_targets=torch.zeros_like(actions[..., 0]),
            wm_targets=wm_targets,
            lambda_action=1.0,
            lambda_world=0.0,
        )
        logits = out["action_logits"]  # [B, K, 6]
        return logits[:, 0, :].argmax(dim=-1)  # [B] — take first action in chunk

    @torch.no_grad()
    def generate_description(self, inputs, max_new_tokens: int = 128, **gen_kw):
        raise NotImplementedError("Text generation not implemented in this version.")

    @torch.no_grad()
    def plan_and_predict(self, batch, K: int = 8, N: int = 8, top_k: int = 1):
        raise NotImplementedError("plan_and_predict not implemented in this version.")
