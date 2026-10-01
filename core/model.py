
from dataclasses import asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig

class SpatialRefinement(nn.Module):

    def __init__(self, dim, width):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(dim, width, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width, dim, 3, padding=1))
        self.skip = nn.AvgPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        return F.relu(self.branch(x) + self.skip(x))

class OceanWorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.world_dim
        channels = [3, 32, 64, 128, d]
        layers = []
        for a, b in zip(channels[:-1], channels[1:]):
            layers.extend([nn.Conv2d(a, b, 3, stride=2, padding=1), nn.ReLU()])
        if cfg.world_arch == "cnn_gru_345_v2":
            layers.append(SpatialRefinement(d, cfg.world_refinement_width))
        self.cnn = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.gru = nn.GRU(d, d, num_layers=2, batch_first=True)
        self.action_embedding = nn.Embedding(6, cfg.action_embed_dim)
        self.dynamics = nn.Sequential(nn.Linear(d+cfg.action_embed_dim, 512), nn.ReLU(),
                                      nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, d))
        self.risk = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))

    def encode(self, images, hidden=None):
        if images.ndim != 5: raise ValueError("World images must be [B,T,3,H,W]")
        b, t, c, h, w = images.shape
        z = self.cnn(images.reshape(b*t, c, h, w)).reshape(b, t, -1)
        return self.gru(z, hidden)

    def next_state(self, latent, action):
        return self.dynamics(torch.cat([latent, self.action_embedding(action)], -1))

    def rollout(self, latent, actions):
        if actions.ndim == 1:
            actions = actions[:, None].expand(-1, self.cfg.prediction_steps)
        if actions.shape[1] != self.cfg.prediction_steps: raise ValueError("Wrong prediction action horizon")
        predictions = []
        for k in range(self.cfg.prediction_steps):
            latent = self.next_state(latent, actions[:, k])
            predictions.append(latent)
        return torch.stack(predictions, 1)

    def forward(self, images, expert_actions, risk_labels):
        """History including anchor + K true future frames, equally spaced.

        expert_actions[:,j] is the action at sampled frame j, over its next
        sampling interval. Risk is collision within five seconds after anchor.
        """
        m, k = self.cfg.memory_frames, self.cfg.prediction_steps
        if images.shape[1] != m+1+k or expert_actions.shape != images.shape[:2]:
            raise ValueError("World sequence/actions not aligned")
        h, _ = self.encode(images)
        anchor = h[:, m]
        prior = self.next_state(h[:, m-1], expert_actions[:, m-1])
        loss_rep = F.mse_loss(prior, anchor.detach())
        pred = self.rollout(anchor, expert_actions[:, m:m+k])
        # Sum over future steps, mean over batch and latent dimension.
        per_step = (pred-h[:, m+1:m+1+k].detach()).square().mean((0, 2))
        risk_logits = self.risk(anchor).squeeze(-1)
        loss_risk = F.binary_cross_entropy_with_logits(risk_logits, risk_labels.float())
        return {"loss": loss_rep+per_step.sum()+0.5*loss_risk,
                "loss_rep": loss_rep, "loss_dyn": per_step.sum(), "loss_risk": loss_risk,
                "mse_1": per_step[0], "mse_k": per_step[-1], "risk_logits": risk_logits}

class MaritimeCLIP(nn.Module):
    def __init__(self, cfg, *, hf_config=None, pretrained=True):
        super().__init__()
        from transformers import CLIPConfig, CLIPModel
        self.cfg = cfg
        if hf_config is not None:
            self.clip = CLIPModel(CLIPConfig.from_dict(hf_config))
        elif pretrained:
            self.clip = CLIPModel.from_pretrained(cfg.clip_id)
            # New maritime adaptation begins at tau=0.07. Strict checkpoint
            # reconstruction takes the hf_config branch and preserves saved tau.
            with torch.no_grad(): self.clip.logit_scale.fill_(math.log(1.0/0.07))
        else:
            raise ValueError("Random CLIP requires an explicit architecture config (tests/checkpoint restore)")
        vc = self.clip.config.vision_config
        tc = self.clip.config.text_config
        if vc.image_size != cfg.image_size: raise ValueError("CLIP and preprocessing image sizes differ")
        if cfg.max_text_length > tc.max_position_embeddings: raise ValueError("Text context exceeds CLIP maximum")
        self.patch_count = (cfg.image_size // vc.patch_size)**2
        self.visual_dim, self.text_dim = vc.hidden_size, tc.hidden_size
        self.configure_trainable()

    def configure_trainable(self):
        # Both text tower and image projection train; only last N visual blocks.
        self.clip.requires_grad_(True)
        self.clip.vision_model.requires_grad_(False)
        n = self.cfg.unfreeze_vision_layers
        if n < 0 or n > len(self.clip.vision_model.encoder.layers):
            raise ValueError("Invalid number of CLIP visual layers to unfreeze")
        if n:
            for layer in self.clip.vision_model.encoder.layers[-n:]: layer.requires_grad_(True)
            self.clip.vision_model.post_layernorm.requires_grad_(True)

    def tokens(self, images, input_ids, attention_mask):
        b, t, c, h, w = images.shape
        visual = self.clip.vision_model(pixel_values=images.reshape(b*t, c, h, w)).last_hidden_state[:, 1:]
        visual = visual.reshape(b, t*self.patch_count, self.visual_dim)
        text = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return visual, text

    def forward(self, images, input_ids, attention_mask):
        # CLIP logit_scale is trainable, initialized by pretrained CLIP.
        result = self.clip(pixel_values=images, input_ids=input_ids,
                           attention_mask=attention_mask, return_loss=True)
        return {"loss": result.loss, "logits": result.logits_per_image}

def focal_loss(logits, targets, gamma=2.0):
    if targets.ndim != 1 or logits.shape != (len(targets), 6): raise ValueError("Expected [B,6] logits/[B] labels")
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = (-ce).exp()  # deliberately differentiable
    return ((1-pt).pow(gamma)*ce).mean()

class OceanVLA(nn.Module):
    def __init__(self, cfg, *, encoder=None, world=None, hf_config=None, pretrained=True):
        super().__init__()
        self.cfg = cfg.validate()
        self.encoder = encoder if encoder is not None else MaritimeCLIP(cfg, hf_config=hf_config, pretrained=pretrained)
        self.world = world if world is not None else OceanWorldModel(cfg)
        self.world.requires_grad_(False)
        self.world.eval()
        d = cfg.policy_dim
        self.visual_projection = nn.Linear(self.encoder.visual_dim, d)
        self.text_projection = nn.Linear(self.encoder.text_dim, d)
        self.world_projection = nn.Linear(cfg.world_dim, d)
        count = (cfg.memory_frames+1)*self.encoder.patch_count + cfg.prediction_steps + cfg.max_text_length + 1
        self.position = nn.Parameter(torch.randn(1, count, d)*0.02)
        self.action_token = nn.Parameter(torch.randn(1, 1, d)*0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.policy_heads, cfg.policy_ffn,
                                          cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        # Pre-LN self-attention blocks with no causal mask and no cross-attention.
        # Encoder API is used deliberately for the paper's non-causal policy.
        self.policy = nn.TransformerEncoder(layer, cfg.policy_layers,
                                            norm=nn.LayerNorm(d), enable_nested_tensor=False)
        # TransformerEncoder clones initialization; independently initialize blocks.
        for block in self.policy.layers:
            for param in block.parameters():
                if param.ndim > 1: nn.init.xavier_uniform_(param)
        self.action_head = nn.Sequential(nn.Linear(d, cfg.action_hidden), nn.ReLU(), nn.Linear(cfg.action_hidden, 6))

    def train(self, mode=True):
        super().train(mode)
        self.world.eval()
        return self

    def forward(self, images, input_ids, attention_mask, last_action):
        """No targets in inference API. History precedes current image.

        The world GRU is warmed on this exact history window in train/eval/live,
        so no hidden recurrent state from a different episode can leak in.
        """
        if images.shape[1] != self.cfg.memory_frames+1: raise ValueError("Need M history frames plus current image")
        if input_ids.shape[1] != self.cfg.max_text_length: raise ValueError("Use fixed padded CLIP context")
        if last_action.dtype != torch.long or last_action.shape != (images.shape[0],): raise ValueError("last_action must be [B] int64")
        visual, text = self.encoder.tokens(images, input_ids, attention_mask)
        with torch.no_grad():
            latents, _ = self.world.encode(images)
            future = self.world.rollout(latents[:, -1], last_action)
        b = images.shape[0]
        visual = visual.reshape(b, self.cfg.memory_frames+1, self.encoder.patch_count, -1)
        memory = self.visual_projection(visual[:, :-1]).flatten(1, 2)
        current = self.visual_projection(visual[:, -1])
        tokens = torch.cat([memory, current, self.world_projection(future),
                            self.text_projection(text), self.action_token.expand(b, -1, -1)], 1)
        tokens = tokens + self.position[:, :tokens.shape[1]]
        prefix = memory.shape[1] + current.shape[1] + future.shape[1]
        padding = torch.cat([torch.zeros(b, prefix, dtype=torch.bool, device=images.device),
                             ~attention_mask.bool(), torch.zeros(b, 1, dtype=torch.bool, device=images.device)], 1)
        hidden = self.policy(tokens, src_key_padding_mask=padding, is_causal=False)
        return self.action_head(hidden[:, -1])

def build_model(stage, cfg, *, hf_config=None, pretrained=True):
    if stage == "world": return OceanWorldModel(cfg)
    if stage == "clip": return MaritimeCLIP(cfg, hf_config=hf_config, pretrained=pretrained)
    if stage == "policy": return OceanVLA(cfg, hf_config=hf_config, pretrained=pretrained)
    raise ValueError(stage)

def clip_config_of(model):
    if isinstance(model, OceanVLA): return model.encoder.clip.config.to_dict()
    if isinstance(model, MaritimeCLIP): return model.clip.config.to_dict()
    return None

def parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
