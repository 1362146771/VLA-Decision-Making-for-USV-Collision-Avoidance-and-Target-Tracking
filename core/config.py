from dataclasses import dataclass, asdict, fields
from pathlib import Path
import hashlib
import json
import yaml

@dataclass
class ModelConfig:
    clip_id: str = "openai/clip-vit-large-patch14"
    image_size: int = 224
    memory_frames: int = 3
    keyframe_stride: int = 5
    frame_period_s: float = 0.1
    prediction_steps: int = 4
    world_dim: int = 256
    world_arch: str = "cnn_gru_v1"
    world_refinement_width: int = 364
    action_embed_dim: int = 64
    policy_dim: int = 768
    policy_layers: int = 12
    policy_heads: int = 12
    policy_ffn: int = 3072
    action_hidden: int = 256
    max_text_length: int = 32
    dropout: float = 0.1
    unfreeze_vision_layers: int = 12
    normalization: str = "imagenet"

    def validate(self):
        if self.world_arch not in ("cnn_gru_v1", "cnn_gru_345_v2"):
            raise ValueError("Unknown world architecture")
        if self.world_refinement_width < 1:
            raise ValueError("Invalid spatial refinement width")
        if self.normalization != "imagenet": raise ValueError("This revision pins ImageNet normalization; do not silently change it")
        if self.image_size < 16 or self.memory_frames < 1 or self.keyframe_stride < 1: raise ValueError("Invalid image/history settings")
        if self.world_dim < 1 or self.prediction_steps < 1 or self.frame_period_s <= 0: raise ValueError("Invalid world/time settings")
        if self.policy_dim % self.policy_heads or self.policy_layers < 1: raise ValueError("Invalid transformer dimensions")
        if self.max_text_length < 2 or not 0 <= self.dropout < 1: raise ValueError("Invalid text/dropout settings")
        return self

def load_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"model", "training"}: raise ValueError("Config needs model and training sections only")
    cfg = ModelConfig(**raw["model"]).validate()
    train = raw["training"]
    required = {"seed", "world", "clip", "policy"}
    if set(train) != required: raise ValueError(f"Training keys must be {required}")
    for stage in ("world", "clip", "policy"):
        s = train[stage]
        keys = {"epochs", "lr", "global_batch", "micro_batch", "weight_decay", "amp", "balanced"}
        if set(s) != keys: raise ValueError(f"Unknown/missing {stage} training fields")
        if min(s["epochs"], s["global_batch"], s["micro_batch"], s["lr"]) <= 0: raise ValueError("Positive training settings required")
        if s["global_batch"] % s["micro_batch"]: raise ValueError("global_batch must be a multiple of micro_batch")
        if s["balanced"] and stage != "policy": raise ValueError("Class balancing is supported only for the policy stage")
    return cfg, train

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()
