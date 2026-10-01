import pytest
import torch
from core.config import ModelConfig
from transformers import CLIPConfig, CLIPVisionConfig, CLIPTextConfig

torch.set_num_threads(2)

@pytest.fixture
def cfg():
    return ModelConfig(clip_id="local-test-only", image_size=32, memory_frames=3, keyframe_stride=2,
                       frame_period_s=.25, prediction_steps=4, world_dim=16, action_embed_dim=8,
                       policy_dim=32, policy_layers=2, policy_heads=4, policy_ffn=64,
                       action_hidden=16, max_text_length=8, dropout=0., unfreeze_vision_layers=1)

@pytest.fixture
def hf_config():
    v = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, image_size=32, patch_size=8)
    t = CLIPTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, max_position_embeddings=8, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    return CLIPConfig.from_text_vision_configs(t,v,projection_dim=16).to_dict()

@pytest.fixture
def batch(cfg):
    ids = torch.tensor([[1, 3, 4, 2, 0, 0, 0, 0],[1, 5, 6, 2, 0, 0, 0, 0]])
    return {"images":torch.randn(2,4,3,32,32), "input_ids":ids, "attention_mask":(ids!=0).long(),
            "last_action":torch.tensor([0,3]), "targets":torch.tensor([1,4])}
