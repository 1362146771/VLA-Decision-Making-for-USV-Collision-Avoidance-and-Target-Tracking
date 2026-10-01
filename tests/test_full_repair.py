from dataclasses import asdict, replace
import json
from types import SimpleNamespace
import pytest
import torch
import yaml
from core.model import OceanWorldModel, OceanVLA
from core.config import ModelConfig
from core.checkpoint import export_weights, restore
from core.data import EpochSampler
from core.pipeline import preflight
from core.metrics import summarize_confusion


def test_revised_world_capacity_resolution_and_gradient():
    cfg = ModelConfig(world_arch="cnn_gru_345_v2")
    model = OceanWorldModel(cfg)
    assert sum(p.numel() for p in model.parameters()) == 3447597
    features = model.cnn[:-1](torch.randn(1, 3, 224, 224))
    assert features.shape == (1, 256, 7, 7)
    result = model(torch.randn(2, 8, 3, 32, 32), torch.randint(0, 6, (2, 8)), torch.tensor([0., 1.]))
    result["loss"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_tensor_only_full_checkpoint_roundtrip(cfg, hf_config, batch, tmp_path):
    cfg = replace(cfg, world_arch="cnn_gru_345_v2", world_refinement_width=24)
    model = OceanVLA(cfg, hf_config=hf_config, pretrained=False).eval()
    path = tmp_path/"best.pt"
    export_weights(path, model, "policy", {"validation": {"top1": .5}})
    raw = torch.load(path, weights_only=True)
    assert all(isinstance(value, torch.Tensor) for value in raw.values())
    assert all(any(k.startswith(p) for k in raw) for p in ("world.", "encoder.", "world_projection."))
    new, ck = restore(path, "policy")
    new.eval()
    inputs = {k:v for k,v in batch.items() if k != "targets"}
    with torch.no_grad():
        assert torch.equal(model(**inputs), new(**inputs))
    assert ck["metadata"]["validation"]["top1"] == .5
    with pytest.raises(FileExistsError):
        export_weights(path, model, "policy", {})
    path.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="sidecar"):
        restore(path)


def test_sidecar_binding_and_no_partial_fallback(cfg, hf_config, tmp_path):
    model = OceanVLA(cfg, hf_config=hf_config, pretrained=False)
    path = tmp_path/"best.pt"
    export_weights(path, model, "policy", {})
    state = torch.load(path, weights_only=True)
    state["action_token"].add_(1)
    torch.save(state, path)
    with pytest.raises(ValueError, match="integrity"):
        restore(path)
    del model.world
    with pytest.raises(ValueError, match="requires encoder"):
        export_weights(tmp_path/"partial.pt", model, "policy", {})


def test_old_weak_checkpoint_not_upgraded_by_metadata(tmp_path):
    path = tmp_path/"old.pt"
    torch.save({"identity": {"schema": "oceanvla.weak-policy.v1"}, "policy_state": {}}, path)
    with pytest.raises(ValueError, match="partial policy"):
        restore(path)


def test_effective_batch_balancing_and_reproducibility():
    rows = [{"target": c} for c in range(6) for _ in range(c+1)]
    class Data:
        def __len__(self): return 1001
    ds = Data(); ds.rows = rows
    sampler = EpochSampler(ds, seed=7, balanced=True, batch_size=128)
    indices = list(sampler)
    assert indices == list(sampler)
    for offset in range(0, len(indices), 128):
        counts = [sum(rows[i]["target"] == c for i in indices[offset:offset+128]) for c in range(6)]
        assert max(counts)-min(counts) <= 1
    sampler.epoch += 1
    assert indices != list(sampler)


def test_missing_class_is_reported_not_assigned_zero_accuracy():
    matrix = [[0]*6 for _ in range(6)]
    matrix[0][0] = 2
    report = summarize_confusion(matrix)
    assert report["class_recall"]["STOP"] is None
    assert "STOP" in report["missing_classes"]


def test_preflight_rejects_missing_supervision_without_changing_inputs(tmp_path, cfg):
    opts = {"epochs":1,"lr":.001,"global_batch":2,"micro_batch":2,"weight_decay":.0,"amp":False,"balanced":False}
    config = tmp_path/"config.yaml"
    config.write_text(yaml.safe_dump({"model":asdict(cfg),"training":{"seed":1,"world":opts,"clip":opts,"policy":opts}}))
    episodes = tmp_path/"episodes"; episodes.mkdir()
    for split in ("train", "val", "test"):
        (episodes/(split+".json")).write_text(json.dumps({"schema":"oceanvla.weak-policy.v1",
            "samples":[{"expert_action_id":None,"collision":None,"pseudo_action_id":3}]}))
    before = {p.name:p.read_bytes() for p in episodes.iterdir()}
    result = preflight(config, episodes, tmp_path/"no_captions")
    assert not result["ready"]
    assert len(result["errors"]) >= 6
    assert result["manifests"]["episodes/train"]["expert_action_present"] == 0
    assert before == {p.name:p.read_bytes() for p in episodes.iterdir()}
