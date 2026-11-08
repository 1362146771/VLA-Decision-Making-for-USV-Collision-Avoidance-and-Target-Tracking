from dataclasses import asdict
import json
from pathlib import Path
import pytest
import torch
import yaml
from transformers import CLIPTokenizerFast
from core.checkpoint import restore, check_tokenizer
from core.config import sha256_file, canonical_hash
from core.dataset_index import resolve_dataset
from core.model import OceanVLA


def test_explicit_configuration(cfg, hf_config, batch, tmp_path):
    model = OceanVLA(cfg, hf_config=hf_config, pretrained=False).eval()
    weights = tmp_path/"policy.pt"
    torch.save(model.state_dict(), weights)
    config = tmp_path/"config.yaml"
    config.write_text(yaml.safe_dump({"model": asdict(cfg)}))
    architecture = tmp_path/"architecture.json"
    architecture.write_text(json.dumps(hf_config))
    new, record = restore(weights, "policy", config=config, clip_config=architecture)
    inputs = {k: v for k, v in batch.items() if k != "targets"}
    with torch.no_grad():
        assert torch.equal(model(**inputs), new.eval()(**inputs))
    assert record["metadata"] == {}
    provenance = tmp_path/"provenance.json"
    provenance.write_text(json.dumps({"weights_sha256": "0"*64, "metadata": {}}))
    with pytest.raises(ValueError, match="weights_sha256"):
        restore(weights, "policy", config=config, clip_config=architecture, provenance=provenance)
    state = model.state_dict()
    state.pop("action_token")
    torch.save(state, weights)
    with pytest.raises(RuntimeError):
        restore(weights, "policy", config=config, clip_config=architecture)


def test_tokenizer_bundle():
    folder = Path(__file__).resolve().parents[1]/"tokenizer"
    check_tokenizer({"metadata": {}}, folder/"policy.pt", folder)
    tokenizer = CLIPTokenizerFast.from_pretrained(folder, local_files_only=True)
    encoded = tokenizer(["avoid collisions", "track target"], padding="max_length", max_length=32, truncation=True)
    assert len(tokenizer) == 49408
    assert all(len(row) == 32 for row in encoded["input_ids"])
    assert tokenizer.bos_token_id == 49406 and tokenizer.eos_token_id == 49407


def test_index_integrity(tmp_path):
    partition = {"splits": {"train": ["a"], "val": ["b"], "test": ["c"]}}
    part = tmp_path/"partition.json"
    part.write_text(json.dumps(partition))
    episode = {"episode_id": "c", "frame_count": 40, "decisions": [{"sample_id": "c:15", "frame_index": 15}]}
    support = {"c": episode["decisions"]}
    manifest = {"schema": "oceanvla.collected_support.v1", "split": "test", "episodes": [episode],
                "decision_support_sha256": canonical_hash(support)}
    test = tmp_path/"test.json"
    test.write_text(json.dumps(manifest))
    item = {"manifest": "test.json", "manifest_sha256": sha256_file(test), "episodes": 1,
            "raw_frames": 40, "usable_windows": 1, "decision_support_sha256": canonical_hash(support)}
    index = {"schema": "oceanvla.dataset_index.v1", "partition": "partition.json",
             "partition_file_sha256": sha256_file(part), "partition_sha256": canonical_hash(partition),
             "episode_count": 3, "frame_count": 120, "usable_windows": 3,
             "splits": {name: dict(item) for name in partition["splits"]}}
    path = tmp_path/"dataset.json"
    path.write_text(json.dumps(index))
    assert resolve_dataset(path) == test
    from core.engine import evaluate
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="recorded_action_labels"):
        evaluate(SimpleNamespace(dataset=path))
    test.write_text(json.dumps({**manifest, "split": "train"}))
    with pytest.raises(ValueError, match="manifest_sha256"):
        resolve_dataset(path)
    index["partition"] = "../partition.json"
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        resolve_dataset(path)


def test_prediction_without_targets(tmp_path, cfg):
    from core.data import prepare, EpisodeDataset
    from test_data_metrics import episodes, Tokenizer
    destination = tmp_path/"split"
    prepare(tmp_path, episodes(tmp_path, cfg), destination, cfg)
    manifest = destination/"test.json"
    doc = json.loads(manifest.read_text())
    for row in doc["samples"]:
        row.pop("target")
    manifest.write_text(json.dumps(doc))
    ds = EpisodeDataset(tmp_path, manifest, cfg, "policy", Tokenizer(), prediction_only=True)
    assert "targets" not in ds[0]
    with pytest.raises(ValueError):
        EpisodeDataset(tmp_path, manifest, cfg, "policy", Tokenizer())
