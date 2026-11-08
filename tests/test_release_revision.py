from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import json
import random
import subprocess
import sys

import pytest
import torch
import yaml

from core import engine
from core.checkpoint import restore
from core.config import load_config
from core.data import EpisodeDataset, EpochSampler, prepare
from core.supervision import flip_allowed
from test_data_metrics import Tokenizer, episodes


@pytest.mark.parametrize("extra,expected", [
    ({}, True),
    ({"allow_horizontal_flip": False}, False),
    ({"dynamic_encounter": True}, False),
    ({"encounter_type": "crossing"}, False),
    ({"encounter_type": "head_on"}, False),
    ({"encounter_type": "overtaking"}, False),
    ({"asymmetric_give_way": True}, False),
])
def test_tracking_flip_constraints(extra, expected):
    assert flip_allowed({"task": "tracking", "allow_horizontal_flip": True, **extra}) is expected


def test_tracking_flip_transforms_supervision_and_images(tmp_path, cfg, monkeypatch):
    raw = episodes(tmp_path, cfg)
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    for record in records:
        record.update(task="tracking", allow_horizontal_flip=True)
    raw.write_text("\n".join(map(json.dumps, records)))
    prepare(tmp_path, raw, tmp_path / "split", cfg)
    observed = []
    texts = []
    def images(paths, size, *, augment, flip):
        observed.append((len(paths), flip))
        return torch.zeros(len(paths), 3, size, size)
    class Text(Tokenizer):
        def __call__(self, text, **kwargs):
            texts.append(text)
            return super().__call__(text, **kwargs)
    monkeypatch.setattr("core.data.image_sequence", images)
    monkeypatch.setattr("core.data.random.random", lambda: 0.)
    manifest = tmp_path / "split/train.json"
    policy = EpisodeDataset(tmp_path, manifest, cfg, "policy", Text(), augment=True)
    index = next(i for i, row in enumerate(policy.rows) if row["target"] == 3)
    item = policy[index]
    assert item["targets"].item() == 2
    assert item["last_action"].item() == 3
    assert texts[-1] == "turn port"
    assert observed[-1] == (4, True)
    world = EpisodeDataset(tmp_path, manifest, cfg, "world", augment=True)
    expected = [{2: 3, 3: 2}.get(a, a) for a in world.rows[0]["world_actions"]]
    assert world[0]["expert_actions"].tolist() == expected
    assert observed[-1] == (8, True)
    for record in records:
        record["frames"][0]["asymmetric_give_way"] = True
    raw.write_text("\n".join(map(json.dumps, records)))
    prepare(tmp_path, raw, tmp_path / "blocked", cfg)
    blocked = EpisodeDataset(tmp_path, tmp_path / "blocked/train.json", cfg, "policy", Text(), augment=True)
    blocked[0]
    assert observed[-1] == (4, False)
    test = EpisodeDataset(tmp_path, tmp_path / "split/test.json", cfg, "policy", Text())
    test[0]
    assert observed[-1] == (4, False)


def test_unbalanced_sampler_visits_each_sample_once():
    class Data:
        rows = [{"target": 0}] * 7 + [{"target": 1}]
        def __len__(self): return len(self.rows)
    sampler = EpochSampler(Data(), seed=3)
    assert sorted(sampler) == list(range(8))
    assert sampler.weights is None


def test_paper_cli_requires_partition_before_io(tmp_path):
    result = subprocess.run([sys.executable, "-m", "core.cli", "prepare", "--paper",
                             "--config", "missing", "--data-root", "missing", "--episodes", "missing",
                             "--output", str(tmp_path / "unused")], capture_output=True)
    assert result.returncode != 0 and b"--paper requires --partition" in result.stderr
    assert not (tmp_path / "unused").exists()


def test_release_configuration_and_data():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "data/dataset.json").exists()
    for filename in ("train.yaml", "train1.yaml"):
        _, training = load_config(root / "configs" / filename)
        assert training["policy"]["patience"] == 10
        assert training["policy"]["epochs"] == 50
        assert all("patience" not in training[stage] for stage in ("world", "clip"))


def test_policy_early_stopping_resume_preserves_counter_and_best(tmp_path, cfg, monkeypatch):
    cfg = replace(cfg, encoder_kind="none", use_world=False)
    option = {"epochs": 8, "lr": .001, "global_batch": 8, "micro_batch": 4,
              "weight_decay": 0., "amp": False, "balanced": False}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"model": asdict(cfg), "training": {
        "seed": 31, "world": option, "clip": {**option, "micro_batch": 8},
        "policy": {**option, "patience": 2}}}))
    prepare(tmp_path, episodes(tmp_path, cfg), tmp_path / "split", cfg)
    class LocalTokenizer(Tokenizer):
        def save_pretrained(self, directory):
            directory.mkdir(parents=True)
            for name in ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"):
                (directory / name).write_text("{}")
    monkeypatch.setattr("transformers.CLIPTokenizerFast.from_pretrained", lambda *a, **k: LocalTokenizer())
    def args(name, resume=None):
        return SimpleNamespace(config=config, stage="policy", output=tmp_path / name, device="cpu",
                               resume=resume, world_checkpoint=None, clip_checkpoint=None,
                               data_root=tmp_path, train_manifest=tmp_path / "split/train.json",
                               val_manifest=tmp_path / "split/val.json")
    def validation(losses):
        values = iter(losses)
        def run(*a, **k): return {"loss": next(values)}, []
        return run
    monkeypatch.setattr(engine, "validate", validation([1., .8, .8, .9]))
    engine.train(args("full"))
    full, full_ck = restore(tmp_path / "full/epoch_004.pt", "policy")
    assert full_ck["training_state"]["bad_epochs"] == 2
    assert full_ck["training_state"]["stop_reason"] == "early_stopping"
    assert not (tmp_path / "full/epoch_005.pt").exists()
    best, best_ck = restore(tmp_path / "full/best.pt", "policy")
    assert best_ck["metadata"]["epoch_completed"] == 2
    save = engine.save_checkpoint
    def interrupt(path, *a, **k):
        result = save(path, *a, **k)
        if Path(path).name == "epoch_003.pt": raise RuntimeError("interrupted")
        return result
    with monkeypatch.context() as local:
        local.setattr(engine, "validate", validation([1., .8, .8]))
        local.setattr(engine, "save_checkpoint", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            engine.train(args("interrupted"))
    monkeypatch.setattr(engine, "validate", validation([.9]))
    engine.train(args("resumed", tmp_path / "interrupted/epoch_003.pt"))
    resumed, resumed_ck = restore(tmp_path / "resumed/epoch_004.pt", "policy")
    assert resumed_ck["training_state"]["bad_epochs"] == 2
    assert resumed_ck["training_state"]["global_step"] == full_ck["training_state"]["global_step"]
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in full.state_dict().items())
    resumed_best, _ = restore(tmp_path / "resumed/best.pt", "policy")
    assert all(torch.equal(v, resumed_best.state_dict()[k]) for k, v in best.state_dict().items())
    with pytest.raises(ValueError, match="early_stopping"):
        engine.train(args("completed", tmp_path / "resumed/epoch_004.pt"))
