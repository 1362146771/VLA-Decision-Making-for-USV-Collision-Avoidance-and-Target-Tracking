from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from core.checkpoint import export_weights, restore
from core.config import load_config
from core.data import EpisodeDataset, policy_prompt, prepare
from core.evaluation import decision_support, score_predictions
from core.metrics import episode_accuracy
from core.model import OceanVLA
from core.inference import LivePolicy
from core.partition import PAPER_COUNTS, make_partition, partition_counts
from core.supervision import flip_allowed
from test_data_metrics import Tokenizer, episodes


def test_episode_macro_is_not_sample_micro():
    rows = [{"episode_id": "short", "task": "tracking", "target": 0, "top3": [0, 1, 2]}]
    rows += [{"episode_id": "long", "task": "tracking", "target": 5, "top3": [0, 1, 2]}] * 3
    result = episode_accuracy(rows)["by_task"]["tracking"]
    assert result["top1"] == result["top3"] == .5
    assert result["sample_top1"] == result["sample_top3"] == .25


def test_paper_partition_enforces_both_counts():
    records = []
    for split, tasks in PAPER_COUNTS.items():
        for task, (episodes_count, frames) in tasks.items():
            base, remainder = divmod(frames, episodes_count)
            records.extend({"episode_id": f"{split}-{task}-{i}", "task": task,
                            "frame_count": base + int(i < remainder)} for i in range(episodes_count))
    with pytest.raises(ValueError, match="--partition"):
        make_partition(records, paper=True)
    fixed = {name: [r["episode_id"] for r in records if r["episode_id"].startswith(name + "-")]
             for name in PAPER_COUNTS}
    split = make_partition(records, paper=True, fixed=fixed)
    assert partition_counts(records, split) == {
        name: {task: {"episodes": n, "frames": f} for task, (n, f) in tasks.items()}
        for name, tasks in PAPER_COUNTS.items()
    }
    with pytest.raises(ValueError, match="Paper count"):
        make_partition(records, paper=True, fixed={"train": split["val"], "val": split["train"], "test": split["test"]})


def test_shared_support_is_kept_for_single_frame_variant(tmp_path, cfg):
    path = tmp_path / "splits"
    prepare(tmp_path, episodes(tmp_path, cfg), path, cfg)
    manifest = path / "test.json"
    full = EpisodeDataset(tmp_path, manifest, cfg, "policy", Tokenizer())
    single = EpisodeDataset(tmp_path, manifest, replace(cfg, memory_frames=0), "policy", Tokenizer())
    assert [row["sample_id"] for row in full.rows] == [row["sample_id"] for row in single.rows]
    assert torch.equal(full[0]["images"][-1], single[0]["images"][0])
    assert len(single[0]["images"]) == 1
    shared_world = EpisodeDataset(tmp_path, manifest, replace(cfg, memory_frames=0, world_memory_frames=3),
                                  "policy", Tokenizer())
    assert torch.equal(full[0]["images"], shared_world[0]["images"])
    support = decision_support(full.doc)
    indices = next(iter(support.values()))["decisions"]
    assert [d["frame_index"] for d in indices] == list(range(6, 24, 2))
    external = {"decision_support_sha256": full.doc["decision_support_sha256"],
                "predictions": [{"sample_id": r["sample_id"], "top3": [r["target"], (r["target"]+1)%6,
                                                                     (r["target"]+2)%6]} for r in full.rows]}
    predictions = tmp_path/"external.json"
    predictions.write_text(json.dumps(external))
    assert score_predictions(manifest, predictions, tmp_path/"metrics.json")["top1"] == 1
    external["predictions"].pop()
    predictions.write_text(json.dumps(external))
    with pytest.raises(ValueError, match="cover every"):
        score_predictions(manifest, predictions, tmp_path/"incomplete.json")


def test_dynamic_flip_veto_survives_preparation(tmp_path, cfg, monkeypatch):
    path = episodes(tmp_path, cfg)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    static = {"scene_type": "static_obstacle", "laterally_symmetric": True, "allow_horizontal_flip": True}
    for episode in records:
        episode.update(static)
        episode["frames"][0]["encounter_type"] = "crossing"
    path.write_text("\n".join(map(json.dumps, records)))
    output = tmp_path/"splits"
    prepare(tmp_path, path, output, cfg)
    ds = EpisodeDataset(tmp_path, output/"train.json", cfg, "policy", Tokenizer(), augment=True)
    seen = []
    def record(paths, size, *, augment, flip):
        seen.append(flip)
        return torch.zeros(len(paths), 3, size, size)
    monkeypatch.setattr("core.data.random.random", lambda: 0.)
    monkeypatch.setattr("core.data.image_sequence", record)
    ds[0]
    ds[1]
    assert seen == [False, False]
    ds.rows[1].update(static)
    ds[1]
    assert seen[-1] is True
    assert not flip_allowed({**static, "encounter_type": "head_on"})


@pytest.mark.parametrize("row", range(1, 7))
def test_ablation_components_and_checkpoint(row, cfg, hf_config, batch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    variant, training = load_config(root/"configs/train.yaml", f"row{row}")
    cfg = replace(cfg, memory_frames=variant.memory_frames, encoder_kind=variant.encoder_kind,
                  use_world=variant.use_world, prompt_mode=variant.prompt_mode, policy_loss=variant.policy_loss,
                  world_memory_frames=variant.world_memory_frames, policy_clip_mode=variant.policy_clip_mode)
    model = OceanVLA(cfg, hf_config=hf_config, pretrained=False).eval()
    args = {k: v for k, v in batch.items() if k != "targets"}
    args["images"] = args["images"][:, -(1+cfg.observation_memory_frames):]
    with torch.no_grad():
        logits = model(**args)
    assert logits.shape == (2, 6)
    assert (model.world is not None) == (row >= 3)
    assert (cfg.policy_loss == "focal") == training["policy"]["balanced"]
    if row == 1:
        weight = tmp_path/"row1.pt"
        export_weights(weight, model, "policy", {})
        restored, _ = restore(weight, "policy")
        with torch.no_grad():
            assert torch.equal(logits, restored.eval()(**args))


def test_visual_ablation_preserves_world_history_and_shared_tokens(cfg, hf_config, batch, monkeypatch):
    cfg = replace(cfg, world_memory_frames=3)
    bank = OceanVLA(cfg, hf_config=hf_config, pretrained=False).eval()
    single = OceanVLA(replace(cfg, memory_frames=0), hf_config=hf_config, pretrained=False).eval()
    single.load_state_dict(bank.state_dict(), strict=True)
    observed = {}
    for name, model in (("bank", bank), ("single", single)):
        original = model.world.encode
        def encode(images, *, name=name, original=original):
            observed[name + "_world"] = images
            return original(images)
        monkeypatch.setattr(model.world, "encode", encode)
        def capture(module, args, name=name):
            observed[name + "_tokens"] = args[0].detach()
        model.policy.register_forward_pre_hook(capture)
        with torch.no_grad():
            model(**{k: v for k, v in batch.items() if k != "targets"})
    assert torch.equal(observed["single_world"], batch["images"])
    assert torch.equal(observed["bank_world"], observed["single_world"])
    extra_tokens = 3 * bank.encoder.patch_count
    torch.testing.assert_close(observed["bank_tokens"][:, extra_tokens:], observed["single_tokens"])


def test_generic_and_maritime_policy_clip_trainability_matches(cfg, hf_config):
    root = Path(__file__).resolve().parents[1]
    masks = []
    for variant in ("row3", "row4"):
        preset, _ = load_config(root/"configs/train.yaml", variant)
        model = OceanVLA(replace(cfg, encoder_kind=preset.encoder_kind, policy_clip_mode=preset.policy_clip_mode),
                         hf_config=hf_config, pretrained=False).train()
        assert model.encoder.training
        masks.append({name for name, param in model.encoder.named_parameters() if param.requires_grad})
    assert masks[0] == masks[1]
    assert any("text_model" in name for name in masks[0])
    assert any("vision_model.encoder.layers.1" in name for name in masks[0])
    assert not any("vision_model.encoder.layers.0" in name for name in masks[0])


def test_live_minimal_prompt_uses_explicit_task_and_shared_world_context(tmp_path, cfg, hf_config):
    raw = episodes(tmp_path, cfg)
    paths = [row["image"] for row in json.loads(raw.read_text().splitlines()[0])["frames"]]
    cfg = replace(cfg, memory_frames=0, world_memory_frames=3, prompt_mode="minimal_task")
    policy = LivePolicy.__new__(LivePolicy)
    policy.model = OceanVLA(cfg, hf_config=hf_config, pretrained=False).eval()
    policy.device, policy.cfg = torch.device("cpu"), cfg
    received = []
    def tokenize(text, **kwargs):
        received.append(text)
        return Tokenizer()(text, **kwargs)
    policy.tokenizer = tokenize
    policy.reset()
    for i, path in enumerate(paths[:7]):
        policy.push_frame(tmp_path/path, i*cfg.frame_period_s)
    assert len(policy.frames) == 7
    with pytest.raises(ValueError, match="require task"):
        policy.decide("Follow the red vessel.", 0)
    assert len(policy.decide("Follow the red vessel.", 0, task="tracking")["probabilities"]) == 6
    assert received == ["track target"]
    assert policy_prompt("Detailed instruction", cfg.prompt_mode, "avoidance") == "avoid collisions"
    assert policy_prompt("Detailed instruction", "minimal") == "Navigate."
