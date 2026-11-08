from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import json
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.nn.parallel import DistributedDataParallel

from core.config import load_config
from core.data import EpochSampler
from core.distributed import DistributedContext, ShardedEpochSampler
from core.engine import stage_loss
from core.model import build_model


def _gradient_worker(rank, size, port, cfg, hf_config, output):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(size),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), USE_LIBUV="0")
    torch.set_num_threads(1)
    context = DistributedContext("cpu")
    try:
        for stage in ("world", "clip", "policy"):
            torch.manual_seed(21)
            model = build_model(stage, cfg, hf_config=hf_config, pretrained=False)
            reference = deepcopy(model)
            runner = DistributedDataParallel(model, broadcast_buffers=False,
                                             find_unused_parameters=stage == "policy")
            n = size * 2
            images = torch.randn(n, 8 if stage == "world" else 4, 3, 32, 32)
            ids = torch.tensor([[1, 3+i, 2, 0, 0, 0, 0, 0] for i in range(n)])
            batch = {"images": images, "expert_actions": torch.randint(0, 6, (n, 8)),
                     "risk_labels": (torch.arange(n) % 2).float(),
                     "input_ids": ids, "attention_mask": ids.ne(0).long(),
                     "targets": torch.arange(n) % 6, "last_action": torch.arange(n) % 6}
            if stage == "clip":
                batch["images"] = images[:, 0]
            expected = stage_loss(reference, stage, batch)["loss"]
            expected.backward()
            local = {k: v[rank*2:(rank+1)*2] for k, v in batch.items()}
            if stage == "clip":
                actual = stage_loss(runner, stage, local)["loss"]
                actual.backward()
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            else:
                with runner.no_sync():
                    stage_loss(runner, stage, {k: v[:1] for k,v in local.items()})["loss"].div(2).backward()
                stage_loss(runner, stage, {k: v[1:] for k,v in local.items()})["loss"].div(2).backward()
            for (name, param), (_, target) in zip(model.named_parameters(), reference.named_parameters()):
                assert (param.grad is None) == (target.grad is None), name
                if param.grad is not None:
                    torch.testing.assert_close(param.grad, target.grad, rtol=3e-4, atol=3e-6, msg=name)
            loss, count = context.totals(float(expected.detach()), 1)
            assert count == size
            context.primary_call(lambda: Path(output, stage).write_text(str(loss / count)))
            dist.barrier()
        with pytest.raises(RuntimeError, match="primary_failure"):
            context.primary_call(lambda: (_ for _ in ()).throw(ValueError("primary_failure")))
    finally:
        context.close()


def test_four_rank_gradients_match_full_batch(tmp_path, cfg, hf_config):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_gradient_worker, args=(4, port, cfg, hf_config, str(tmp_path)), nprocs=4, join=True)
    assert all((tmp_path/stage).is_file() for stage in ("world", "clip", "policy"))


class Rows:
    def __init__(self, n):
        self.rows = [{"target": i % 6} for i in range(n)]

    def __len__(self):
        return len(self.rows)


@pytest.mark.parametrize("balanced", [False, True])
def test_sharding_preserves_global_sampling_stream(balanced):
    ds = Rows(103)
    samplers = [ShardedEpochSampler(EpochSampler(ds, 42, balanced), rank, 4) for rank in range(4)]
    for epoch in (0, 1, 5):
        reference = EpochSampler(ds, 42, balanced)
        reference.epoch = epoch
        for sampler in samplers:
            sampler.epoch = epoch
        shards = [list(sampler) for sampler in samplers]
        assert all(len(shard) == 25 for shard in shards)
        reconstructed = [value for step in zip(*shards) for value in step]
        assert reconstructed == list(reference)[:100]
        if not balanced:
            assert len(set(reconstructed)) == 100


def test_paper_four_gpu_batches():
    cfg, training = load_config(Path(__file__).resolve().parents[1]/"configs/train_4gpu.yaml")
    assert cfg.world_arch == "cnn_gru_345_v2"
    assert cfg.prediction_horizon_s == 2
    for stage, total in (("world", 64), ("clip", 128), ("policy", 128)):
        assert training[stage]["global_batch"] == total
        assert training[stage]["micro_batch"] * 4 == total


class WorldDataset:
    def __init__(self, root, manifest, cfg, stage, tokenizer, augment=False):
        self.cfg, self.augment = cfg, augment
        split = Path(manifest).stem
        self.doc = {"split": split, "dataset_hash": "test-data",
                    "episodes": [split], "assets": {}}
        self.rows = [{} for _ in range(11 if split == "train" else 5)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        generator = None if self.augment else torch.Generator().manual_seed(index)
        return {"images": torch.randn(8, 3, 32, 32, generator=generator),
                "expert_actions": torch.arange(8) % 6,
                "risk_labels": torch.tensor(float(index % 2))}


def _resume_worker(rank, size, ports, cfg, output, barrier):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(size),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(ports[0]), USE_LIBUV="0")
    torch.set_num_threads(1)
    from core import engine
    from core.checkpoint import restore
    engine.EpisodeDataset = WorldDataset
    root = Path(output)
    def arguments(name, resume=None):
        return SimpleNamespace(config=root/"config.yaml", stage="world", output=root/name,
                               device="cpu", resume=resume, data_root=root,
                               train_manifest=root/"train.json", val_manifest=root/"val.json")
    engine.train(arguments("full"))
    barrier.wait(timeout=60)
    os.environ["MASTER_PORT"] = str(ports[1])
    original = engine.save_checkpoint
    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("interrupted_epoch")
    engine.save_checkpoint = interrupt
    with pytest.raises(RuntimeError, match="interrupted_epoch"):
        engine.train(arguments("partial"))
    barrier.wait(timeout=60)
    os.environ["MASTER_PORT"] = str(ports[2])
    engine.save_checkpoint = original
    engine.train(arguments("resumed", root/"partial/epoch_001.pt"))
    full, fc = restore(root/"full/epoch_002.pt", "world")
    resumed, rc = restore(root/"resumed/epoch_002.pt", "world")
    assert fc["training_state"]["global_step"] == rc["training_state"]["global_step"]
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=1e-5, atol=1e-7)
    for before, after in zip(fc["training_state"]["rng_by_rank"], rc["training_state"]["rng_by_rank"]):
        assert torch.equal(before["torch_rng"], after["torch_rng"])
        assert before["python_rng"] == after["python_rng"]
        assert before["numpy_rng"] == after["numpy_rng"]
    assert len(rc["training_state"]["rng_by_rank"]) == size
    assert rc["metadata"]["training_protocol"]["dropped_tail_samples"] == 3
    assert rc["metadata"]["validation"]["samples"] == 5
    assert not any(key.startswith("module.") for key in resumed.state_dict())


def test_four_rank_epoch_resume(tmp_path, cfg):
    option = {"epochs": 2, "lr": .001, "global_batch": 8, "micro_batch": 1,
              "weight_decay": 0., "amp": False, "balanced": False}
    config = {"model": asdict(cfg), "training": {"seed": 31, "world": option,
              "clip": {**option, "micro_batch": 2}, "policy": option}}
    (tmp_path/"config.yaml").write_text(yaml.safe_dump(config))
    for split in ("train", "val"):
        (tmp_path/f"{split}.json").write_text("{}")
    sockets = [socket.socket() for _ in range(3)]
    for sock in sockets:
        sock.bind(("127.0.0.1", 0))
    ports = [sock.getsockname()[1] for sock in sockets]
    for sock in sockets:
        sock.close()
    barrier = mp.get_context("spawn").Barrier(4)
    mp.spawn(_resume_worker, args=(4, ports, cfg, str(tmp_path), barrier), nprocs=4, join=True)
    records = [json.loads(line) for line in (tmp_path/"full/metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2
