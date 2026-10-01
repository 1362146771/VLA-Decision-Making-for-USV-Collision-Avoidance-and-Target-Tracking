
from pathlib import Path
from dataclasses import asdict
import json
import math
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, Sampler
from .actions import action_id, mirror_action, ACTION_SCHEMA
from .config import canonical_hash, sha256_file
from .supervision import reject_proxy_fields, validate_caption, validate_episode_sample

def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError(f"Data paths must be relative and inside dataset root: {relative}")
    return path

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]

def prepare(root, episodes_file, out_dir, cfg, seed=42):
    """Raw input: one episode object per JSONL line; no image or label rewriting."""
    root, out_dir = Path(root), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()): raise FileExistsError("Use an empty output folder; never overwrite a split")
    episodes = read_jsonl(episodes_file)
    if len(episodes) < 3: raise ValueError("Need at least 3 independent episodes for train/val/test")
    seen, image_owners, asset_hashes = set(), {}, {}
    for ep in episodes:
        reject_proxy_fields(ep)
        eid = ep["episode_id"]
        if not isinstance(eid, str) or not eid or eid in seen: raise ValueError("Unique nonempty episode_id required")
        seen.add(eid)
        if ep.get("action_schema") != ACTION_SCHEMA: raise ValueError("Action schema missing/mismatch")
        if ep.get("task") not in ("avoidance", "tracking"): raise ValueError("Unknown task")
        if not isinstance(ep.get("instruction"), str) or not ep["instruction"].strip(): raise ValueError("Missing task instruction")
        action_id(ep["initial_action_id"])
        frames = ep["frames"]
        if not frames: raise ValueError(f"Empty episode: {eid}")
        last_t = None
        for f in frames:
            reject_proxy_fields(f)
            action_id(f["action_id"])
            t = f["timestamp_s"]
            if isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t): raise ValueError("Invalid timestamp")
            if last_t is not None and abs(t-last_t-cfg.frame_period_s) > 0.02*cfg.frame_period_s:
                raise ValueError(f"Missing/irregular frame in {eid}; resample explicitly before preparing")
            last_t = t
            if not isinstance(f.get("collision"), bool): raise ValueError("collision must be an explicit boolean")
            rel = f["image"]
            if rel in image_owners: raise ValueError("Each captured frame path must occur exactly once")
            image_owners[rel] = eid
            p = safe_path(root, rel)
            with Image.open(p) as im: im.verify()
            asset_hashes[rel] = sha256_file(p)
        # The action must be held over each 0.5 s interval in this sampling contract.
        for i, f in enumerate(frames):
            if f["action_id"] != frames[(i//cfg.keyframe_stride)*cfg.keyframe_stride]["action_id"]:
                raise ValueError(f"Actions in {eid} must be held between decision frames; check collection timing")
    ids = sorted(seen)
    random.Random(seed).shuffle(ids)
    n_train = min(len(ids)-2, max(1, int(len(ids)*0.7)))
    n_val = min(len(ids)-n_train-1, max(1, int(len(ids)*0.15)))
    splits = {"train": ids[:n_train], "val": ids[n_train:n_train+n_val], "test": ids[n_train+n_val:]}
    # Prevent byte-identical RGB leakage across splits. Repeated frames within
    # an episode/split are allowed but must be reviewed for duplicate sampling.
    hash_split = {}
    for split, split_ids in splits.items():
        for rel, owner in image_owners.items():
            if owner not in split_ids: continue
            h = asset_hashes[rel]
            if h in hash_split and hash_split[h] != split: raise ValueError("Identical image bytes cross dataset splits")
            hash_split[h] = split
    dataset_hash = canonical_hash({"episodes": episodes, "assets": asset_hashes})
    staged = {}
    for split, split_ids in splits.items():
        samples = []
        for ep in episodes:
            if ep["episode_id"] not in split_ids: continue
            fs = ep["frames"]
            for i in range(0, len(fs), cfg.keyframe_stride):
                m, k, s = cfg.memory_frames, cfg.prediction_steps, cfg.keyframe_stride
                history_idx = [max(0, i-(m-j)*s) for j in range(m+1)]
                f = fs[i]
                sample = {"sample_id": f"{ep['episode_id']}:{i}", "episode_id": ep["episode_id"],
                          "task": ep["task"], "timestamp_s": f["timestamp_s"],
                          "history": [fs[j]["image"] for j in history_idx],
                          "instruction": ep["instruction"], "instruction_mirrored": ep.get("instruction_mirrored"),
                          "caption": f.get("caption"), "caption_mirrored": f.get("caption_mirrored"),
                          "target": f["action_id"], "last_action": fs[i-1]["action_id"] if i else ep["initial_action_id"]}
                future = [j for j in range(i+1, min(len(fs), i+math.ceil(5.0/cfg.frame_period_s)+2))
                          if fs[j]["timestamp_s"] <= f["timestamp_s"]+5.0+1e-6]
                collision = any(fs[j]["collision"] for j in future)
                full_risk_window = fs[-1]["timestamp_s"] >= f["timestamp_s"]+5.0-1e-6
                if i >= m*s and i+k*s < len(fs) and (collision or full_risk_window):
                    idx = [i+(j-m)*s for j in range(m+1+k)]
                    sample.update(world_images=[fs[j]["image"] for j in idx],
                                  world_actions=[fs[j]["action_id"] for j in idx], risk=int(collision))
                samples.append(sample)
        staged[split] = {"schema": "oceanvla.dataset.v1", "split": split,
                         "dataset_hash": dataset_hash, "split_seed": seed,
                         "episodes": split_ids, "model_config": asdict(cfg),
                         "assets": {r: h for r, h in asset_hashes.items() if image_owners[r] in split_ids},
                         "samples": samples}
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, doc in staged.items():
        (out_dir/f"{split}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {split: {"episodes": len(doc["episodes"]), "policy_samples": len(doc["samples"]),
                       "world_samples": sum("world_images" in x for x in doc["samples"]),
                       "clip_pairs": sum(bool(x.get("caption")) for x in doc["samples"])} for split, doc in staged.items()}
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def prepare_clip(root, pairs_file, out_dir, cfg, seed=42):
    """Separate public/synthetic maritime caption corpus, split by source group.

    A group is a complete source video/episode (not each frame). Synthetic
    group IDs must match policy episode IDs so overlap checks remain possible.
    """
    root, out_dir = Path(root), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()): raise FileExistsError("Use an empty output folder")
    pairs = read_jsonl(pairs_file)
    groups = sorted({p["group_id"] for p in pairs})
    if len(groups) < 3: raise ValueError("Need at least three independent caption source groups")
    assets, paths, content_groups = {}, set(), {}
    for p in pairs:
        validate_caption(p)
        for key in ("group_id", "caption", "caption_mirrored", "source", "license"):
            if not isinstance(p.get(key), str) or not p[key].strip(): raise ValueError(f"Missing {key}")
        rel = p["image"]
        if rel in paths: raise ValueError("Duplicate caption image; deduplicate before splitting")
        paths.add(rel)
        path = safe_path(root, rel)
        with Image.open(path) as image: image.verify()
        digest = sha256_file(path)
        if digest in content_groups and content_groups[digest] != p["group_id"]:
            raise ValueError("Same image content assigned to different caption groups")
        content_groups[digest] = p["group_id"]; assets[rel] = digest
    random.Random(seed).shuffle(groups)
    n = min(len(groups)-2, max(1, int(.7*len(groups))))
    nv = min(len(groups)-n-1, max(1, int(.15*len(groups))))
    splits = {"train":groups[:n], "val":groups[n:n+nv], "test":groups[n+nv:]}
    digest = canonical_hash({"pairs":pairs,"assets":assets})
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, ids in splits.items():
        rows = [p for p in pairs if p["group_id"] in ids]
        doc = {"schema":"oceanvla.dataset.v1", "split":split, "dataset_hash":digest,
               "corpus_kind":"caption_pairs", "episodes":ids, "split_seed":seed, "model_config":asdict(cfg),
               "assets":{p["image"]:assets[p["image"]] for p in rows},
               "samples":[{"sample_id":str(i),"episode_id":p["group_id"],"history":[p["image"]],
                           "caption":p["caption"],"caption_mirrored":p["caption_mirrored"],
                           "source":p["source"],"license":p["license"],
                           "source_kind":p["source_kind"],"annotation_method":p["annotation_method"]} for i,p in enumerate(rows)]}
        (out_dir/f"{split}.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding="utf-8")
        summary[split] = {"groups":len(ids),"clip_pairs":len(rows)}
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def image_sequence(paths, size, *, augment=False, flip=False):
    """Same crop/jitter/flip for every frame; noise is per pixel/frame."""
    area = random.uniform(0.8, 1.0) if augment else 1.0
    edge = math.sqrt(area)
    ox, oy = random.random()*(1-edge), random.random()*(1-edge)
    brightness, contrast = (random.uniform(.8, 1.2), random.uniform(.85, 1.15)) if augment else (1., 1.)
    tensors = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            w, h = image.size
            image = image.crop((int(ox*w), int(oy*h), max(int((ox+edge)*w), 1), max(int((oy+edge)*h), 1)))
            image = image.resize((size, size), Image.Resampling.BICUBIC)
            if flip: image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if augment:
                image = ImageEnhance.Brightness(image).enhance(brightness)
                image = ImageEnhance.Contrast(image).enhance(contrast)
            x = torch.from_numpy(np.array(image, dtype=np.float32).copy()).permute(2, 0, 1)/255.
            if augment: x = (x + torch.randn_like(x)*.05).clamp(0, 1)
            tensors.append((x-MEAN)/STD)
    return torch.stack(tensors)

class EpisodeDataset(Dataset):
    def __init__(self, root, manifest, cfg, stage, tokenizer=None, augment=False, verify_assets=True):
        self.root, self.cfg, self.stage = Path(root), cfg, stage
        self.tokenizer, self.augment = tokenizer, augment
        self.doc = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if self.doc.get("schema") != "oceanvla.dataset.v1": raise ValueError("Unknown dataset schema")
        if stage == "clip" and self.doc.get("corpus_kind") != "caption_pairs":
            raise ValueError("CLIP training requires the separate maritime caption corpus")
        if self.doc.get("corpus_kind") == "caption_pairs" and stage != "clip": raise ValueError("Caption-only corpus cannot train a policy/world model")
        for row in self.doc["samples"]:
            if stage == "clip": validate_caption(row)
            else: validate_episode_sample(row, cfg)
        # Network capacity is not a dataset property. Architecture changes must
        # not require editing otherwise identical immutable sample manifests.
        data_keys = ("image_size", "memory_frames", "keyframe_stride", "frame_period_s",
                     "prediction_steps", "max_text_length", "normalization")
        if any(self.doc["model_config"].get(k) != getattr(cfg, k) for k in data_keys):
            raise ValueError("Prepared data/preprocessing or time configuration differ")
        if augment and self.doc["split"] != "train": raise ValueError("No augmentation outside training")
        self.rows = [r for r in self.doc["samples"] if (stage != "world" or "world_images" in r) and (stage != "clip" or r.get("caption"))]
        if not self.rows: raise ValueError(f"No valid {stage} samples in {manifest}")
        if stage != "world" and tokenizer is None: raise ValueError("CLIP tokenizer is required")
        if verify_assets:
            for rel, expected in self.doc["assets"].items():
                if sha256_file(safe_path(self.root, rel)) != expected: raise ValueError(f"Data changed since preparation: {rel}")
        if augment and stage != "world":
            key = "caption_mirrored" if stage == "clip" else "instruction_mirrored"
            if any(not isinstance(r.get(key), str) or not r[key].strip() for r in self.rows):
                raise ValueError(f"Training flip requires author-provided {key}; do not guess directional language")

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        flip = self.augment and random.random() < .5
        paths = r["world_images"] if self.stage == "world" else r["history"]
        if self.stage == "clip": paths = paths[-1:]
        images = image_sequence([safe_path(self.root, p) for p in paths], self.cfg.image_size, augment=self.augment, flip=flip)
        if self.stage == "world":
            actions = [mirror_action(a) if flip else a for a in r["world_actions"]]
            return {"images": images, "expert_actions": torch.tensor(actions), "risk_labels": torch.tensor(r["risk"], dtype=torch.float32)}
        key = "caption" if self.stage == "clip" else "instruction"
        text = r[key+"_mirrored"] if flip else r[key]
        enc = self.tokenizer(text, padding="max_length", max_length=self.cfg.max_text_length,
                             truncation=True, return_tensors="pt")
        result = {"images": images[0] if self.stage == "clip" else images,
                  "input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0]}
        if self.stage == "policy":
            result.update(last_action=torch.tensor(mirror_action(r["last_action"]) if flip else r["last_action"]),
                          targets=torch.tensor(mirror_action(r["target"]) if flip else r["target"]))
        return result

class EpochSampler(Sampler):
    """Deterministic class balancing within each effective optimizer batch.

    Six class counts differ by at most one before stochastic augmentation.
    128 is not divisible by six; remainder classes rotate across batches/epochs.
    """
    def __init__(self, dataset, seed, balanced=False, batch_size=None):
        self.dataset, self.seed, self.balanced, self.epoch = dataset, seed, balanced, 0
        self.batch_size = batch_size
        self.groups = [[i for i, r in enumerate(dataset.rows) if r["target"] == c] for c in range(6)] if balanced else []
        if balanced and any(not g for g in self.groups): raise ValueError("Class-balanced training needs all six classes")

    def __len__(self): return len(self.dataset)

    def __iter__(self):
        rng = random.Random(self.seed+self.epoch)
        if self.balanced:
            ids = []
            size = self.batch_size or len(self)
            for offset in range(0, len(self), size):
                n = min(size, len(self)-offset)
                classes = [(offset+i+self.epoch) % 6 for i in range(n)]
                batch = [rng.choice(self.groups[c]) for c in classes]
                rng.shuffle(batch)
                ids.extend(batch)
        else:
            ids = list(range(len(self)))
            rng.shuffle(ids)
        return iter(ids)

def assert_disjoint(train, val):
    if train.doc["split"] != "train" or val.doc["split"] != "val": raise ValueError("Expected train and val manifests, never test for training")
    if train.doc["dataset_hash"] != val.doc["dataset_hash"]: raise ValueError("Train/val must come from the same preparation")
    if set(train.doc["episodes"]) & set(val.doc["episodes"]): raise ValueError("Episode leakage")
