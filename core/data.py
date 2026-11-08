
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
from .supervision import CAPTION_COUNTS, flip_allowed, reject_proxy_fields, validate_caption, validate_episode_sample
from .partition import make_partition, partition_counts

def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError(f"Data paths must be relative and inside dataset root: {relative}")
    return path

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]

def prepare(root, episodes_file, out_dir, cfg, seed=42, *, paper=False, partition=None):
    """Raw input: one episode object per JSONL line; no image or label rewriting."""
    root, out_dir = Path(root), Path(out_dir)
    if paper and partition is None:
        raise ValueError("--paper requires --partition")
    if paper and (cfg.memory_frames, cfg.keyframe_stride, cfg.frame_period_s, cfg.prediction_steps) != (3, 5, .1, 4):
        raise ValueError("The paper profile requires 3 history frames, stride 5, 10 Hz and 4 rollout steps")
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
    inventory = [{"episode_id": e["episode_id"], "task": e["task"], "frame_count": len(e["frames"])}
                 for e in episodes]
    fixed = json.loads(Path(partition).read_text(encoding="utf-8"))["splits"] if partition else None
    splits = make_partition(inventory, seed, paper=paper, fixed=fixed)
    counts = partition_counts(inventory, splits)
    # Prevent byte-identical RGB leakage across splits. Repeated frames within
    # an episode/split are allowed but must be reviewed for duplicate sampling.
    hash_split = {}
    for split, split_ids in splits.items():
        for rel, owner in image_owners.items():
            if owner not in split_ids: continue
            h = asset_hashes[rel]
            if h in hash_split and hash_split[h] != split: raise ValueError("Identical image bytes cross dataset splits")
            hash_split[h] = split
    partition_record = {"splits": splits, "counts": counts, "seed": seed, "paper_counts": paper,
                        "episodes_file_sha256": sha256_file(episodes_file)}
    partition_hash = canonical_hash(partition_record)
    dataset_hash = canonical_hash({"episodes": episodes, "assets": asset_hashes, "partition": partition_hash})
    staged = {}
    for split, split_ids in splits.items():
        samples, support = [], {}
        for ep in episodes:
            if ep["episode_id"] not in split_ids: continue
            fs = ep["frames"]
            episode_flip = flip_allowed(ep) and not any(
                f.get("allow_horizontal_flip") is False or f.get("dynamic_encounter") is True
                or f.get("asymmetric_give_way") is True
                or f.get("encounter_type") not in (None, "none") for f in fs)
            m, k, s = cfg.memory_frames, cfg.prediction_steps, cfg.keyframe_stride
            eid = ep["episode_id"]
            indices = range(0, len(fs), s) if split == "train" else range(m*s, len(fs)-k*s, s)
            if split != "train" and not indices:
                raise ValueError(f"No valid evaluation decisions in {eid}")
            support[eid] = {"task": ep["task"], "raw_frames": len(fs), "decisions": []}
            for i in indices:
                history_idx = [max(0, i-(m-j)*s) for j in range(m+1)]
                f = fs[i]
                sample = {"sample_id": f"{eid}:{i}", "episode_id": eid, "frame_index": i,
                          "task": ep["task"], "timestamp_s": f["timestamp_s"],
                          "history": [fs[j]["image"] for j in history_idx],
                          "instruction": ep["instruction"], "instruction_mirrored": ep.get("instruction_mirrored"),
                          "caption": f.get("caption"), "caption_mirrored": f.get("caption_mirrored"),
                          "target": f["action_id"], "last_action": fs[i-1]["action_id"] if i else ep["initial_action_id"]}
                for key in ("allow_horizontal_flip", "scene_type", "laterally_symmetric", "dynamic_encounter", "encounter_type", "asymmetric_give_way"):
                    if key in ep:
                        sample[key] = ep[key]
                sample["allow_horizontal_flip"] = episode_flip
                support[eid]["decisions"].append({key: sample[key] for key in ("sample_id", "frame_index", "timestamp_s")})
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
                         "partition_sha256": partition_hash, "raw_counts": counts[split],
                         "decision_support": support, "decision_support_sha256": canonical_hash(support),
                         "episodes": split_ids, "model_config": asdict(cfg),
                         "assets": {r: h for r, h in asset_hashes.items() if image_owners[r] in split_ids},
                         "asset_episodes": {r: owner for r, owner in image_owners.items() if owner in split_ids},
                         "samples": samples}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"partition.json").write_text(json.dumps(partition_record, ensure_ascii=False, indent=2), encoding="utf-8")
    for split, doc in staged.items():
        (out_dir/f"{split}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {split: {"episodes": len(doc["episodes"]),
                       "raw_frames": sum(x["frames"] for x in doc["raw_counts"].values()),
                       "by_task": doc["raw_counts"], "policy_samples": len(doc["samples"]),
                       "world_samples": sum("world_images" in x for x in doc["samples"]),
                       "clip_pairs": sum(bool(x.get("caption")) for x in doc["samples"])} for split, doc in staged.items()}
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def prepare_clip(root, pairs_file, out_dir, cfg, seed=42, *, episode_manifests=None):
    """Separate public/synthetic maritime caption corpus, split by source group.

    A group is a complete source video/episode (not each frame). Synthetic
    group IDs must match policy episode IDs so overlap checks remain possible.
    """
    root, out_dir = Path(root), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()): raise FileExistsError("Use an empty output folder")
    pairs = read_jsonl(pairs_file)
    groups = sorted({p["group_id"] for p in pairs})
    if not episode_manifests and len(groups) < 3:
        raise ValueError("Need at least three independent caption source groups")
    assets, paths, content_groups = {}, set(), {}
    for p in pairs:
        validate_caption(p)
        for key in ("group_id", "caption", "source", "license"):
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
    if episode_manifests:
        docs = {s: json.loads((Path(episode_manifests)/f"{s}.json").read_text(encoding="utf-8"))
                for s in ("train", "val", "test")}
        episode_split, image_owners = {}, {}
        for split, doc in docs.items():
            if doc["split"] != split or set(doc["episodes"]) & episode_split.keys():
                raise ValueError("Invalid episode split identity")
            episode_split.update(dict.fromkeys(doc["episodes"], split))
            for rel, digest in doc["assets"].items():
                image_owners.setdefault(digest, set()).add(doc["asset_episodes"][rel])
        assignments = {}
        train_counts = dict.fromkeys(CAPTION_COUNTS, 0)
        for pair in pairs:
            split = pair.get("split", "train")
            gid = pair["group_id"]
            if split not in docs or assignments.setdefault(gid, split) != split:
                raise ValueError("Caption group crosses splits")
            owners = image_owners.get(assets[pair["image"]], set())
            if pair["source_kind"] == "synthetic":
                if episode_split.get(gid) != split or gid not in owners or any(episode_split[o] != split for o in owners):
                    raise ValueError("Synthetic caption must match its source episode and split")
            elif gid in episode_split or owners:
                raise ValueError("Public caption overlaps simulator episodes")
            if split == "train":
                train_counts[pair["source_kind"]] += 1
        if train_counts != CAPTION_COUNTS:
            raise ValueError(f"Expected training caption counts {CAPTION_COUNTS}; found {train_counts}")
        splits = {s: [gid for gid in groups if assignments[gid] == s] for s in docs}
        if not splits["val"]:
            raise ValueError("Supply separate validation caption pairs")
    else:
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
                           "caption":p["caption"],"caption_mirrored":p.get("caption_mirrored"),
                           **{key: p[key] for key in ("allow_horizontal_flip", "scene_type", "laterally_symmetric",
                                                     "dynamic_encounter", "encounter_type", "asymmetric_give_way", "task") if key in p},
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

def policy_prompt(instruction, mode, task=None):
    if mode == "minimal":
        return "Navigate."
    if mode == "minimal_task":
        prompts = {"avoidance": "avoid collisions", "tracking": "track target"}
        if task not in prompts:
            raise ValueError("Task-specific minimal prompts require task=avoidance or task=tracking")
        return prompts[task]
    return instruction


class EpisodeDataset(Dataset):
    def __init__(self, root, manifest, cfg, stage, tokenizer=None, augment=False, verify_assets=True, prediction_only=False):
        self.root, self.cfg, self.stage = Path(root), cfg, stage
        self.tokenizer, self.augment = tokenizer, augment
        self.prediction_only = prediction_only
        if prediction_only and (stage != "policy" or augment):
            raise ValueError("prediction_only")
        self.doc = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if self.doc.get("schema") != "oceanvla.dataset.v1": raise ValueError("Unknown dataset schema")
        if stage == "clip" and self.doc.get("corpus_kind") != "caption_pairs":
            raise ValueError("CLIP training requires the separate maritime caption corpus")
        if self.doc.get("corpus_kind") == "caption_pairs" and stage != "clip": raise ValueError("Caption-only corpus cannot train a policy/world model")
        prepared = type(cfg)(**self.doc["model_config"]).validate()
        for row in self.doc["samples"]:
            if stage == "clip": validate_caption(row)
            elif prediction_only:
                reject_proxy_fields(row)
                action_id(row["last_action"])
                if len(row["history"]) != prepared.memory_frames + 1:
                    raise ValueError("history")
            else: validate_episode_sample(row, prepared)
        # Network capacity is not a dataset property. Architecture changes must
        # not require editing otherwise identical immutable sample manifests.
        data_keys = ("image_size", "keyframe_stride", "frame_period_s",
                     "prediction_steps", "max_text_length", "normalization")
        if any(self.doc["model_config"].get(k) != getattr(cfg, k) for k in data_keys):
            raise ValueError("Prepared data/preprocessing or time configuration differ")
        if (stage == "world" and prepared.memory_frames != cfg.memory_frames
                or stage == "policy" and prepared.memory_frames < cfg.observation_memory_frames):
            raise ValueError("Prepared history is incompatible with model")
        if augment and self.doc["split"] != "train": raise ValueError("No augmentation outside training")
        self.rows = [r for r in self.doc["samples"] if (stage != "world" or "world_images" in r) and (stage != "clip" or r.get("caption"))]
        if stage == "policy" and self.doc["split"] in ("val", "test"):
            from .evaluation import decision_support
            decision_support(self.doc, self.doc["split"])
        if not self.rows: raise ValueError(f"No valid {stage} samples in {manifest}")
        if stage != "world" and tokenizer is None: raise ValueError("CLIP tokenizer is required")
        if verify_assets:
            for rel, expected in self.doc["assets"].items():
                if sha256_file(safe_path(self.root, rel)) != expected: raise ValueError(f"Data changed since preparation: {rel}")
        if augment and stage != "world":
            key = "caption_mirrored" if stage == "clip" else "instruction_mirrored"
            if any(flip_allowed(r) and (not isinstance(r.get(key), str) or not r[key].strip()) for r in self.rows):
                raise ValueError(f"Training flip requires author-provided {key}; do not guess directional language")

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        flip = self.augment and flip_allowed(r) and random.random() < .5
        paths = r["world_images"] if self.stage == "world" else r["history"]
        if self.stage == "clip": paths = paths[-1:]
        if self.stage == "policy": paths = paths[-(self.cfg.observation_memory_frames+1):]
        images = image_sequence([safe_path(self.root, p) for p in paths], self.cfg.image_size, augment=self.augment, flip=flip)
        if self.stage == "world":
            actions = [mirror_action(a) if flip else a for a in r["world_actions"]]
            return {"images": images, "expert_actions": torch.tensor(actions), "risk_labels": torch.tensor(r["risk"], dtype=torch.float32)}
        key = "caption" if self.stage == "clip" else "instruction"
        text = r[key+"_mirrored"] if flip else r[key]
        if self.stage == "policy":
            text = policy_prompt(text, self.cfg.prompt_mode, r["task"])
        enc = self.tokenizer(text, padding="max_length", max_length=self.cfg.max_text_length,
                             truncation=True, return_tensors="pt")
        result = {"images": images[0] if self.stage == "clip" else images,
                  "input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0]}
        if self.stage == "policy":
            result.update(episode_id=r["episode_id"], task=r["task"],
                          last_action=torch.tensor(mirror_action(r["last_action"]) if flip else r["last_action"]))
            if not self.prediction_only:
                result["targets"] = torch.tensor(mirror_action(r["target"]) if flip else r["target"])
        return result

class EpochSampler(Sampler):
    def __init__(self, dataset, seed, balanced=False, batch_size=None):
        self.dataset, self.seed, self.balanced, self.epoch = dataset, seed, balanced, 0
        self.batch_size = batch_size
        self.groups = [[i for i, r in enumerate(dataset.rows) if r["target"] == c] for c in range(6)] if balanced else []
        if balanced and any(not g for g in self.groups): raise ValueError("Class-balanced training needs all six classes")
        self.weights = [1.0 / len(self.groups[r["target"]]) for r in dataset.rows] if balanced else None

    def __len__(self): return len(self.dataset)

    def __iter__(self):
        rng = random.Random(self.seed+self.epoch)
        if self.balanced:
            ids = rng.choices(range(len(self.dataset.rows)), weights=self.weights, k=len(self))
        else:
            ids = list(range(len(self)))
            rng.shuffle(ids)
        return iter(ids)

def assert_disjoint(train, val):
    if train.doc["split"] != "train" or val.doc["split"] != "val": raise ValueError("Expected train and val manifests, never test for training")
    if train.doc["dataset_hash"] != val.doc["dataset_hash"]: raise ValueError("Train/val must come from the same preparation")
    if set(train.doc["episodes"]) & set(val.doc["episodes"]): raise ValueError("Episode leakage")
