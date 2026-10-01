
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from .config import load_config, sha256_file
from .supervision import CAPTION_COUNTS, validate_caption, validate_episode_sample


def preflight(config, episode_dir, caption_dir):
    cfg, training = load_config(config)
    errors, warnings, manifests, hashes = [], [], {}, {}
    caption_counts = dict.fromkeys(CAPTION_COUNTS, 0)
    groups = {"episodes": Path(episode_dir), "captions": Path(caption_dir)}
    for kind, folder in groups.items():
        for split in ("train", "val", "test"):
            path = folder / (split + ".json")
            name = f"{kind}/{split}"
            if not path.is_file():
                errors.append(f"Missing {name} manifest: {path}")
                continue
            hashes[name] = sha256_file(path)
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                errors.append(f"Invalid {name}: {exc}")
                continue
            rows = doc.get("samples", [])
            item = {"samples": len(rows), "schema": doc.get("schema"),
                    "episodes": doc.get("episodes", []), "assets": doc.get("assets", {}),
                    "dataset_hash": doc.get("dataset_hash")}
            manifests[name] = item
            if doc.get("schema") != "oceanvla.dataset.v1":
                item["expert_action_present"] = sum(r.get("expert_action_id") is not None for r in rows)
                item["collision_present"] = sum(r.get("collision") is not None for r in rows)
                errors.append(f"{name}: requires expert episode/caption schema; found {doc.get('schema')}; no relabeling performed")
                continue
            if doc.get("split") != split:
                errors.append(f"{name}: split identity mismatch")
            for key in ("image_size", "memory_frames", "keyframe_stride", "frame_period_s",
                        "prediction_steps", "max_text_length", "normalization"):
                if doc.get("model_config", {}).get(key) != getattr(cfg, key):
                    errors.append(f"{name}: incompatible preprocessing/time field {key}")
            if not rows:
                errors.append(f"{name}: empty dataset")
            for row in rows:
                try:
                    if kind == "episodes": validate_episode_sample(row, cfg)
                    else:
                        validate_caption(row)
                        caption_counts[row["source_kind"]] += 1
                except (ValueError, TypeError) as exc:
                    errors.append(f"{name}: {exc}")
                    break
            if kind == "episodes":
                counts = [sum(type(r.get("target")) is int and r["target"] == c for r in rows) for c in range(6)]
                item["class_counts"] = counts
                if sum(counts) != len(rows):
                    errors.append(f"{name}: missing/invalid expert target")
                if any("pseudo_action_id" in r for r in rows):
                    errors.append(f"{name}: pseudo-action records cannot be used by the full pipeline")
                world = [r for r in rows if "world_images" in r]
                risks = [sum(r.get("risk") == c for r in world) for c in (0, 1)]
                item["world_samples"], item["risk_counts"] = len(world), risks
                if split in ("train", "val") and (not world or not all(risks) or sum(risks) != len(world)):
                    errors.append(f"{name}: world/risk training and AUROC require actual positive and negative collision labels")
                if split == "train":
                    if not all(counts):
                        errors.append(f"{name}: six-action balancing requires every class, including STOP")
                    if any(not r.get("instruction_mirrored") for r in rows):
                        errors.append(f"{name}: mirrored instruction required for directional augmentation")
                    if len(world) < training["world"]["global_batch"] or len(rows) < training["policy"]["global_batch"]:
                        errors.append(f"{name}: fewer samples than a global batch")
                elif not all(counts):
                    warnings.append(f"{name}: some actions absent; report class support and undefined recall")
            else:
                if doc.get("corpus_kind") != "caption_pairs":
                    errors.append(f"{name}: requires separate maritime image-text corpus")
                if any(not r.get("caption") for r in rows):
                    errors.append(f"{name}: missing maritime caption")
                if split == "train":
                    if any(not r.get("caption_mirrored") for r in rows):
                        errors.append(f"{name}: mirrored captions required")
                    if len(rows) < training["clip"]["global_batch"]:
                        errors.append(f"{name}: insufficient examples for contrastive batch")
    for kind in groups:
        docs = [manifests.get(f"{kind}/{split}") for split in ("train", "val", "test")]
        if all(docs):
            if len({d["dataset_hash"] for d in docs}) != 1:
                errors.append(f"{kind}: split dataset identities differ")
            for i in range(3):
                for j in range(i+1, 3):
                    if set(docs[i]["episodes"]) & set(docs[j]["episodes"]):
                        errors.append(f"{kind}: overlapping episodes across splits")
                    if set(docs[i]["assets"].values()) & set(docs[j]["assets"].values()):
                        errors.append(f"{kind}: identical image content across splits")
    caption_train = manifests.get("captions/train", {})
    for split in ("val", "test"):
        ep = manifests.get(f"episodes/{split}", {})
        if set(caption_train.get("episodes", [])) & set(ep.get("episodes", [])) or set(caption_train.get("assets", {}).values()) & set(ep.get("assets", {}).values()):
            errors.append(f"Maritime CLIP training overlaps policy {split}")
    for kind, expected in CAPTION_COUNTS.items():
        if caption_counts[kind] != expected:
            errors.append(f"Expected {expected} {kind} caption pairs; found {caption_counts[kind]}")
    summaries = {k: {a:b for a,b in v.items() if a not in ("assets", "episodes")} for k,v in manifests.items()}
    return {"ready": not errors, "errors": errors, "warnings": warnings, "manifests": summaries,
            "input_sha256": hashes, "model_config": asdict(cfg), "training": training,
            "data_policy": "read_only", "world_architecture": cfg.world_arch,
            "caption_source_counts": caption_counts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/train1.yaml")
    p.add_argument("--episodes", required=True, help="Existing train/val/test episode manifest directory")
    p.add_argument("--captions", required=True, help="Existing train/val/test caption manifest directory")
    p.add_argument("--data-root", required=True)
    p.add_argument("--caption-root")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--report")
    a = p.parse_args()
    result = preflight(a.config, a.episodes, a.captions)
    if a.report:
        report = Path(a.report)
        protected = [Path(a.episodes).resolve(), Path(a.captions).resolve(), Path(a.data_root).resolve()]
        if a.caption_root:
            protected.append(Path(a.caption_root).resolve())
        if any(report.resolve().is_relative_to(root) for root in protected):
            raise ValueError("Report must be outside read-only dataset directories")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["ready"]:
        return 2
    if a.check_only:
        return 0
    output = Path(a.output).resolve()
    roots = [Path(x).resolve() for x in (a.episodes, a.captions, a.data_root, a.caption_root or a.data_root)]
    if any(output.is_relative_to(root) for root in roots):
        raise ValueError("Training output must be outside read-only dataset directories")
    if output.exists():
        raise FileExistsError("Use a new run directory; resume individual stages explicitly")
    output.mkdir(parents=True)
    (output/"preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    def run(*args):
        subprocess.run([sys.executable, "-m", "core.cli", *map(str, args)], check=True)
    for stage in ("world", "clip", "policy"):
        folder = Path(a.captions if stage == "clip" else a.episodes)
        root = a.caption_root or a.data_root if stage == "clip" else a.data_root
        extra = ["--world-checkpoint", output/"world/best.pt", "--clip-checkpoint", output/"clip/best.pt"] if stage == "policy" else []
        run("train", "--stage", stage, "--config", a.config, "--data-root", root,
            "--train-manifest", folder/"train.json", "--val-manifest", folder/"val.json",
            "--output", output/stage, "--device", a.device, *extra)
        run("inspect", "--checkpoint", output/stage/"best.pt")
    run("evaluate", "--checkpoint", output/"policy/best.pt", "--data-root", a.data_root,
        "--manifest", Path(a.episodes)/"test.json", "--output", output/"test_metrics.json", "--device", a.device)
    (output/"COMPLETE.json").write_text(json.dumps({"policy": "policy/best.pt", "offline_metrics": "test_metrics.json",
        "closed_loop_evaluated": False}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
