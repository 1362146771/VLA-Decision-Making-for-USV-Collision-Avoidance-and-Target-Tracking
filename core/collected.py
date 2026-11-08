import json
from pathlib import Path
from .config import canonical_hash, sha256_file
from .partition import make_partition, partition_counts
from .telemetry import infer


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def captured_episodes(root):
    reference = root / "verified_references.json"
    audited = reference.is_file() or any(root.glob("*/attempt_*/colregs_audit.json"))
    if audited and not (root / "COMPLETE.json").is_file():
        raise ValueError(f"Encounter campaign is incomplete: {root.name}")
    if reference.is_file():
        for item in read(reference):
            if not item["audit"]["accepted"]:
                continue
            source = Path(item["directory"])
            for name, expected in item["files_sha256"].items():
                if sha256_file(source / name) != expected:
                    raise ValueError("Referenced capture content changed")
            yield source
    for file in sorted(root.glob("*/attempt_*/summary.json")):
        result = read(file)
        if result.get("status") != "captured" or not result.get("rgb_available"):
            continue
        if audited:
            audit = file.parent / "colregs_audit.json"
            if not audit.is_file() or not read(audit)["accepted"]:
                continue
        yield file.parent


def index_capture(campaign, source, cfg):
    scenario_path = source / "scenario.json"
    frame_path = source / "frames.jsonl"
    scenario = read(scenario_path)
    report = read(source / "summary.json")
    eid = f"{campaign}/{scenario['episode_id']}"
    with frame_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != report["frame_count"] or any(row["frame_index"] != i for i, row in enumerate(rows)):
        raise ValueError(f"Incomplete capture: {eid}")
    for row in rows:
        if not row.get("image") or not (source / row["image"]).is_file():
            raise ValueError(f"Missing capture image: {eid}")
    labels = [infer(row, scenario) for row in rows]
    stride = cfg.keyframe_stride
    start, tail = cfg.memory_frames * stride, cfg.prediction_steps * stride
    period = cfg.frame_period_s * stride
    decisions = []
    for i in range(start, len(rows) - tail, stride):
        indices = range(i - start, i + tail + 1, stride)
        if any(labels[j] is None or labels[j]["executed_proxy"] is None for j in indices):
            continue
        stamps = [rows[j]["host_monotonic_s"] for j in indices]
        if any(not period - .2 < b - a < period + .2 for a, b in zip(stamps, stamps[1:])):
            continue
        decisions.append({"sample_id": f"{eid}:{i}", "frame_index": i,
                          "host_monotonic_s": rows[i]["host_monotonic_s"]})
    if not decisions:
        raise ValueError(f"No usable windows: {eid}")
    return {"episode_id": eid, "task": scenario["task"],
            "stratum": scenario.get("encounter_type", "general"),
            "frame_count": len(rows), "usable_windows": len(decisions),
            "source_attempt": str(source.resolve()), "scenario_sha256": sha256_file(scenario_path),
            "frames_sha256": sha256_file(frame_path), "decisions": decisions,
            "recorded_action_frames": sum(row.get("action_id") is not None for row in rows),
            "collision_event_frames": sum(type(row.get("collision")) is bool for row in rows)}


def prepare_collected(campaigns, output, cfg, seed=42):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    episodes = []
    for campaign in map(Path, campaigns):
        if not campaign.is_dir():
            raise FileNotFoundError(campaign)
        episodes.extend(index_capture(campaign.name, source, cfg) for source in captured_episodes(campaign))
    if len({e["frames_sha256"] for e in episodes}) != len(episodes):
        raise ValueError("Duplicate raw trajectories")
    splits = make_partition(episodes, seed)
    counts = partition_counts(episodes, splits)
    inventory = {e["episode_id"]: e for e in episodes}
    summary = {"schema": "oceanvla.collected_partition.v1", "split_basis": "raw_rgb_frames",
               "seed": seed, "paper_dataset": False, "frame_count": sum(e["frame_count"] for e in episodes),
               "episode_count": len(episodes), "usable_windows": sum(e["usable_windows"] for e in episodes),
               "recorded_action_frames": sum(e["recorded_action_frames"] for e in episodes),
               "collision_event_frames": sum(e["collision_event_frames"] for e in episodes),
               "splits": {}}
    partition = {"splits": splits, "seed": seed, "counts": counts,
                 "input_trace_sha256": {e["episode_id"]: e["frames_sha256"] for e in episodes}}
    summary["partition_sha256"] = canonical_hash(partition)
    output.mkdir(parents=True)
    for split, ids in splits.items():
        group = [inventory[eid] for eid in ids]
        frames = sum(e["frame_count"] for e in group)
        summary["splits"][split] = {"episodes": len(group), "raw_frames": frames,
                                    "frame_fraction": frames / summary["frame_count"],
                                    "usable_windows": sum(e["usable_windows"] for e in group),
                                    "by_task": counts[split]}
        manifest = {"schema": "oceanvla.collected_support.v1", "split": split,
                    "partition_sha256": summary["partition_sha256"], "episodes": group}
        manifest["decision_support_sha256"] = canonical_hash({e["episode_id"]: e["decisions"] for e in group})
        (output / f"{split}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "partition.json").write_text(json.dumps(partition, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
