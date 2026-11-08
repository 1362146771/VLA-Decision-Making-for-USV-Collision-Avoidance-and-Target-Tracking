import json
from pathlib import Path
from .actions import action_id
from .config import canonical_hash, sha256_file
from .metrics import episode_accuracy


def decision_support(doc, expected_split="test"):
    if doc.get("schema") != "oceanvla.dataset.v1" or doc.get("corpus_kind") == "caption_pairs":
        raise ValueError("Expected recorded-action episode manifest")
    support = doc.get("decision_support")
    if not support or canonical_hash(support) != doc.get("decision_support_sha256"):
        raise ValueError("Missing or invalid decision support; regenerate with prepare")
    if doc.get("split") != expected_split or set(support) != set(doc["episodes"]):
        raise ValueError("Expected complete held-out decision support")
    cfg = doc["model_config"]
    stride = cfg["keyframe_stride"]
    start = cfg["memory_frames"] * stride
    tail = cfg["prediction_steps"] * stride
    samples = {r["sample_id"]: r for r in doc["samples"]}
    ids = set()
    for eid, episode in support.items():
        expected = list(range(start, episode["raw_frames"] - tail, stride))
        rows = episode["decisions"]
        if not expected or [r["frame_index"] for r in rows] != expected:
            raise ValueError(f"Unexpected evaluation boundaries for {eid}")
        for row in rows:
            sid = row["sample_id"]
            sample = samples.get(sid)
            if sid in ids or sample is None or sample["episode_id"] != eid or sample["task"] != episode["task"]:
                raise ValueError("Decision support does not match manifest")
            if any(sample[key] != row[key] for key in ("frame_index", "timestamp_s")):
                raise ValueError("Decision timestamps do not match manifest")
            ids.add(sid)
    if len(ids) != len(doc["samples"]):
        raise ValueError("Unexpected or duplicate manifest samples")
    return support


def annotate_support(result, support):
    for eid, episode in support.items():
        result["by_episode"][eid]["raw_frames"] = episode["raw_frames"]
        result["by_episode"][eid]["valid_frame_indices"] = [d["frame_index"] for d in episode["decisions"]]
    for task, group in result["by_task"].items():
        group["raw_frames"] = sum(e["raw_frames"] for e in support.values() if e["task"] == task)
    result["raw_frames"] = sum(e["raw_frames"] for e in support.values())
    result["decision_support_sha256"] = canonical_hash(support)


def score_predictions(manifest, predictions, output):
    doc = json.loads(Path(manifest).read_text(encoding="utf-8"))
    support = decision_support(doc)
    received = json.loads(Path(predictions).read_text(encoding="utf-8"))
    if received.get("decision_support_sha256") != canonical_hash(support):
        raise ValueError("Predictions must identify the shared decision support")
    by_id = {}
    for row in received["predictions"]:
        sid = row["sample_id"]
        if sid in by_id or len(row["top3"]) != 3 or len(set(row["top3"])) != 3:
            raise ValueError("Expected unique samples and three distinct predicted actions")
        by_id[sid] = [action_id(a) for a in row["top3"]]
    if set(by_id) != {r["sample_id"] for r in doc["samples"]}:
        raise ValueError("Predictions must cover every shared decision exactly once")
    result = episode_accuracy([{**r, "target": action_id(r["target"]), "top3": by_id[r["sample_id"]]}
                               for r in doc["samples"]])
    annotate_support(result, support)
    result.update(manifest_sha256=sha256_file(manifest), predictions_sha256=sha256_file(predictions))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result
