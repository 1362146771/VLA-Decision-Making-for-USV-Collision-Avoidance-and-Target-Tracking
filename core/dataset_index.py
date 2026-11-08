import json
from pathlib import Path
from .config import canonical_hash, sha256_file
from .data import safe_path


def resolve_dataset(path, split="test"):
    path = Path(path)
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("schema") != "oceanvla.dataset_index.v1":
        raise ValueError("schema")
    partition_path = safe_path(path.parent, index["partition"])
    if sha256_file(partition_path) != index["partition_file_sha256"]:
        raise ValueError("partition_file_sha256")
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    if canonical_hash(partition) != index["partition_sha256"]:
        raise ValueError("partition_sha256")
    groups = partition["splits"]
    if set(groups) != {"train", "val", "test"} or set(index["splits"]) != set(groups):
        raise ValueError("splits")
    ids = [eid for values in groups.values() for eid in values]
    if len(ids) != len(set(ids)) or len(ids) != index["episode_count"]:
        raise ValueError("episode_count")
    if sum(x["raw_frames"] for x in index["splits"].values()) != index["frame_count"]:
        raise ValueError("frame_count")
    if sum(x["usable_windows"] for x in index["splits"].values()) != index["usable_windows"]:
        raise ValueError("usable_windows")
    for name, values in groups.items():
        if len(values) != index["splits"][name]["episodes"]:
            raise ValueError("episodes")
    entry = index["splits"][split]
    manifest = safe_path(path.parent, entry["manifest"])
    if sha256_file(manifest) != entry["manifest_sha256"]:
        raise ValueError("manifest_sha256")
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    if doc.get("split") != split or doc.get("decision_support_sha256") != entry["decision_support_sha256"]:
        raise ValueError("decision_support_sha256")
    if doc.get("schema") == "oceanvla.collected_support.v1":
        observed = [e["episode_id"] for e in doc["episodes"]]
        support = {e["episode_id"]: e["decisions"] for e in doc["episodes"]}
        frames = sum(e["frame_count"] for e in doc["episodes"])
        count = sum(len(e["decisions"]) for e in doc["episodes"])
    elif doc.get("schema") == "oceanvla.dataset.v1":
        observed = doc["episodes"]
        support = doc["decision_support"]
        frames = sum(e["raw_frames"] for e in support.values())
        count = len(doc["samples"])
    else:
        raise ValueError("schema")
    if len(observed) != len(set(observed)) or set(observed) != set(groups[split]):
        raise ValueError("episodes")
    if frames != entry["raw_frames"] or count != entry["usable_windows"]:
        raise ValueError("split_counts")
    if canonical_hash(support) != entry["decision_support_sha256"]:
        raise ValueError("decision_support_sha256")
    return manifest
