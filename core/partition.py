import random

SPLITS = ("train", "val", "test")
TASKS = ("avoidance", "tracking")
FRAME_RATIOS = (.70, .15, .15)
PAPER_COUNTS = {
    "train": {"avoidance": (118, 17791), "tracking": (84, 12569)},
    "val": {"avoidance": (19, 3812), "tracking": (14, 2694)},
    "test": {"avoidance": (19, 3813), "tracking": (14, 2692)},
}


def partition_counts(episodes, splits):
    by_id = {e["episode_id"]: e for e in episodes}
    return {split: {task: {
        "episodes": sum(by_id[eid]["task"] == task for eid in splits[split]),
        "frames": sum(by_id[eid]["frame_count"] for eid in splits[split] if by_id[eid]["task"] == task),
    } for task in TASKS} for split in SPLITS}


def validate_partition(episodes, splits, paper=False):
    if set(splits) != set(SPLITS):
        raise ValueError("Partition must contain train, val and test")
    ids = [eid for split in SPLITS for eid in splits[split]]
    known = [e["episode_id"] for e in episodes]
    if len(ids) != len(set(ids)) or len(known) != len(set(known)) or set(ids) != set(known):
        raise ValueError("Partition must assign each source episode exactly once")
    if any(not splits[s] for s in SPLITS):
        raise ValueError("Partition contains an empty split")
    if paper:
        counts = partition_counts(episodes, splits)
        for split in SPLITS:
            for task in TASKS:
                observed = counts[split][task]
                if (observed["episodes"], observed["frames"]) != PAPER_COUNTS[split][task]:
                    raise ValueError(f"Paper count mismatch for {split}/{task}")
    return splits


def make_partition(episodes, seed=42, *, paper=False, fixed=None):
    if any(e["task"] not in TASKS or type(e["frame_count"]) is not int or e["frame_count"] < 1 for e in episodes):
        raise ValueError("Each episode needs a supported task and positive frame count")
    if paper and fixed is None:
        raise ValueError("--paper requires --partition")
    if fixed is not None:
        return validate_partition(episodes, fixed, paper)
    rng = random.Random(seed)
    splits = {s: [] for s in SPLITS}
    strata = sorted({(e["task"], e.get("stratum", "")) for e in episodes})
    for task, stratum in strata:
        rows = sorted((e for e in episodes if e["task"] == task and e.get("stratum", "") == stratum),
                      key=lambda e: e["episode_id"])
        rng.shuffle(rows)
        rows.sort(key=lambda e: e["frame_count"], reverse=True)
        total = sum(r["frame_count"] for r in rows)
        targets = [total * fraction for fraction in FRAME_RATIOS]
        sizes, groups = [0, 0, 0], [[], [], []]
        for i, row in enumerate(rows):
            size = row["frame_count"]
            empty = [j for j, g in enumerate(groups) if not g]
            candidates = empty if len(rows) >= 3 and len(rows) - i == len(empty) else range(3)
            j = min(candidates, key=lambda k: ((sizes[k] + size - targets[k]) ** 2
                                              - (sizes[k] - targets[k]) ** 2) / targets[k])
            groups[j].append(row["episode_id"])
            sizes[j] += size
        for split, group in zip(SPLITS, groups):
            splits[split].extend(group)
    return validate_partition(episodes, splits, paper)
