import math
from collections import defaultdict
import torch

def summarize_confusion(matrix):
    from .actions import ACTION_NAMES
    counts = [sum(row) for row in matrix]
    recalls = [matrix[i][i]/n if n else None for i, n in enumerate(counts)]
    present = [r for r in recalls if r is not None]
    return {"confusion_matrix": matrix, "class_support": dict(zip(ACTION_NAMES, counts)),
            "class_recall": dict(zip(ACTION_NAMES, recalls)),
            "macro_recall_present_classes": sum(present)/len(present) if present else None,
            "missing_classes": [ACTION_NAMES[i] for i, n in enumerate(counts) if not n]}

def classification_counts(logits, targets):
    top = logits.topk(3, -1).indices
    return {"samples": len(targets), "top1_correct": int((top[:, 0] == targets).sum()),
            "top3_correct": int((top == targets[:, None]).any(1).sum())}

def episode_accuracy(predictions):
    episodes = {}
    for row in predictions:
        episode = episodes.setdefault(row["episode_id"], {"task": row["task"], "samples": 0,
                                                           "top1_correct": 0, "top3_correct": 0})
        if episode["task"] != row["task"]:
            raise ValueError("An episode cannot span tasks")
        episode["samples"] += 1
        episode["top1_correct"] += int(row["target"] == row["top3"][0])
        episode["top3_correct"] += int(row["target"] in row["top3"])
    for episode in episodes.values():
        for key in ("top1", "top3"):
            episode[key] = episode[key + "_correct"] / episode["samples"]

    def aggregate(group):
        samples = sum(e["samples"] for e in group)
        result = {"episodes": len(group), "samples": samples}
        for key in ("top1", "top3"):
            hits = sum(e[key + "_correct"] for e in group)
            result[key] = sum(e[key] for e in group) / len(group)
            result[key + "_correct"] = hits
            result["sample_" + key] = hits / samples
        return result

    if not episodes:
        raise ValueError("No evaluated episodes")
    tasks = defaultdict(list)
    for episode in episodes.values():
        tasks[episode["task"]].append(episode)
    return {**aggregate(list(episodes.values())), "accuracy_aggregation": "episode_macro",
            "by_task": {task: aggregate(group) for task, group in tasks.items()}, "by_episode": episodes}

def binary_auc(scores, labels):
    """Rank AUC with ties; null when one class is absent (never fake 0.5)."""
    pairs = sorted(zip(scores, labels))
    npos = sum(int(y) for _, y in pairs)
    nneg = len(pairs)-npos
    if not npos or not nneg: return None
    rank_sum, i = 0., 0
    while i < len(pairs):
        j = i+1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]: j += 1
        rank_sum += ((i+1+j)/2)*sum(int(y) for _, y in pairs[i:j])
        i = j
    return (rank_sum-npos*(npos+1)/2)/(npos*nneg)

def wilson(successes, n):
    if not n: raise ValueError("Empty evaluation")
    p, z = successes/n, 1.959963984540054
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [center-half, center+half]

def two_proportion_z(success_a, n_a, success_b, n_b):
    if not n_a or not n_b: raise ValueError("Empty group")
    p = (success_a+success_b)/(n_a+n_b)
    variance = p*(1-p)*(1/n_a+1/n_b)
    if not variance: return {"z": 0., "p_two_sided": 1.}
    z = (success_a/n_a-success_b/n_b)/math.sqrt(variance)
    return {"z": z, "p_two_sided": math.erfc(abs(z)/math.sqrt(2))}
