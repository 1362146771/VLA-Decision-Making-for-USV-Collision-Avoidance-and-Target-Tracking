
import math
import statistics
from collections import defaultdict
from pathlib import Path
import json
from .metrics import wilson
from .config import sha256_file

def summarize_rollouts(path, output, expected_trials=50):
    trials = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    groups, seen = defaultdict(list), set()
    for t in trials:
        if t["trial_id"] in seen: raise ValueError("Duplicate trial_id")
        seen.add(t["trial_id"])
        sha = t["checkpoint_sha256"]
        if len(sha) != 64 or any(x not in "0123456789abcdef" for x in sha): raise ValueError("Checkpoint SHA256 required")
        if t["mode"] not in ("policy_only", "radar_assisted"): raise ValueError("Unknown safety mode")
        if t["task"] not in ("avoidance", "tracking", "colregs"): raise ValueError("Unknown task")
        if not isinstance(t["seed"], int): raise ValueError("Evaluation seed required")
        if not isinstance(t["completed"], bool): raise ValueError("completed flag required")
        if t["mode"] == "policy_only" and t.get("radar_override_count", 0): raise ValueError("Radar intervention in policy-only trial")
        if t["mode"] == "radar_assisted" and t.get("radar_threshold_m") != 30.0: raise ValueError("This protocol uses 30m proactive radar")
        trace = t["trace"]
        if not trace: raise ValueError("Empty telemetry trace")
        prior = None
        for x in trace:
            for key in ("timestamp_s", "obstacle_distance_m", "pursuit_progress_m"):
                if not isinstance(x[key], (int,float)) or not math.isfinite(x[key]): raise ValueError("Invalid telemetry scalar")
            if x["obstacle_distance_m"] < 0 or x["pursuit_progress_m"] < 0: raise ValueError("Negative geometry")
            if prior is not None and x["timestamp_s"] <= prior: raise ValueError("Non-monotonic trace")
            prior = x["timestamp_s"]
            for key in ("collision", "inside_navigable_area", "target_visible"):
                if not isinstance(x[key], bool): raise ValueError(f"Missing boolean {key}")
        collision = any(x["collision"] for x in trace)
        cpa = 0.0 if collision else min(x["obstacle_distance_m"] for x in trace)
        inside = all(x["inside_navigable_area"] for x in trace)
        if t["task"] == "tracking":
            for x in trace:
                if not isinstance(x["target_distance_m"], (float,int)) or not math.isfinite(x["target_distance_m"]): raise ValueError("Invalid target distance")
            success = t["completed"] and not collision and inside and trace[-1]["pursuit_progress_m"] >= 200 and all(
                x["target_visible"] and 20 <= x["target_distance_m"] <= 40 for x in trace)
        else: success = t["completed"] and not collision and inside and cpa >= 20
        if t["task"] == "avoidance" and t["initial_distance_m"] not in (100, 300): raise ValueError("Unknown avoidance distance")
        if t["task"] == "colregs":
            if t["scenario"] not in ("head_on", "crossing", "overtaking"): raise ValueError("Unknown encounter")
            if not isinstance(t["give_way_compliant"], bool): raise ValueError("Simulator rule evaluator must supply compliance flag")
            if not isinstance(t["rule_violations"], int) or t["rule_violations"] < 0: raise ValueError("Invalid violation count")
        key = (sha, t["mode"], t["task"], t["scenario"] if t["task"] == "colregs" else str(t.get("initial_distance_m", "200m")))
        groups[key].append({**t, "success": success, "collision_result": collision, "cpa": cpa})
    result = []
    for (sha, mode, task, condition), items in sorted(groups.items()):
        n = len(items)
        if n != expected_trials: raise ValueError(f"Group {(mode,task,condition)} has {n}, expected {expected_trials}")
        if len({x["seed"] for x in items}) != n: raise ValueError("Independent trials require distinct environment seeds")
        successes = sum(x["success"] for x in items)
        collisions = sum(x["collision_result"] for x in items)
        nc = [x["cpa"] for x in items if not x["collision_result"]]
        near = sum(not x["collision_result"] and 0 < x["cpa"] < 20 for x in items)
        r = {"checkpoint_sha256": sha, "mode": mode, "task": task, "condition": condition,
             "trials": n, "successes": successes, "sr": successes/n, "sr_ci95": wilson(successes,n),
             "collision_free": (n-collisions)/n, "collision_free_ci95": wilson(n-collisions,n),
             "collision_rate": collisions/n, "collision_ci95": wilson(collisions,n),
             "near_miss": near/n, "near_miss_ci95": wilson(near,n),
             "cpa_noncollision_count": len(nc), "cpa_mean": statistics.mean(nc) if nc else None,
             "cpa_min": min(nc) if nc else None, "cpa_std_population": statistics.pstdev(nc) if nc else None}
        if task == "colregs":
            compliant = sum(x["give_way_compliant"] for x in items)
            r.update(give_way_compliance=compliant/n, give_way_ci95=wilson(compliant,n),
                     avg_rule_violations=sum(x["rule_violations"] for x in items)/n)
        result.append(r)
    if not result: raise ValueError("No trials")
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema":"oceanvla.rollout_metrics.v1", "source_sha256":sha256_file(path),
               "scope":"simulation-only; conditional on supplied telemetry and fixed checkpoints", "groups":result}
    output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return payload
