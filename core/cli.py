import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("export", help="Export full tensor-only weights with a matching JSON sidecar")
    q.add_argument("--checkpoint", required=True); q.add_argument("--output", required=True)
    q = sub.add_parser("prepare", help="Validate raw episodes and create immutable splits")
    q.add_argument("--config", required=True); q.add_argument("--data-root", required=True)
    q.add_argument("--episodes", required=True); q.add_argument("--output", required=True)
    q.add_argument("--split-seed", type=int, default=42)
    q = sub.add_parser("prepare-clip", help="Prepare a separate licensed caption corpus")
    q.add_argument("--config", required=True); q.add_argument("--data-root", required=True)
    q.add_argument("--pairs", required=True); q.add_argument("--output", required=True)
    q.add_argument("--split-seed", type=int, default=42)
    q = sub.add_parser("train")
    q.add_argument("--config", required=True); q.add_argument("--stage", choices=["world","clip","policy"], required=True)
    q.add_argument("--data-root", required=True); q.add_argument("--train-manifest", required=True)
    q.add_argument("--val-manifest", required=True); q.add_argument("--output", required=True)
    q.add_argument("--device", default="cuda"); q.add_argument("--resume")
    q.add_argument("--world-checkpoint"); q.add_argument("--clip-checkpoint")
    q = sub.add_parser("evaluate")
    q.add_argument("--checkpoint", required=True); q.add_argument("--data-root", required=True)
    q.add_argument("--manifest", required=True); q.add_argument("--output", required=True)
    q.add_argument("--device", default="cuda"); q.add_argument("--batch-size", type=int, default=4)
    q = sub.add_parser("inspect", help="Safe metadata plus strict architecture/state_dict load")
    q.add_argument("--checkpoint", required=True)
    q = sub.add_parser("predict", help="Offline input file; no expert targets")
    q.add_argument("--checkpoint", required=True); q.add_argument("--request", required=True)
    q.add_argument("--device", default="cpu")
    q = sub.add_parser("rollouts", help="Aggregate exported simulator telemetry")
    q.add_argument("--input", required=True); q.add_argument("--output", required=True)
    q.add_argument("--expected-trials", type=int, default=50)
    args = p.parse_args()
    if args.command == "export":
        from .checkpoint import export_checkpoint
        result = {"weights_sha256": export_checkpoint(args.checkpoint, args.output), "output": args.output}
    elif args.command == "prepare":
        from .config import load_config
        from .data import prepare
        cfg, _ = load_config(args.config)
        result = prepare(args.data_root,args.episodes,args.output,cfg,args.split_seed)
    elif args.command == "prepare-clip":
        from .config import load_config
        from .data import prepare_clip
        cfg, _ = load_config(args.config)
        result = prepare_clip(args.data_root,args.pairs,args.output,cfg,args.split_seed)
    elif args.command == "train":
        from .engine import train
        result = {"output":str(train(args))}
    elif args.command == "evaluate":
        from .engine import evaluate
        result = evaluate(args)
    elif args.command == "inspect":
        from .checkpoint import restore
        from .model import parameter_counts
        from .config import sha256_file
        model, ck = restore(args.checkpoint)
        result = {"stage":ck["stage"],"config":ck["model_config"],"parameters":parameter_counts(model),
                  "sha256":sha256_file(args.checkpoint),"metadata":ck["metadata"],
                  "missing_keys":0,"unexpected_keys":0,"shape_mismatch":0}
    elif args.command == "predict":
        from .inference import LivePolicy
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        policy = LivePolicy(args.checkpoint,args.device)
        for frame in request["frames"]:
            policy.push_frame(Path(args.request).parent/frame["image"],frame["timestamp_s"])
        result = policy.decide(request["instruction"],request["last_executed_action"])
    else:
        from .rollouts import summarize_rollouts
        result = summarize_rollouts(args.input,args.output,args.expected_trials)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
