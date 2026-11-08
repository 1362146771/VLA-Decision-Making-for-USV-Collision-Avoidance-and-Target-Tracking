from collections import deque
from pathlib import Path
import torch
from .checkpoint import restore, check_tokenizer
from .actions import action_id, ACTION_NAMES
from .data import image_sequence, policy_prompt

class LivePolicy:
    """Adapter for a future simulator integration. No Unity executable assumed.

    push_frame is called for every 10 Hz observation; decide at 2 Hz. Pass the
    LAST ACTUALLY EXECUTED action (after intervention), never the expert label
    for the decision being predicted. Reset explicitly between episodes.
    """
    def __init__(self, checkpoint, device="cpu", *, config=None, clip_config=None, tokenizer=None):
        from transformers import CLIPTokenizerFast
        self.model, self.metadata = restore(checkpoint, "policy", config=config, clip_config=clip_config)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.cfg = self.model.cfg
        self.tokenizer = CLIPTokenizerFast.from_pretrained(check_tokenizer(self.metadata, checkpoint, tokenizer), local_files_only=True)
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=self.cfg.observation_memory_frames*self.cfg.keyframe_stride+1)
        self.last_timestamp = None
        self.last_decision = None

    def push_frame(self, path, timestamp_s):
        if self.last_timestamp is not None and abs(timestamp_s-self.last_timestamp-self.cfg.frame_period_s) > .02*self.cfg.frame_period_s:
            raise ValueError("Irregular live frame timing; resample or reset explicitly")
        path = Path(path)
        if not path.is_file(): raise FileNotFoundError(path)
        self.frames.append(path)
        self.last_timestamp = timestamp_s

    @torch.no_grad()
    def decide(self, instruction, last_executed_action, *, task=None):
        if not self.frames: raise ValueError("No observation received")
        period = self.cfg.frame_period_s*self.cfg.keyframe_stride
        if self.last_decision is not None and self.last_timestamp-self.last_decision < period-1e-6:
            raise ValueError("Decision requested before the next control interval")
        action_id(last_executed_action)
        frames = list(self.frames)
        history = self.cfg.observation_memory_frames
        paths = [frames[max(0, len(frames)-1-j*self.cfg.keyframe_stride)] for j in range(history, -1, -1)]
        text = policy_prompt(instruction, self.cfg.prompt_mode, task)
        images = image_sequence(paths, self.cfg.image_size)[None].to(self.device)
        enc = self.tokenizer(text, padding="max_length", truncation=True,
                             max_length=self.cfg.max_text_length, return_tensors="pt")
        logits = self.model(images, enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device),
                            torch.tensor([last_executed_action], dtype=torch.long, device=self.device))
        probs = logits.softmax(-1)[0].cpu().tolist()
        selected = int(logits.argmax(-1))
        self.last_decision = self.last_timestamp
        return {"action_id": selected, "action_name": ACTION_NAMES[selected], "probabilities": probs}


@torch.no_grad()
def predict_dataset(args):
    import json
    from torch.utils.data import DataLoader
    from .data import EpisodeDataset, safe_path
    from .config import sha256_file
    from .dataset_index import resolve_dataset
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    manifest = resolve_dataset(args.dataset, args.split)
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    policy = LivePolicy(args.checkpoint, args.device, config=args.config,
                        clip_config=args.clip_config, tokenizer=args.tokenizer)
    predictions = []
    if doc["schema"] == "oceanvla.dataset.v1":
        ds = EpisodeDataset(args.data_root, manifest, policy.cfg, "policy", policy.tokenizer,
                            prediction_only=True)
        cursor = 0
        for batch in DataLoader(ds, batch_size=args.batch_size, shuffle=False):
            logits = policy.model(*(batch[k].to(policy.device) for k in
                                  ("images", "input_ids", "attention_mask", "last_action")))
            probs = logits.softmax(-1).cpu().tolist()
            top = logits.topk(3).indices.cpu().tolist()
            for row, values, ranking in zip(ds.rows[cursor:cursor+len(probs)], probs, top):
                predictions.append({"sample_id": row["sample_id"], "episode_id": row["episode_id"],
                                    "task": row["task"], "top3": ranking, "probabilities": values})
            cursor += len(probs)
    else:
        for episode in doc["episodes"]:
            source = Path(episode["source_attempt"])
            if not source.is_absolute():
                source = safe_path(args.data_root, episode["source_attempt"])
            for name, key in (("scenario.json", "scenario_sha256"), ("frames.jsonl", "frames_sha256")):
                if sha256_file(source/name) != episode[key]:
                    raise ValueError(key)
            scenario = json.loads((source/"scenario.json").read_text(encoding="utf-8"))
            frames = [json.loads(line) for line in (source/"frames.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(frames) != episode["frame_count"] or any(f["frame_index"] != i for i, f in enumerate(frames)):
                raise ValueError("frame_count")
            seen = set()
            for decision in episode["decisions"]:
                i = decision["frame_index"]
                start = i - policy.cfg.observation_memory_frames * policy.cfg.keyframe_stride
                if start < 0 or i >= len(frames) or decision["sample_id"] in seen:
                    raise ValueError("decisions")
                seen.add(decision["sample_id"])
                previous = frames[i-1].get("action_id") if i else scenario.get("initial_action_id")
                action_id(previous)
                selected = [frames[j] for j in range(start, i+1, policy.cfg.keyframe_stride)]
                stamps = [f["timestamp_s"] for f in selected]
                step = policy.cfg.frame_period_s * policy.cfg.keyframe_stride
                if any(abs(b-a-step) > .02*step for a, b in zip(stamps, stamps[1:])):
                    raise ValueError("timestamp_s")
                images = image_sequence([safe_path(source, f["image"]) for f in selected], policy.cfg.image_size)[None].to(policy.device)
                instruction = frames[i].get("instruction", scenario.get("instruction"))
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError("instruction")
                text = policy_prompt(instruction, policy.cfg.prompt_mode, episode["task"])
                enc = policy.tokenizer(text, padding="max_length", truncation=True,
                                       max_length=policy.cfg.max_text_length, return_tensors="pt")
                logits = policy.model(images, enc["input_ids"].to(policy.device), enc["attention_mask"].to(policy.device),
                                      torch.tensor([previous], dtype=torch.long, device=policy.device))
                predictions.append({"sample_id": decision["sample_id"], "episode_id": episode["episode_id"],
                                    "task": episode["task"], "top3": logits.topk(3).indices[0].cpu().tolist(),
                                    "probabilities": logits.softmax(-1)[0].cpu().tolist()})
    result = {"checkpoint_sha256": sha256_file(args.checkpoint), "manifest_sha256": sha256_file(manifest),
              "dataset_index_sha256": sha256_file(args.dataset), "split": args.split,
              "decision_support_sha256": doc["decision_support_sha256"], "samples": len(predictions),
              "predictions": predictions}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return {k: v for k, v in result.items() if k != "predictions"}
