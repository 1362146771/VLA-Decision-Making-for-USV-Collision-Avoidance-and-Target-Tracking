
from dataclasses import asdict
from pathlib import Path
import contextlib
import json
import math
import os
import random
import uuid
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import load_config, ModelConfig, sha256_file, canonical_hash
from .data import EpisodeDataset, EpochSampler, assert_disjoint
from .model import build_model, OceanVLA, focal_loss, parameter_counts
from .checkpoint import restore, read_checkpoint, save_checkpoint, source_identity, check_tokenizer, export_weights
from .metrics import classification_counts, binary_auc, summarize_confusion

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def move(batch, device): return {k:v.to(device) for k,v in batch.items()}

def stage_loss(model, stage, batch):
    if stage == "world": return model(batch["images"], batch["expert_actions"], batch["risk_labels"])
    if stage == "clip": return model(batch["images"], batch["input_ids"], batch["attention_mask"])
    logits = model(batch["images"], batch["input_ids"], batch["attention_mask"], batch["last_action"])
    return {"loss": focal_loss(logits, batch["targets"]), "logits": logits}

@torch.no_grad()
def validate(model, stage, loader, device, save_predictions=False):
    model.eval()
    total, loss_sum, c1, c3 = 0, 0., 0, 0
    sums, risks, labels, predictions = {}, [], [], []
    confusion = torch.zeros((6, 6), dtype=torch.int64)
    for raw in loader:
        batch = move(raw, device)
        out = stage_loss(model, stage, batch)
        n = len(batch["images"])
        if not torch.isfinite(out["loss"]): raise FloatingPointError("Non-finite validation loss")
        total += n; loss_sum += float(out["loss"])*n
        if stage == "world":
            for key in ("mse_1", "mse_k", "loss_rep", "loss_dyn", "loss_risk"):
                sums[key] = sums.get(key, 0.) + float(out[key])*n
            risks.extend(out["risk_logits"].cpu().tolist()); labels.extend(batch["risk_labels"].cpu().tolist())
        if stage == "policy":
            flat = (batch["targets"] * 6 + out["logits"].argmax(-1)).cpu()
            confusion += torch.bincount(flat, minlength=36).reshape(6, 6)
            counts = classification_counts(out["logits"], batch["targets"])
            c1 += counts["top1_correct"]; c3 += counts["top3_correct"]
            if save_predictions:
                for target, top in zip(batch["targets"].cpu().tolist(), out["logits"].topk(3).indices.cpu().tolist()):
                    predictions.append({"target": target, "top3": top})
    if not total: raise ValueError("Empty validation/evaluation")
    result = {"samples": total, "loss": loss_sum/total, **{k:v/total for k,v in sums.items()}}
    if stage == "world": result["risk_auroc"] = binary_auc(risks, labels)
    if stage == "policy":
        result.update(top1=c1/total, top3=c3/total, top1_correct=c1, top3_correct=c3)
        result.update(summarize_confusion(confusion.tolist()))
    return result, predictions

def train(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1: raise ValueError("This verified runner is single-device. Do not use torchrun.")
    cfg, training = load_config(args.config)
    stage = args.stage
    options = training[stage]
    if stage == "clip" and options["micro_batch"] != options["global_batch"]:
        raise ValueError("CLIP negatives require global_batch in one forward; accumulation does not reproduce contrastive batch")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()): raise FileExistsError("Use a new output directory, including for resume")
    seed_all(training["seed"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA is unavailable")
    parents, resume = {}, None
    if args.resume:
        model, resume = restore(args.resume, stage)
        if "training_state" not in resume:
            raise ValueError("Resume requires an epoch checkpoint with optimizer/RNG state; best.pt is for inference")
        if resume["model_config"] != asdict(cfg) or resume["metadata"]["training_config"] != training:
            raise ValueError("Resume config differs; start an explicitly new experiment instead")
        if resume["metadata"]["source"]["source_sha256"] != source_identity()["source_sha256"]:
            raise ValueError("Exact resume requires identical source code")
        parents = {"resume": sha256_file(args.resume)}
    elif stage == "policy":
        if not args.world_checkpoint or not args.clip_checkpoint: raise ValueError("Policy stage requires trained world and CLIP checkpoints")
        world, wc = restore(args.world_checkpoint, "world")
        encoder, cc = restore(args.clip_checkpoint, "clip")
        if wc["model_config"] != asdict(cfg) or cc["model_config"] != asdict(cfg): raise ValueError("Parent model configs differ")
        model = OceanVLA(cfg, world=world, encoder=encoder)
        parents = {"world": sha256_file(args.world_checkpoint), "clip": sha256_file(args.clip_checkpoint)}
    else: model = build_model(stage, cfg)
    tokenizer = None
    if stage != "world":
        from transformers import CLIPTokenizerFast
        tok_source = check_tokenizer(resume, args.resume) if resume else (check_tokenizer(cc, args.clip_checkpoint) if stage == "policy" else cfg.clip_id)
        tokenizer = CLIPTokenizerFast.from_pretrained(tok_source)
    ds = EpisodeDataset(args.data_root, args.train_manifest, cfg, stage, tokenizer, augment=True)
    val = EpisodeDataset(args.data_root, args.val_manifest, cfg, stage, tokenizer, augment=False)
    assert_disjoint(ds, val)
    if stage == "policy" and not resume:
        if wc["metadata"]["dataset_hash"] != ds.doc["dataset_hash"]:
            raise ValueError("World and policy stages must use the same episode split/preparation")
        if set(cc["metadata"].get("training_episode_ids", [])) & set(val.doc["episodes"]):
            raise ValueError("CLIP training episodes overlap policy validation")
        if set(cc["metadata"].get("training_image_hashes", [])) & set(val.doc["assets"].values()):
            raise ValueError("CLIP training images overlap policy validation")
    data_identity = {"train": sha256_file(args.train_manifest), "val": sha256_file(args.val_manifest)}
    if resume and resume["metadata"]["data_manifests"] != data_identity: raise ValueError("Resume data changed")
    if len(ds) < options["global_batch"]: raise ValueError("Training split must contain at least one global batch")
    sampler = EpochSampler(ds, training["seed"], options["balanced"], options["global_batch"])
    # num_workers=0 keeps epoch-boundary RNG resume exact on Windows and Linux.
    loader = DataLoader(ds, batch_size=options["micro_batch"], sampler=sampler, num_workers=0, drop_last=False)
    vloader = DataLoader(val, batch_size=options["micro_batch"], shuffle=False, num_workers=0, drop_last=False)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim_cls = torch.optim.Adam if stage == "world" else torch.optim.AdamW
    optimizer = optim_cls(params, lr=options["lr"], weight_decay=options["weight_decay"])
    accum = options["global_batch"]//options["micro_batch"]
    total_steps = math.ceil(len(loader)/accum)*options["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: .5*(1+math.cos(math.pi*min(s,total_steps)/total_steps)))
    use_amp = bool(options["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start, step, best = 0, 0, float("inf")
    if resume:
        state = resume["training_state"]
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        start, step = state["epoch_completed"], state["global_step"]
        best = state["best_val_loss"]
        if resume["metadata"]["validation"]["loss"] > best:
            raise ValueError("Resume from the best epoch checkpoint so the previous best weights remain available")
        if start >= options["epochs"]: raise ValueError("Checkpoint already completed the configured run")
        torch.set_rng_state(state["torch_rng"])
        if use_amp and state["cuda_rng"]: torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        nr = state["numpy_rng"]; np.random.set_state((nr[0], np.array(nr[1], dtype=np.uint32), nr[2], nr[3], nr[4]))
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_files = {}
    if tokenizer:
        tokenizer.save_pretrained(output/"tokenizer")
        tokenizer_files = {p.name:sha256_file(p) for p in (output/"tokenizer").iterdir() if p.is_file()}
    metadata = {"run_id": str(uuid.uuid4()), "training_seed_design": "single fixed training seed",
                "seed": training["seed"], "training_config": training, "source": source_identity(),
                "dataset_hash": ds.doc["dataset_hash"], "data_manifests": data_identity,
                "training_episode_ids": ds.doc["episodes"], "validation_episode_ids": val.doc["episodes"],
                "parents": parents, "tokenizer_files": tokenizer_files, "scope": "simulation only"}
    metadata["training_image_hashes"] = sorted(set(ds.doc["assets"].values()))
    metadata["pretraining_episode_ids"] = (resume["metadata"].get("pretraining_episode_ids", []) if resume else
        cc["metadata"]["training_episode_ids"] if stage == "policy" else [])
    metadata["pretraining_image_hashes"] = (resume["metadata"].get("pretraining_image_hashes", []) if resume else
        cc["metadata"].get("training_image_hashes", []) if stage == "policy" else [])
    (output/"run.json").write_text(json.dumps({**metadata, "model_config": asdict(cfg), "parameter_count": parameter_counts(model)}, indent=2), encoding="utf-8")
    generated = []
    best_filename = None
    # Resume into a NEW directory: retain the source checkpoint by reference if
    # no new epoch beats it, rather than pretending a worse new model is best.
    if resume:
        export_weights(output/"best.pt", model, stage, resume["metadata"])
        (output/"best_selection.json").write_text(json.dumps({"external_resume_checkpoint": str(Path(args.resume).resolve()),
            "sha256": sha256_file(args.resume), "best_val_loss": best}, indent=2), encoding="utf-8")
    for epoch in range(start, options["epochs"]):
        sampler.epoch = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sum_loss, seen, group_count = 0., 0, 0
        for bi, raw in enumerate(loader):
            batch = move(raw, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                result = stage_loss(model, stage, batch)
                n = len(batch["images"])
                loss = result["loss"]
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite training loss")
            # Accumulate sums then divide gradients by ACTUAL examples, including tail.
            scaler.scale(loss*n).backward()
            group_count += n; seen += n; sum_loss += float(loss.detach())*n
            if (bi+1) % accum == 0 or bi+1 == len(loader):
                scaler.unscale_(optimizer)
                if not any(p.grad is not None for p in params): raise RuntimeError("No gradients: broken stage")
                for p in params:
                    if p.grad is not None: p.grad.div_(group_count)
                # GradScaler records overflow during unscale and skips the update;
                # allow it to reduce the scale instead of aborting before scaler.step.
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=not use_amp)
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale: scheduler.step(); step += 1
                optimizer.zero_grad(set_to_none=True); group_count = 0
                if stage == "clip":
                    with torch.no_grad(): model.clip.logit_scale.clamp_(max=math.log(100))
        metrics, _ = validate(model, stage, vloader, device)
        best = min(best, metrics["loss"])
        record = {"epoch_completed": epoch+1, "global_step": step, "train_loss": sum_loss/seen,
                  "validation": metrics, "best_val_loss": best}
        with (output/"metrics.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(record)+"\n")
        nr = np.random.get_state()
        state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch_completed": epoch+1, "global_step": step, "best_val_loss": best,
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if use_amp else [],
                 "python_rng": random.getstate(), "numpy_rng": [nr[0], nr[1].tolist(), nr[2], nr[3], nr[4]]}
        filename = f"epoch_{epoch+1:03d}.pt"
        digest = save_checkpoint(output/filename, model, stage, {**metadata, "validation": metrics}, training_state=state)
        # Pointer only, never duplicate or overwrite an old weight file.
        if metrics["loss"] <= best:
            best_filename = filename
            export_weights(output/"best.pt", model, stage,
                           {**metadata, "validation": metrics, "epoch_completed": epoch+1,
                            "global_step": step, "training_checkpoint_sha256": digest}, replace=True)
            (output/"best_selection.json").write_text(json.dumps({"checkpoint": filename, "sha256": digest, "validation": metrics}, indent=2), encoding="utf-8")
        generated.append(output/filename)
        # Keep best and latest only, within this new run directory. Never touch
        # input checkpoints, other runs, or any path not created in this call.
        for old in list(generated):
            if old.name in (filename, best_filename): continue
            if old.resolve().parent != output.resolve(): raise ValueError("Checkpoint retention path escaped output")
            old.unlink()
            old.with_suffix(old.suffix+".sha256").unlink()
            generated.remove(old)
        (output/"latest.json").write_text(json.dumps({"checkpoint":filename,"sha256":digest},indent=2),encoding="utf-8")
        print(json.dumps(record), flush=True)
    return output

def evaluate(args):
    model, ck = restore(args.checkpoint, "policy")
    cfg = model.cfg
    from transformers import CLIPTokenizerFast
    tokenizer = CLIPTokenizerFast.from_pretrained(check_tokenizer(ck, args.checkpoint), local_files_only=True)
    ds = EpisodeDataset(args.data_root, args.manifest, cfg, "policy", tokenizer)
    if ds.doc["split"] != "test": raise ValueError("Formal evaluation requires the test manifest")
    training_ids = set(ck["metadata"]["training_episode_ids"]) | set(ck["metadata"].get("pretraining_episode_ids", []))
    if training_ids & set(ds.doc["episodes"]): raise ValueError("Test/train overlap")
    trained_images = set(ck["metadata"].get("pretraining_image_hashes", [])) | set(ck["metadata"].get("training_image_hashes", []))
    if trained_images & set(ds.doc["assets"].values()):
        raise ValueError("Training/pretraining images overlap test data")
    if ds.doc["dataset_hash"] != ck["metadata"]["dataset_hash"]: raise ValueError("Test manifest belongs to a different dataset")
    output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    device = torch.device(args.device); model.to(device)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    result, predictions = validate(model, "policy", loader, device, True)
    for r, pred in zip(ds.rows, predictions): pred.update(sample_id=r["sample_id"], task=r["task"], episode_id=r["episode_id"])
    by_task = {}
    for task in ("avoidance", "tracking"):
        rows = [p for p in predictions if p["task"] == task]
        if rows: by_task[task] = {"samples": len(rows), "top1": sum(p["target"] == p["top3"][0] for p in rows)/len(rows),
                                 "top3": sum(p["target"] in p["top3"] for p in rows)/len(rows)}
    result.update(by_task=by_task, checkpoint_sha256=sha256_file(args.checkpoint),
                  manifest_sha256=sha256_file(args.manifest), run_id=ck["metadata"]["run_id"],
                  evaluation_source=source_identity(), predictions=predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k:v for k,v in result.items() if k != "predictions"}
