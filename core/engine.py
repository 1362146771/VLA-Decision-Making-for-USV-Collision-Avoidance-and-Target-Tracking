
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
from torch.nn.parallel import DistributedDataParallel
from .distributed import DistributedContext, ShardedEpochSampler
from .config import load_config, ModelConfig, sha256_file, canonical_hash
from .data import EpisodeDataset, EpochSampler, assert_disjoint
from .model import build_model, MaritimeCLIP, OceanVLA, focal_loss, parameter_counts
from .checkpoint import restore, read_checkpoint, save_checkpoint, source_identity, check_tokenizer, export_weights
from .metrics import episode_accuracy, binary_auc, summarize_confusion

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def move(batch, device): return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

def stage_loss(model, stage, batch):
    if stage == "world": return model(batch["images"], batch["expert_actions"], batch["risk_labels"])
    if stage == "clip":
        return model(batch["images"], batch["input_ids"], batch["attention_mask"],
                     distributed=isinstance(model, DistributedDataParallel))
    logits = model(batch["images"], batch["input_ids"], batch["attention_mask"], batch["last_action"])
    base = model.module if isinstance(model, DistributedDataParallel) else model
    loss_fn = focal_loss if base.cfg.policy_loss == "focal" else torch.nn.functional.cross_entropy
    return {"loss": loss_fn(logits, batch["targets"]), "logits": logits}

@torch.no_grad()
def validate(model, stage, loader, device, save_predictions=False):
    model.eval()
    total, loss_sum = 0, 0.
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
            for eid, task, target, top in zip(batch["episode_id"], batch["task"], batch["targets"].cpu().tolist(),
                                              out["logits"].topk(3).indices.cpu().tolist()):
                predictions.append({"episode_id": eid, "task": task, "target": target, "top3": top})
    if not total: raise ValueError("Empty validation/evaluation")
    result = {"samples": total, "loss": loss_sum/total, **{k:v/total for k,v in sums.items()}}
    if stage == "world": result["risk_auroc"] = binary_auc(risks, labels)
    if stage == "policy":
        result.update(episode_accuracy(predictions))
        result.update(summarize_confusion(confusion.tolist()))
    return result, predictions if save_predictions else []

def train(args):
    context = DistributedContext(args.device)
    try:
        return _train(args, context)
    finally:
        context.close()


def _train(args, context):
    stage = args.stage
    variant = getattr(args, "variant", None)
    if variant is not None and stage != "policy":
        raise ValueError("Ablation variants apply to policy training only")
    cfg, training = load_config(args.config, variant)
    options = training[stage]
    patience = options.get("patience") if stage == "policy" else None
    if options["global_batch"] % (options["micro_batch"] * context.size):
        raise ValueError("global_batch must be divisible by micro_batch * world_size")
    if stage == "clip" and options["micro_batch"] * context.size != options["global_batch"]:
        raise ValueError("CLIP negatives require global_batch in one forward; accumulation does not reproduce contrastive batch")
    output = Path(args.output)
    def check_output():
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Use a new output directory, including for resume")
    context.primary_call(check_output)
    seed_all(training["seed"])
    device = context.device
    if device.type == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA is unavailable")
    parents, resume, wc, cc = {}, None, None, None
    if args.resume:
        model, resume = restore(args.resume, stage)
        if "training_state" not in resume:
            raise ValueError("Resume requires an epoch checkpoint with optimizer/RNG state; best.pt is for inference")
        if resume["model_config"] != asdict(cfg) or resume["metadata"]["training_config"] != training:
            raise ValueError("Resume config differs; start an explicitly new experiment instead")
        if resume["metadata"]["source"]["source_sha256"] != source_identity()["source_sha256"]:
            raise ValueError("Exact resume requires identical source code")
        protocol = resume["metadata"].get("training_protocol", {})
        if protocol.get("device_count", 1) != context.size:
            raise ValueError("Resume world_size differs")
        if context.size > 1 and len(resume["training_state"].get("rng_by_rank", [])) != context.size:
            raise ValueError("Resume requires per-rank RNG state")
        parents = {"resume": sha256_file(args.resume)}
    elif stage == "policy":
        if bool(args.world_checkpoint) != cfg.use_world or bool(args.clip_checkpoint) != (cfg.encoder_kind == "maritime"):
            raise ValueError("Supply exactly the world and maritime CLIP parents enabled by the variant")
        world, encoder = None, None
        if cfg.use_world:
            world, wc = restore(args.world_checkpoint, "world")
            fields = ("world_arch", "world_refinement_width", "world_dim", "action_embed_dim",
                      "prediction_steps", "frame_period_s", "keyframe_stride", "image_size", "normalization")
            if any(wc["model_config"][k] != getattr(cfg, k) for k in fields):
                raise ValueError("World parent model config differs")
            parents["world"] = sha256_file(args.world_checkpoint)
        if cfg.encoder_kind == "maritime":
            encoder, cc = restore(args.clip_checkpoint, "clip")
            fields = ("image_size", "max_text_length", "normalization", "unfreeze_vision_layers")
            if any(cc["model_config"][k] != getattr(cfg, k) for k in fields):
                raise ValueError("CLIP parent model config differs")
            parents["clip"] = sha256_file(args.clip_checkpoint)
        elif variant is not None and cfg.encoder_kind == "generic":
            encoder = MaritimeCLIP(cfg)
        if variant is not None:
            seed_all(training["seed"])
        model = OceanVLA(cfg, world=world, encoder=encoder)
    else: model = build_model(stage, cfg)
    tokenizer = None
    if stage != "world":
        from transformers import CLIPTokenizerFast
        tok_source = check_tokenizer(resume, args.resume) if resume else (
            check_tokenizer(cc, args.clip_checkpoint) if cc else cfg.clip_id)
        tokenizer = CLIPTokenizerFast.from_pretrained(tok_source)
    ds = EpisodeDataset(args.data_root, args.train_manifest, cfg, stage, tokenizer, augment=True)
    val = EpisodeDataset(args.data_root, args.val_manifest, cfg, stage, tokenizer, augment=False)
    assert_disjoint(ds, val)
    if stage == "policy" and not resume:
        if wc and wc["metadata"]["dataset_hash"] != ds.doc["dataset_hash"]:
            raise ValueError("World and policy stages must use the same episode split/preparation")
        if cc and set(cc["metadata"].get("training_episode_ids", [])) & set(val.doc["episodes"]):
            raise ValueError("CLIP training episodes overlap policy validation")
        if cc and set(cc["metadata"].get("training_image_hashes", [])) & set(val.doc["assets"].values()):
            raise ValueError("CLIP training images overlap policy validation")
    data_identity = {"train": sha256_file(args.train_manifest), "val": sha256_file(args.val_manifest)}
    if resume and resume["metadata"]["data_manifests"] != data_identity: raise ValueError("Resume data changed")
    if len(ds) < options["global_batch"]: raise ValueError("Training split must contain at least one global batch")
    sampler = EpochSampler(ds, training["seed"], options["balanced"], options["global_batch"])
    if context.size > 1:
        sampler = ShardedEpochSampler(sampler, context.rank, context.size)
    loader = DataLoader(ds, batch_size=options["micro_batch"], sampler=sampler, num_workers=0, drop_last=False)
    vloader = DataLoader(val, batch_size=options["global_batch"] if stage == "clip" else options["micro_batch"], shuffle=False, num_workers=0, drop_last=False)
    model.to(device)
    runner = (DistributedDataParallel(model,
              device_ids=[device.index] if device.type == "cuda" else None,
              broadcast_buffers=False, find_unused_parameters=stage == "policy")
              if context.size > 1 else model)
    if context.size > 1:
        seed_all(training["seed"] + context.rank)
    params = [p for p in model.parameters() if p.requires_grad]
    optim_cls = torch.optim.Adam if stage == "world" else torch.optim.AdamW
    optimizer = optim_cls(params, lr=options["lr"], weight_decay=options["weight_decay"])
    accum = options["global_batch"]//(options["micro_batch"] * context.size)
    total_steps = math.ceil(len(loader)/accum)*options["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: .5*(1+math.cos(math.pi*min(s,total_steps)/total_steps)))
    use_amp = bool(options["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start, step, best, bad_epochs = 0, 0, float("inf"), 0
    resume_best = None
    if resume:
        state = resume["training_state"]
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        start, step = state["epoch_completed"], state["global_step"]
        best = state["best_val_loss"]
        if patience is not None and "bad_epochs" not in state:
            raise ValueError("bad_epochs")
        bad_epochs = state.get("bad_epochs", 0)
        if patience is not None and bad_epochs >= patience:
            raise ValueError("early_stopping")
        if bad_epochs > 0 or resume["metadata"]["validation"]["loss"] > best:
            resume_best = restore(Path(args.resume).parent / "best.pt", stage)
            best_model, best_record = resume_best
            if (best_record["model_config"] != resume["model_config"]
                    or best_record["metadata"]["run_id"] != resume["metadata"]["run_id"]
                    or best_record["metadata"]["validation"]["loss"] != best):
                raise ValueError("best_checkpoint")
        if start >= options["epochs"]: raise ValueError("Checkpoint already completed the configured run")
        rng = state["rng_by_rank"][context.rank] if context.size > 1 else state
        torch.set_rng_state(rng["torch_rng"])
        if device.type == "cuda" and rng["cuda_rng"]:
            if context.size > 1:
                torch.cuda.set_rng_state(rng["cuda_rng"][0], device)
            else:
                torch.cuda.set_rng_state_all(rng["cuda_rng"])
        random.setstate(rng["python_rng"])
        nr = rng["numpy_rng"]; np.random.set_state((nr[0], np.array(nr[1], dtype=np.uint32), nr[2], nr[3], nr[4]))
    def initialize_output():
        output.mkdir(parents=True, exist_ok=True)
        tokenizer_files = {}
        if tokenizer:
            tokenizer.save_pretrained(output/"tokenizer")
            tokenizer_files = {p.name:sha256_file(p) for p in (output/"tokenizer").iterdir() if p.is_file()}
        return tokenizer_files
    tokenizer_files = context.primary_call(initialize_output)
    metadata = {"run_id": str(uuid.uuid4()), "training_seed_design": "single fixed training seed",
                 "training_protocol": {"device_count": context.size,
                                       "backend": "nccl" if context.size > 1 and device.type == "cuda" else "gloo" if context.size > 1 else None,
                                       "micro_batch_per_rank": options["micro_batch"],
                                       "global_batch": options["global_batch"],
                                       "gradient_accumulation_steps": accum,
                                       "train_samples_per_epoch": len(sampler) * context.size,
                                       "dropped_tail_samples": len(ds) - len(sampler) * context.size,
                                       "termination": "early_stopping" if patience else "fixed_epochs",
                                       "patience": patience,
                                      "checkpoint_selection": "minimum_validation_loss"},
                "seed": training["seed"], "training_config": training, "source": source_identity(),
                "dataset_hash": ds.doc["dataset_hash"], "data_manifests": data_identity,
                "training_episode_ids": ds.doc["episodes"], "validation_episode_ids": val.doc["episodes"],
                "parents": parents, "tokenizer_files": tokenizer_files, "scope": "simulation only"}
    metadata["training_image_hashes"] = sorted(set(ds.doc["assets"].values()))
    metadata["pretraining_episode_ids"] = (resume["metadata"].get("pretraining_episode_ids", []) if resume else
        cc["metadata"]["training_episode_ids"] if cc else [])
    metadata["pretraining_image_hashes"] = (resume["metadata"].get("pretraining_image_hashes", []) if resume else
        cc["metadata"].get("training_image_hashes", []) if cc else [])
    metadata = context.primary_call(lambda: metadata)
    def write_run():
        (output/"run.json").write_text(json.dumps({**metadata, "model_config": asdict(cfg), "parameter_count": parameter_counts(model)}, indent=2), encoding="utf-8")
    context.primary_call(write_run)
    generated = []
    best_filename = None
    def write_resume_selection():
        if resume:
            best_model, best_record = resume_best or (model, resume)
            export_weights(output/"best.pt", best_model, stage, {**best_record["metadata"], "run_id": metadata["run_id"]})
            best_source = Path(args.resume).parent / "best.pt" if resume_best else Path(args.resume)
            (output/"best_selection.json").write_text(json.dumps({"external_resume_checkpoint": str(best_source.resolve()),
                "sha256": sha256_file(best_source), "best_val_loss": best}, indent=2), encoding="utf-8")
    context.primary_call(write_resume_selection)
    for epoch in range(start, options["epochs"]):
        sampler.epoch = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sum_loss, seen, group_count = 0., 0, 0
        for bi, raw in enumerate(loader):
            batch = move(raw, device)
            synchronize = (bi+1) % accum == 0 or bi+1 == len(loader)
            sync_context = (runner.no_sync() if context.size > 1 and not synchronize
                            else contextlib.nullcontext())
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    result = stage_loss(runner, stage, batch)
                    n = len(batch["images"])
                    loss = result["loss"]
                context.check_finite(loss)
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
        sum_loss, seen = context.totals(sum_loss, seen)
        metrics, _ = context.primary_call(lambda: validate(model, stage, vloader, device))
        improved = metrics["loss"] < best
        if improved:
            best, bad_epochs = metrics["loss"], 0
        else:
            bad_epochs += 1
        stop_reason = ("early_stopping" if patience is not None and bad_epochs >= patience
                       else "max_epochs" if epoch + 1 == options["epochs"] else None)
        record = {"epoch_completed": epoch+1, "global_step": step, "train_loss": sum_loss/seen,
                  "validation": metrics, "best_val_loss": best, "bad_epochs": bad_epochs,
                  "stop_reason": stop_reason}
        nr = np.random.get_state()
        rng = {"torch_rng": torch.get_rng_state(),
               "cuda_rng": ([torch.cuda.get_rng_state(device)] if context.size > 1 else torch.cuda.get_rng_state_all()) if device.type == "cuda" else [],
               "python_rng": random.getstate(),
               "numpy_rng": [nr[0], nr[1].tolist(), nr[2], nr[3], nr[4]]}
        rng_by_rank = context.gather_objects(rng)
        state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch_completed": epoch+1, "global_step": step, "best_val_loss": best,
                 "bad_epochs": bad_epochs, "stop_reason": stop_reason,
                 **rng, "rng_by_rank": rng_by_rank}
        def write_epoch():
            nonlocal best_filename
            with (output/"metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record)+"\n")
            filename = f"epoch_{epoch+1:03d}.pt"
            digest = save_checkpoint(output/filename, model, stage, {**metadata, "validation": metrics}, training_state=state)
            if improved:
                best_filename = filename
                export_weights(output/"best.pt", model, stage,
                               {**metadata, "validation": metrics, "epoch_completed": epoch+1,
                                "global_step": step, "training_checkpoint_sha256": digest}, replace=True)
                (output/"best_selection.json").write_text(json.dumps({"checkpoint": filename, "sha256": digest, "validation": metrics}, indent=2), encoding="utf-8")
            generated.append(output/filename)
            for old in list(generated):
                if old.name in (filename, best_filename): continue
                if old.resolve().parent != output.resolve(): raise ValueError("Checkpoint retention path escaped output")
                old.unlink()
                old.with_suffix(old.suffix+".sha256").unlink()
                generated.remove(old)
            (output/"latest.json").write_text(json.dumps({"checkpoint":filename,"sha256":digest},indent=2),encoding="utf-8")
            print(json.dumps(record), flush=True)
        context.primary_call(write_epoch)
        if stop_reason == "early_stopping":
            break
    return output

def evaluate(args):
    from .evaluation import decision_support, annotate_support
    from .dataset_index import resolve_dataset
    manifest = resolve_dataset(args.dataset) if getattr(args, "dataset", None) else Path(args.manifest)
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    if doc.get("schema") != "oceanvla.dataset.v1":
        raise ValueError("recorded_action_labels")
    model, ck = restore(args.checkpoint, "policy", config=getattr(args, "config", None),
                        clip_config=getattr(args, "clip_config", None),
                        provenance=getattr(args, "provenance", None))
    cfg = model.cfg
    from transformers import CLIPTokenizerFast
    tokenizer = CLIPTokenizerFast.from_pretrained(
        check_tokenizer(ck, args.checkpoint, getattr(args, "tokenizer", None)), local_files_only=True)
    if not all(key in ck["metadata"] for key in ("training_episode_ids", "dataset_hash", "run_id")):
        raise ValueError("evaluation_provenance")
    if not ck["metadata"]["training_episode_ids"]:
        raise ValueError("training_episode_ids")
    ds = EpisodeDataset(args.data_root, manifest, cfg, "policy", tokenizer)
    if ds.doc["split"] != "test": raise ValueError("Formal evaluation requires the test manifest")
    support = decision_support(ds.doc)
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
    for r, pred in zip(ds.rows, predictions):
        pred.update({key: r[key] for key in ("sample_id", "frame_index", "timestamp_s")})
    annotate_support(result, support)
    result.update(checkpoint_sha256=sha256_file(args.checkpoint),
                  manifest_sha256=sha256_file(manifest), run_id=ck["metadata"]["run_id"],
                  evaluation_source=source_identity(), predictions=predictions)
    if getattr(args, "dataset", None):
        result["dataset_index_sha256"] = sha256_file(args.dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k:v for k,v in result.items() if k != "predictions"}
