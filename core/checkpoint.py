from dataclasses import asdict
from pathlib import Path
import json
import platform
import subprocess
import os
import shutil
import uuid
import torch
from .actions import ACTION_SCHEMA, ACTION_NAMES
from .config import ModelConfig, sha256_file, canonical_hash
from .model import build_model, clip_config_of, parameter_counts

FORMAT = "oceanvla.checkpoint.v2"
WEIGHTS_FORMAT = "oceanvla.weights.v3"

def source_identity():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        try: return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError): return None
    files = {}
    for folder in ("core", "configs"):
        for p in sorted((root/folder).rglob("*")):
            if p.is_file() and p.suffix in (".py", ".yaml"): files[p.relative_to(root).as_posix()] = sha256_file(p)
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "source_sha256": canonical_hash(files), "source_files": files}

def read_checkpoint(path):
    path = Path(path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ck, dict) and ck and all(isinstance(v, torch.Tensor) for v in ck.values()):
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            raise ValueError("Tensor-only checkpoint requires its matching .json sidecar")
        doc = json.loads(sidecar.read_text(encoding="utf-8"))
        if doc.get("format") != WEIGHTS_FORMAT or doc.get("weights_sha256") != sha256_file(path):
            raise ValueError("Weights/sidecar integrity mismatch")
        ck = {**doc, "format": FORMAT, "model_state_dict": ck}
    if not isinstance(ck, dict) or ck.get("format") != FORMAT:
        raise ValueError("Checkpoint is not a full versioned model. A partial policy cannot be converted into a trained full model; provide world and maritime CLIP checkpoints.")
    if ck.get("action_schema") != ACTION_SCHEMA or ck.get("action_names") != ACTION_NAMES: raise ValueError("Checkpoint action mapping mismatch")
    ck["model_config"] = asdict(ModelConfig(**ck["model_config"]).validate())
    return ck

def restore(path, expected_stage=None):
    ck = read_checkpoint(path)
    if expected_stage and ck["stage"] != expected_stage: raise ValueError(f"Expected {expected_stage} checkpoint")
    cfg = ModelConfig(**ck["model_config"])
    model = build_model(ck["stage"], cfg, hf_config=ck["clip_config"], pretrained=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return model, ck

def save_checkpoint(path, model, stage, metadata, **training_state):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ck = {"format": FORMAT, "stage": stage, "model_config": asdict(model.cfg),
          "clip_config": clip_config_of(model), "action_schema": ACTION_SCHEMA,
          "action_names": ACTION_NAMES, "model_state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
          "parameter_count": parameter_counts(model), "metadata": metadata,
          "environment": {"python": platform.python_version(), "torch": str(torch.__version__)},
          **training_state}
    temporary = path.with_suffix(path.suffix+".partial")
    torch.save(ck, temporary)
    os.replace(temporary, path)
    digest = sha256_file(path)
    path.with_suffix(path.suffix+".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest

def check_tokenizer(ck, checkpoint_path):
    directory = Path(checkpoint_path).parent / "tokenizer"
    expected = ck["metadata"].get("tokenizer_files")
    if not expected: raise ValueError("Missing tokenizer manifest")
    for name, digest in expected.items():
        if Path(name).name != name or sha256_file(directory/name) != digest: raise ValueError("Tokenizer integrity failure")
    return directory


def export_weights(path, model, stage, metadata, *, replace=False):
    """Write tensors only in .pt; retain configuration and provenance in JSON.

    No fabricated metrics or missing modules are synthesized. The reader checks
    the digest before loading, including after an interrupted two-file update.
    """
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if not replace and (path.exists() or sidecar.exists()):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temp = path.with_name(path.name + "." + token + ".partial")
    temp_json = sidecar.with_name(sidecar.name + "." + token + ".partial")
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if not state or not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Cannot export empty/nonfinite model")
    if stage == "policy" and not all(any(k.startswith(prefix) for k in state)
                                      for prefix in ("encoder.", "world.", "world_projection.", "policy.", "action_head.")):
        raise ValueError("Full policy export requires encoder, world, projection, policy and action head")
    torch.save(state, temp)
    digest = sha256_file(temp)
    doc = {"format": WEIGHTS_FORMAT, "stage": stage, "model_config": asdict(model.cfg),
           "clip_config": clip_config_of(model), "action_schema": ACTION_SCHEMA,
           "action_names": ACTION_NAMES, "parameter_count": parameter_counts(model),
           "metadata": metadata, "weights_sha256": digest,
           "environment": {"python": platform.python_version(), "torch": str(torch.__version__)}}
    temp_json.write_text(json.dumps(doc, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)
    os.replace(temp_json, sidecar)
    return digest


def export_checkpoint(source, destination):
    model, ck = restore(source)
    destination = Path(destination)
    if ck["stage"] != "world":
        tokenizer = check_tokenizer(ck, source)
        target = destination.parent / "tokenizer"
        if tokenizer.resolve() != target.resolve():
            if target.exists():
                check_tokenizer(ck, destination)
            else:
                shutil.copytree(tokenizer, target)
    return export_weights(destination, model, ck["stage"], ck["metadata"])
