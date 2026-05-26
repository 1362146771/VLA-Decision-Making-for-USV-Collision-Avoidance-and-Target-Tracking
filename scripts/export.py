# core/scripts/export.py
import os, sys, json, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.modeling.ocean_model import OceanVLA

def main(ckpt_out="checkpoints/oceanvla-min.pt", config_out="checkpoints/oceanvla-min.json"):
    os.makedirs(os.path.dirname(ckpt_out), exist_ok=True)
    model = OceanVLA()
    torch.save(model.state_dict(), ckpt_out)
    cfg = {"note": "minimal export", "model": "OceanVLA"}
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"Saved: {ckpt_out}\nSaved: {config_out}")

if __name__ == "__main__":
    main()
