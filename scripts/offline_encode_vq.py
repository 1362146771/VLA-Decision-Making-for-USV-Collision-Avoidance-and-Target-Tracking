import os, argparse, glob
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

from core.modeling.world.vqgan_tokenizer import VQGANTokenizer

def build_preprocess(size=256):
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),                 # [0,1]
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--vq_yaml", required=True)
    ap.add_argument("--vq_ckpt", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    preprocess = build_preprocess(args.img_size)
    vq = VQGANTokenizer(args.vq_yaml, args.vq_ckpt)

    paths = sorted(glob.glob(os.path.join(args.img_dir, "*")))
    batch = []
    names = []

    def flush():
        if not batch: return
        imgs = torch.stack(batch, dim=0)
        codes, (Gh, Gw) = vq.encode(imgs)
        for i, name in enumerate(names):
            np.save(os.path.join(args.out_dir, os.path.basename(name) + ".npy"),
                    codes[i].cpu().numpy())
        batch.clear(); names.clear()

    for p in paths:
        im = Image.open(p).convert("RGB")
        x = preprocess(im)
        batch.append(x); names.append(os.path.splitext(os.path.basename(p))[0])
        if len(batch) == args.batch_size:
            flush()
    flush()
    print("Done.")

if __name__ == "__main__":
    main()
