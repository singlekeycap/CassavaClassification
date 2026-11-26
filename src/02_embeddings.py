"""Extract L2-normalised DINOv2 embeddings for every training image.

Output: features/embeddings.npy (N x D float32, row order = train.csv order).
Used to find visually near-identical images (same plant / same photo session).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


class ImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "dataset", type=Path)
    ap.add_argument("--out", default=ROOT / "features" / "embeddings.npy", type=Path)
    ap.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--batch-size", default=64, type=int)
    ap.add_argument("--workers", default=8, type=int)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=224).eval().to(device)
    cfg = timm.data.resolve_data_config({"input_size": (3, 224, 224)}, model=model)
    transform = timm.data.create_transform(**cfg)

    df = pd.read_csv(args.data / "train.csv")
    paths = [args.data / "train_images" / f for f in df.image_id]
    loader = DataLoader(ImageDataset(paths, transform), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=True)

    feats = []
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        for x in tqdm(loader):
            f = model(x.to(device, non_blocking=True)).float()
            feats.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())

    emb = np.concatenate(feats).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, emb)
    print(f"wrote embeddings {emb.shape} -> {args.out}")


if __name__ == "__main__":
    main()
