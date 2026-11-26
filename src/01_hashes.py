"""Compute perceptual hashes for every training image.

Output: features/hashes.csv with columns image_id, label, phash, dhash (hex strings).
Used to find exact / near-exact duplicates for the grouped split.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def hash_one(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        return str(imagehash.phash(im, hash_size=16)), str(imagehash.dhash(im, hash_size=16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "dataset", type=Path)
    ap.add_argument("--out", default=ROOT / "features" / "hashes.csv", type=Path)
    ap.add_argument("--workers", default=12, type=int)
    args = ap.parse_args()

    df = pd.read_csv(args.data / "train.csv")
    paths = [args.data / "train_images" / f for f in df.image_id]

    with Pool(args.workers) as pool:
        hashes = list(tqdm(pool.imap(hash_one, paths, chunksize=64), total=len(paths)))

    df["phash"], df["dhash"] = zip(*hashes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} hashes -> {args.out}")


if __name__ == "__main__":
    main()
