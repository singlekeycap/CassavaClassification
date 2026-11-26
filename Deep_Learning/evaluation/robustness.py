"""Robustness study: re-score trained models on corrupted copies of the test set.

No retraining. Each test image is preprocessed exactly as in training evaluation
(Resize(384) -> CenterCrop(384)), then corrupted, then normalised and scored by every model.

Corruptions (3 severities each), chosen as things a phone photo taken in a field can suffer:
    blur        Gaussian blur, sigma 1 / 2 / 4 px                 (out of focus, motion)
    dark        brightness x0.6 / x0.4 / x0.25                     (shade, evening)
    bright      brightness x1.4 / x1.7 / x2.0                      (direct sun, overexposed)
    noise       Gaussian noise, std 10 / 20 / 35 (0-255 scale)     (low light, cheap sensor)
    jpeg        JPEG quality 30 / 15 / 7                           (compression when sent/uploaded)
    lowres      downsample to 192 / 96 / 64 px, upsample back      (low-resolution camera, zoom)

Before anything else, the clean predictions are checked against each run's test_probs.npy.

Output (default reports/robustness_eval/):
    probs.npz      probs[run][variant] -> (n_test, 5) float16
    metrics.csv    one row per (run, variant): acc, balanced acc, macro-F1, per-class recall
    examples.png   one test image under every corruption (visual check)

Run from the project root (picks a free GPU):
    venv/bin/python gpu_run.py -- venv/bin/python -m Deep_Learning.evaluation.robustness
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from ..config import IMAGE_DIR, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, ROOT, RUNS_DIR, SPLITS_CSV
from ..models import build_resnet
from .metrics import compute_metrics

SEVERITIES = {
    "blur": [1.0, 2.0, 4.0],
    "dark": [0.6, 0.4, 0.25],
    "bright": [1.4, 1.7, 2.0],
    "noise": [10, 20, 35],
    "jpeg": [30, 15, 7],
    "lowres": [192, 96, 64],
}


# ----------------------------------------------------------------------- data
def load_test_uint8(img_size: int, cache: Path) -> np.ndarray:
    """(N, 3, H, W) uint8 test images after Resize + CenterCrop (cached on disk)."""
    if cache.exists():
        return np.load(cache)
    test = pd.read_csv(SPLITS_CSV).query("split == 'test'").reset_index(drop=True)
    out = np.empty((len(test), 3, img_size, img_size), np.uint8)
    for i, image_id in enumerate(test.image_id):
        with Image.open(IMAGE_DIR / image_id) as im:
            im = TF.center_crop(TF.resize(im.convert("RGB"), img_size), img_size)
            out[i] = np.asarray(im).transpose(2, 0, 1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def corrupt(x: np.ndarray, kind: str, level, seed: int = 0) -> np.ndarray:
    """Apply one corruption to a (N, 3, H, W) uint8 batch; returns uint8."""
    if kind == "clean":
        return x
    if kind == "jpeg":
        out = np.empty_like(x)
        for i, img in enumerate(x):
            buf = io.BytesIO()
            Image.fromarray(img.transpose(1, 2, 0)).save(buf, format="JPEG", quality=int(level))
            out[i] = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")).transpose(2, 0, 1)
        return out
    t = torch.from_numpy(x).float()
    if kind == "blur":
        k = int(2 * round(3 * level) + 1)
        t = TF.gaussian_blur(t, kernel_size=[k, k], sigma=[level, level])
    elif kind in ("dark", "bright"):
        t = t * level
    elif kind == "noise":
        g = torch.Generator().manual_seed(seed)
        t = t + torch.randn(t.shape, generator=g) * level
    elif kind == "lowres":
        size = t.shape[-1]
        t = F.interpolate(t, size=(int(level), int(level)), mode="bilinear", antialias=True, align_corners=False)
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    else:
        raise ValueError(kind)
    return t.round().clamp(0, 255).to(torch.uint8).numpy()


def variants():
    yield "clean", 0, None
    for kind, levels in SEVERITIES.items():
        for s, level in enumerate(levels, 1):
            yield kind, s, level


# ----------------------------------------------------------------------- models
def grouped_full_runs(pattern: str) -> list[Path]:
    """Full-data, grouped-split runs (the 20 runs of steps 1-2)."""
    runs = []
    for d in sorted(RUNS_DIR.glob(pattern)):
        if not (d / "best.pt").exists() or not (d / "test_probs.npy").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        if cfg.get("split", "grouped") == "grouped" and cfg.get("train_frac", 1.0) == 1.0:
            runs.append(d)
    return runs


def load_model(run: Path, device) -> torch.nn.Module:
    ckpt = torch.load(run / "best.pt", map_location="cpu")
    args = ckpt["args"]
    model = build_resnet(args["model"], NUM_CLASSES, pretrained=False, dropout=args["dropout"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval().to(memory_format=torch.channels_last)


@torch.no_grad()
def predict(model, x_uint8: np.ndarray, device, batch: int) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1) * 255
    out = []
    for i in range(0, len(x_uint8), batch):
        xb = torch.from_numpy(x_uint8[i:i + batch]).to(device, non_blocking=True).float()
        xb = ((xb - mean) / std).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(xb)
        out.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------- main
def save_examples(x: np.ndarray, path: Path, idx: int = 0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = list(SEVERITIES)
    fig, axes = plt.subplots(len(kinds), 4, figsize=(8, 2 * len(kinds)))
    for r, kind in enumerate(kinds):
        for c in range(4):
            img = x[idx:idx + 1] if c == 0 else corrupt(x[idx:idx + 1], kind, SEVERITIES[kind][c - 1])
            axes[r, c].imshow(img[0].transpose(1, 2, 0))
            axes[r, c].set_title("clean" if c == 0 else f"{kind} s{c}", fontsize=8)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="r50_*")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "robustness_eval")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    x = load_test_uint8(args.img_size, args.out / f"test_uint8_{args.img_size}.npy")
    y = pd.read_csv(SPLITS_CSV).query("split == 'test'").label.to_numpy()
    print(f"[data] {len(x)} test images cached ({time.time() - t0:.0f}s)")
    save_examples(x, args.out / "examples.png")

    runs = grouped_full_runs(args.runs)
    print(f"[models] {len(runs)} runs: {[r.name for r in runs]}")
    models = {r.name: load_model(r, device) for r in runs}

    probs, rows = {r: {} for r in models}, []
    for kind, sev, level in variants():
        t1 = time.time()
        xc = corrupt(x, kind, level)
        tag = "clean" if kind == "clean" else f"{kind}_s{sev}"
        for name, model in models.items():
            p = predict(model, xc, device, args.batch_size)
            if kind == "clean":   # same pipeline as training? compare with the saved test predictions
                saved = np.load(RUNS_DIR / name / "test_probs.npy")
                agree = (p.argmax(1) == saved.argmax(1)).mean()
                if agree < 0.995:
                    raise SystemExit(f"[error] {name}: clean predictions agree with test_probs.npy on "
                                     f"only {agree:.2%} of images; the pipeline differs from training eval")
            probs[name][tag] = p.astype(np.float16)
            pred = p.argmax(1)
            recall = [(pred[y == c] == c).mean() for c in range(NUM_CLASSES)]
            cfg = json.loads((RUNS_DIR / name / "config.json").read_text())
            rows.append({"run": name, "method": cfg["imbalance"], "fold": cfg["val_fold"],
                         "corruption": kind, "severity": sev, "level": level,
                         **compute_metrics(y, pred), **{f"recall_{c}": r for c, r in enumerate(recall)}})
        print(f"[eval] {tag:<10} {time.time() - t1:5.0f}s  mean macro-F1 "
              f"{np.mean([r['macro_f1'] for r in rows if r['corruption'] == kind and r['severity'] == sev]):.4f}",
              flush=True)

    pd.DataFrame(rows).to_csv(args.out / "metrics.csv", index=False, float_format="%.5f")
    np.savez_compressed(args.out / "probs.npz",
                        **{f"{run}__{tag}": p for run, d in probs.items() for tag, p in d.items()})
    print(f"[done] {time.time() - t0:.0f}s, outputs in {args.out}")


if __name__ == "__main__":
    main()
