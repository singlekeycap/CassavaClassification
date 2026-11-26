"""Dataset analysis for the interim report.

Reads dataset/, features/ and splits/ and writes figures + a stats summary to reports/.
Run after 01_hashes.py, 02_embeddings.py and 03_group_and_split.py.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Reference palette (dataviz skill), light mode only: figures go into a printed report.
SURFACE, INK, INK_2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, RED = "#2a78d6", "#e34948"
SHORT = {0: "CBB", 1: "CBSD", 2: "CGM", 3: "CMD", 4: "Healthy"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
})


def image_stats(path):
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB").resize((200, 150)), dtype=np.float32)
        return im.size[0], im.size[1], im.mode, *a.mean((0, 1)), a.mean()


def plot_class_distribution(df, names, out):
    counts = df.label.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    y = np.arange(len(counts))[::-1]
    ax.barh(y, counts.values, height=0.6, color=BLUE)
    ax.set_yticks(y, [names[k] for k in counts.index])
    for yi, c in zip(y, counts.values):
        ax.text(c + 150, yi, f"{c:,}  ({c / len(df):.1%})", va="center", color=INK_2)
    ax.set_xlim(0, counts.max() * 1.35)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlabel("Training images")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle(f"Class distribution (n = {len(df):,}; imbalance ratio {counts.max() / counts.min():.1f}:1)",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_samples(df, data, out, per_class=5, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(5, per_class, figsize=(per_class * 1.6, 5 * 1.25))
    for label in range(5):
        ids = rng.choice(df[df.label == label].image_id.values, per_class, replace=False)
        for j, img_id in enumerate(ids):
            ax = axes[label, j]
            ax.imshow(Image.open(data / "train_images" / img_id))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[label, 0].set_ylabel(SHORT[label], rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle("Random training images per class", x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def nn_similarity(features):
    emb = np.load(features / "embeddings.npy")
    emb = emb - emb.mean(0)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(emb).to(device)
    best = []
    for i in range(0, len(x), 2048):
        s = x[i:i + 2048] @ x.T
        s[torch.arange(len(s)), torch.arange(i, i + len(s))] = -1
        best.append(s.max(1).values.cpu())
    return torch.cat(best).numpy()


def plot_nn_similarity(nn1, threshold, out):
    fig, ax = plt.subplots(figsize=(5.5, 2.3))
    ax.hist(nn1, bins=60, color=BLUE, rwidth=0.85)
    ax.axvline(threshold, color=INK, lw=1, ls="--")
    above = (nn1 >= threshold).mean()
    ax.text(threshold + 0.005, ax.get_ylim()[1] * 0.9, f"grouping threshold {threshold:.2f}\n{above:.1%} of images above",
            color=INK_2, va="top")
    ax.set_xlabel("Cosine similarity to nearest other image (mean-centred DINOv2 embedding)")
    ax.set_ylabel("Images")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Nearest-neighbour similarity", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_groups(splits, data, out, n_groups=5, per_group=5, seed=0):
    """Example multi-image groups: two with conflicting labels, the rest consistent."""
    rng = np.random.default_rng(seed)
    multi = splits[splits.group_size.between(3, 12)]
    n_labels = multi.groupby("group_id").label.nunique()
    conflict = rng.choice(n_labels[n_labels > 1].index, 2, replace=False)
    clean = rng.choice(n_labels[n_labels == 1].index, n_groups - 2, replace=False)
    fig, axes = plt.subplots(n_groups, per_group, figsize=(per_group * 1.6, n_groups * 1.3))
    for row, g in enumerate([*clean, *conflict]):
        members = splits[splits.group_id == g].head(per_group)
        for j in range(per_group):
            ax = axes[row, j]
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j < len(members):
                r = members.iloc[j]
                ax.imshow(Image.open(data / "train_images" / r.image_id))
                ax.set_title(SHORT[r.label], fontsize=8, color=RED if g in conflict else INK_2, pad=2)
        axes[row, 0].set_ylabel(f"group\n(n={members.group_size.iloc[0]})", rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("Example near-duplicate groups (bottom two rows: conflicting labels in red)",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "dataset", type=Path)
    ap.add_argument("--features", default=ROOT / "features", type=Path)
    ap.add_argument("--splits", default=ROOT / "splits", type=Path)
    ap.add_argument("--out", default=ROOT / "reports", type=Path)
    ap.add_argument("--workers", default=12, type=int)
    args = ap.parse_args()
    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data / "train.csv")
    names = {int(k): v for k, v in json.loads((args.data / "label_num_to_disease_map.json").read_text()).items()}
    splits = pd.read_csv(args.splits / "splits.csv")
    split_stats = json.loads((args.splits / "split_stats.json").read_text())

    # Integrity + per-image colour statistics.
    paths = [args.data / "train_images" / f for f in df.image_id]
    with Pool(args.workers) as pool:
        rows = pool.map(image_stats, paths, chunksize=64)
    img = pd.DataFrame(rows, columns=["w", "h", "mode", "r", "g", "b", "brightness"])
    img["label"] = df.label.values

    counts = df.label.value_counts().sort_index()
    nn1 = nn_similarity(args.features)
    threshold = split_stats["params"]["threshold"]

    plot_class_distribution(df, names, fig_dir / "class_distribution.png")
    plot_samples(df, args.data, fig_dir / "samples_per_class.png")
    plot_nn_similarity(nn1, threshold, fig_dir / "nn_similarity.png")
    plot_groups(splits, args.data, fig_dir / "group_examples.png")

    summary = {
        "n_train_images": len(df),
        "n_test_images_provided": len(list((args.data / "test_images").iterdir())),
        "missing_files": int(sum(not p.exists() for p in paths)),
        "duplicate_ids": int(df.image_id.duplicated().sum()),
        "image_sizes": {f"{w}x{h}": int(c) for (w, h), c in img.groupby(["w", "h"]).size().items()},
        "image_modes": img["mode"].value_counts().to_dict(),
        "class_counts": {SHORT[k]: int(v) for k, v in counts.items()},
        "class_pct": {SHORT[k]: round(v / len(df) * 100, 2) for k, v in counts.items()},
        "imbalance_ratio_max_min": round(counts.max() / counts.min(), 2),
        "majority_baseline_accuracy": round(counts.max() / len(df), 4),
        "majority_baseline_macro_f1": round((2 * counts.max() / (counts.max() + len(df))) / 5, 4),
        "inverse_freq_class_weights": {SHORT[k]: round(len(df) / (5 * v), 3) for k, v in counts.items()},
        "mean_rgb": [round(float(img[c].mean()), 1) for c in "rgb"],
        "std_rgb_of_image_means": [round(float(img[c].std()), 1) for c in "rgb"],
        "brightness_by_class": {SHORT[k]: round(float(v), 1) for k, v in img.groupby("label").brightness.mean().items()},
        "nn_similarity_quantiles": {str(q): round(float(np.quantile(nn1, q)), 3) for q in (0.5, 0.9, 0.99)},
        "grouping": {k: split_stats[k] for k in ("n_groups", "n_multi_image_groups", "images_in_multi_image_groups",
                                                 "largest_group", "groups_with_conflicting_labels")},
        "split_sizes": split_stats["split_sizes"],
    }
    (args.out / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
