"""Class-imbalance figures, tables and statistics for the paper.

Outputs (default: reports/figures/imbalance/):
    fig1_class_distribution.{png,pdf}   images per class, full dataset
    fig2_split_distribution.{png,pdf}   class mix of full / train / val / test (stratification check)
    fig3_resampling_effect.{png,pdf}    training class mix under each imbalance strategy
    fig4_class_weights.{png,pdf}        inverse-frequency loss weights (power 1 and 0.5)
    imbalance_table.csv / .tex          counts and % per class and split (LaTeX booktabs)
    imbalance_stats.json                imbalance ratio, entropy, majority baseline, etc.

Figures are light-mode, print-oriented: PNG at 300 dpi plus vector PDF for LaTeX.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from ..config import NUM_CLASSES, TRAIN_CSV, class_names, short_class_names
from ..data import class_counts, class_weights, get_splits, oversample

# Reference palette (dataviz skill), light mode; same tokens as src/04_eda.py.
SURFACE, INK, INK_2, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE = "#2a78d6"
HALO = dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none")   # text sits on top of ref lines
# Categorical slots 1-5 in fixed order; colour follows the class, never its rank.
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SERIES_2 = ["#2a78d6", "#eb6834"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType fonts (journal requirement)
})


# ----------------------------------------------------------------------- stats
def imbalance_metrics(labels) -> dict:
    counts = class_counts(labels)
    p = counts / counts.sum()
    majority = int(counts.argmax())
    # Predicting the majority class for everything:
    f1_major = 2 * p[majority] / (1 + p[majority])
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {
        "n_images": int(counts.sum()),
        "counts": dict(zip(short_class_names(), counts.tolist())),
        "percent": dict(zip(short_class_names(), (100 * p).round(2).tolist())),
        "majority_class": short_class_names()[majority],
        "minority_class": short_class_names()[int(counts.argmin())],
        "imbalance_ratio_max_min": round(float(counts.max() / counts.min()), 2),
        "imbalance_ratio_max_median": round(float(counts.max() / np.median(counts)), 2),
        "normalised_entropy": round(entropy / np.log(len(counts)), 4),   # 1 = perfectly balanced
        "majority_baseline_accuracy": round(float(p[majority]), 4),
        "majority_baseline_macro_f1": round(float(f1_major / len(counts)), 4),
    }


def split_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows = classes (+ Total), columns = (split, n / %)."""
    cols = {}
    for name, df in frames.items():
        c = class_counts(df.label)
        cols[(name, "n")] = list(c) + [c.sum()]
        cols[(name, "%")] = list((100 * c / c.sum()).round(2)) + [100.0]
    return pd.DataFrame(cols, index=class_names() + ["Total"])


def to_latex(table: pd.DataFrame, path: Path):
    splits = list(dict.fromkeys(s for s, _ in table.columns))
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Class distribution of the Cassava Leaf Disease dataset and of the grouped "
        r"train / validation / test splits.}",
        r"\label{tab:class-distribution}",
        r"\begin{tabular}{l" + "rr" * len(splits) + "}", r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{s}}}" for s in splits) + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(splits))),
        "Class & " + " & ".join([r"$n$ & \%"] * len(splits)) + r" \\", r"\midrule",
    ]
    for i, (cls, row) in enumerate(table.iterrows()):
        if i == len(table) - 1:
            lines.append(r"\midrule")
        cells = []
        for s in splits:
            cells += [f"{int(row[(s, 'n')]):,}", f"{row[(s, '%')]:.1f}"]
        lines.append(cls.replace("&", r"\&") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------- figures
def _save(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _clean_axis(ax, grid_axis="x"):
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def plot_class_distribution(labels, out: Path):
    """Fig 1: horizontal bars sorted by size, with counts, % and the balanced reference."""
    counts = class_counts(labels)
    n = counts.sum()
    order = np.argsort(counts)               # smallest at bottom -> largest on top
    names = short_class_names()
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    y = np.arange(NUM_CLASSES)
    ax.barh(y, counts[order], height=0.62, color=BLUE)
    ax.set_yticks(y, [names[k] for k in order])
    for yi, k in zip(y, order):
        ax.text(counts[k] + n * 0.006, yi, f"{counts[k]:,}  ({counts[k] / n:.1%})", va="center", color=INK_2,
                bbox=HALO, zorder=3)
    balanced = n / NUM_CLASSES
    ax.axvline(balanced, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(balanced, NUM_CLASSES - 0.45, f" balanced = {balanced:,.0f} per class", color=MUTED,
            fontsize=8, va="bottom")
    ax.set_xlim(0, counts.max() * 1.32)
    ax.set_ylim(-0.6, NUM_CLASSES - 0.1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0"))
    ax.set_xlabel("Number of images")
    ax.spines["left"].set_visible(False)
    _clean_axis(ax, "x")
    ax.set_title(f"Class distribution (n = {n:,}; imbalance ratio {counts.max() / counts.min():.1f} : 1)",
                 loc="left", fontsize=10)
    _save(fig, out, "fig1_class_distribution")


def _stacked_share(ax, rows: dict[str, np.ndarray], right_note: dict[str, str] | None = None):
    """100% stacked horizontal bars, one row per entry, segments in class order."""
    labels = list(rows)[::-1]                 # first entry on top
    names = short_class_names()
    for yi, key in enumerate(labels):
        share = rows[key] / rows[key].sum()
        left = 0.0
        for k in range(NUM_CLASSES):
            ax.barh(yi, share[k], left=left, height=0.62, color=CLASS_COLORS[k],
                    edgecolor=SURFACE, linewidth=1.5, label=names[k] if yi == 0 else None)
            if share[k] >= 0.045:             # direct label only where it fits
                ax.text(left + share[k] / 2, yi, f"{share[k]:.0%}", ha="center", va="center",
                        fontsize=7.5, color=INK)
            left += share[k]
        if right_note:
            ax.text(1.01, yi, right_note[key], va="center", color=INK_2, fontsize=8)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(ncol=NUM_CLASSES, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False,
              handlelength=1.2, columnspacing=1.2, fontsize=8)


def plot_split_distribution(frames: dict[str, pd.DataFrame], out: Path):
    """Fig 2: class mix per split - shows the imbalance is preserved in every split."""
    rows = {k: class_counts(v.label) for k, v in frames.items()}
    notes = {k: f"n = {v.sum():,}" for k, v in rows.items()}
    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(rows) + 1.0))
    _stacked_share(ax, rows, notes)
    ax.set_title("Class proportions per split", loc="left", fontsize=10, pad=22)
    _save(fig, out, "fig2_split_distribution")


def expected_sampler_counts(labels, power: float) -> np.ndarray:
    """Expected class counts per epoch under WeightedRandomSampler(weights = class_weight ** power)."""
    counts = class_counts(labels).astype(float)
    w = class_weights(labels, power).numpy()
    mass = counts * w
    return mass / mass.sum() * counts.sum()


def plot_resampling_effect(train: pd.DataFrame, out: Path):
    """Fig 3: training-set class mix under each imbalance strategy available in data/."""
    rows = {
        "Original": class_counts(train.label),
        "Sampler (sqrt inv. freq.)": expected_sampler_counts(train.label, 0.5),
        "Sampler (inv. freq.)": expected_sampler_counts(train.label, 1.0),
        "Oversample (median)": class_counts(oversample(train, "median").label),
        "Oversample (max)": class_counts(oversample(train, "max").label),
    }
    notes = {k: f"{v.sum():,.0f} / epoch" for k, v in rows.items()}
    fig, ax = plt.subplots(figsize=(6.8, 0.5 * len(rows) + 1.0))
    _stacked_share(ax, rows, notes)
    ax.set_title("Training-set class mix under each imbalance strategy", loc="left", fontsize=10, pad=22)
    _save(fig, out, "fig3_resampling_effect")
    return {k: np.round(v).astype(int).tolist() for k, v in rows.items()}


def plot_class_weights(train: pd.DataFrame, out: Path):
    """Fig 4: inverse-frequency loss weights, full vs square-root."""
    names = short_class_names()
    w1 = class_weights(train.label, 1.0).numpy()
    w05 = class_weights(train.label, 0.5).numpy()
    x = np.arange(NUM_CLASSES)
    bw = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    for off, w, lab, col in ((-bw / 2, w1, "inverse frequency", SERIES_2[0]),
                             (bw / 2, w05, "sqrt inverse frequency", SERIES_2[1])):
        ax.bar(x + off, w, width=bw - 0.03, color=col, label=lab)
        for xi, v in zip(x + off, w):
            ax.text(xi, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK_2,
                    bbox=HALO, zorder=3)
    ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1, label="unweighted (1.0)")
    ax.set_xticks(x, names)
    ax.set_ylabel("Loss weight (mean = 1)")
    ax.set_ylim(0, max(w1.max(), w05.max()) * 1.3)
    ax.spines["left"].set_visible(False)
    _clean_axis(ax, "y")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_title("Class weights for the weighted loss (training split)", loc="left", fontsize=10)
    _save(fig, out, "fig4_class_weights")


# ------------------------------------------------------------------------ main
def run(out: Path | str, val_fold: int = 0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(TRAIN_CSV)
    train, val, test = get_splits(val_fold)
    frames = {"Full": full, "Train": train, "Validation": val, "Test": test}

    table = split_table(frames)
    table.to_csv(out / "imbalance_table.csv")
    to_latex(table, out / "imbalance_table.tex")

    plot_class_distribution(full.label, out)
    plot_split_distribution(frames, out)
    resampled = plot_resampling_effect(train, out)
    plot_class_weights(train, out)

    stats = {
        "full_dataset": imbalance_metrics(full.label),
        "per_split": {k: imbalance_metrics(v.label) for k, v in frames.items() if k != "Full"},
        "training_counts_per_strategy": resampled,
        "class_weights_train": {
            "power_1.0": dict(zip(short_class_names(), class_weights(train.label, 1.0).numpy().round(3).tolist())),
            "power_0.5": dict(zip(short_class_names(), class_weights(train.label, 0.5).numpy().round(3).tolist())),
        },
    }
    (out / "imbalance_stats.json").write_text(json.dumps(stats, indent=2))

    fs = stats["full_dataset"]
    print(table.to_string(), "\n")
    print(f"imbalance ratio (max/min): {fs['imbalance_ratio_max_min']} : 1   "
          f"normalised entropy: {fs['normalised_entropy']}   "
          f"majority baseline: acc {fs['majority_baseline_accuracy']:.3f}, "
          f"macro-F1 {fs['majority_baseline_macro_f1']:.3f}")
    print(f"\n[done] figures, tables and stats in {out}")
