"""Group near-duplicate / same-session images and build a grouped, stratified split.

Unit of correlation: the photo session / plant. The dataset ships no plant, field or
farmer IDs, so we approximate it by linking images whose mean-centred DINOv2
embeddings are mutual k-nearest neighbours with cosine similarity >= threshold, and
taking connected components as groups. pHash near-duplicates (Hamming <= --phash-max)
are linked as well.

Split: StratifiedGroupKFold -> one fold held out as the test set (~1/test_folds of the
data), then 5-fold StratifiedGroupKFold CV on the remainder. No group crosses a split.

Output: splits/splits.csv  (image_id, label, group_id, group_size, split, fold)
        splits/split_stats.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]


def knn(emb, k, chunk=2048):
    """Top-k cosine neighbours (excluding self) for L2-normalised rows."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(emb).to(device)
    sims, idx = [], []
    for i in range(0, len(x), chunk):
        s = x[i:i + chunk] @ x.T
        s[torch.arange(len(s)), torch.arange(i, i + len(s))] = -1
        v, j = s.topk(k, dim=1)
        sims.append(v.cpu())
        idx.append(j.cpu())
    return torch.cat(sims).numpy(), torch.cat(idx).numpy()


def phash_pairs(hashes, max_dist):
    bits = np.array([[int(b) for b in bin(int(h, 16))[2:].zfill(len(h) * 4)] for h in hashes], dtype=np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b = torch.from_numpy(bits).to(device)
    rows, cols = [], []
    for i in range(0, len(b), 2048):
        a = b[i:i + 2048]
        d = a @ (1 - b).T + (1 - a) @ b.T
        r, c = torch.where(d <= max_dist)
        keep = (r + i) < c
        rows.append((r[keep] + i).cpu().numpy())
        cols.append(c[keep].cpu().numpy())
    return np.concatenate(rows), np.concatenate(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=ROOT / "features", type=Path)
    ap.add_argument("--out", default=ROOT / "splits", type=Path)
    ap.add_argument("--k", default=3, type=int, help="mutual-kNN neighbourhood size")
    ap.add_argument("--threshold", default=0.80, type=float, help="cosine sim on centred embeddings")
    ap.add_argument("--phash-max", default=24, type=int, help="max pHash Hamming distance (256-bit)")
    ap.add_argument("--test-folds", default=7, type=int, help="hold out 1/test_folds as test")
    ap.add_argument("--cv-folds", default=5, type=int)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    df = pd.read_csv(args.features / "hashes.csv")
    emb = np.load(args.features / "embeddings.npy")
    n = len(df)

    # Centre so similarity reflects instance differences, not "it's a cassava photo".
    emb = emb - emb.mean(0)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    sims, idx = knn(emb, args.k)
    neighbours = [set(row) for row in idx]
    r, c = np.where(sims >= args.threshold)
    c = idx[r, c]
    mutual = np.array([a in neighbours[b] for a, b in zip(r, c)], dtype=bool)
    emb_r, emb_c = r[mutual], c[mutual]

    ph_r, ph_c = phash_pairs(df.phash, args.phash_max)
    rows = np.concatenate([emb_r, ph_r])
    cols = np.concatenate([emb_c, ph_c])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, df["group_id"] = connected_components(graph, directed=False)
    sizes = df.group_id.map(df.group_id.value_counts())
    df["group_size"] = sizes

    # Held-out test set.
    outer = StratifiedGroupKFold(n_splits=args.test_folds, shuffle=True, random_state=args.seed)
    trainval_idx, test_idx = next(outer.split(df, df.label, df.group_id))
    df["split"] = "trainval"
    df.loc[test_idx, "split"] = "test"
    df["fold"] = -1

    # CV folds on train+val.
    tv = df.iloc[trainval_idx]
    inner = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    for fold, (_, val_idx) in enumerate(inner.split(tv, tv.label, tv.group_id)):
        df.loc[tv.index[val_idx], "fold"] = fold

    # Sanity checks: no group is shared between test and trainval, or between folds.
    assert df.groupby("group_id").split.nunique().max() == 1
    assert df[df.split == "trainval"].groupby("group_id").fold.nunique().max() == 1

    group_counts = df.group_id.value_counts()
    multi = group_counts[group_counts > 1]
    conflict = int((df[df.group_id.isin(multi.index)].groupby("group_id").label.nunique() > 1).sum())
    by_split = pd.crosstab(df.split.where(df.split == "test", "fold" + df.fold.astype(str)), df.label)
    stats = {
        "n_images": n,
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "edges": {"embedding_mutual_knn": int(mutual.sum()), "phash": int(len(ph_r))},
        "n_groups": int(len(group_counts)),
        "n_multi_image_groups": int(len(multi)),
        "images_in_multi_image_groups": int(multi.sum()),
        "largest_group": int(group_counts.max()),
        "groups_with_conflicting_labels": conflict,
        "split_sizes": df.split.where(df.split == "test", "fold" + df.fold.astype(str)).value_counts().sort_index().to_dict(),
        "class_counts_by_split": by_split.to_dict(orient="index"),
        "class_pct_by_split": (by_split.div(by_split.sum(1), axis=0) * 100).round(2).to_dict(orient="index"),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    df[["image_id", "label", "group_id", "group_size", "split", "fold"]].to_csv(args.out / "splits.csv", index=False)
    (args.out / "split_stats.json").write_text(json.dumps(stats, indent=2))

    print(json.dumps({k: v for k, v in stats.items() if k not in ("class_counts_by_split", "params")}, indent=2))
    print("\nclass % by split:\n", (by_split.div(by_split.sum(1), axis=0) * 100).round(1))


if __name__ == "__main__":
    main()
