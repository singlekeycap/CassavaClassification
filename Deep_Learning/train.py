"""Train a ResNet on the Cassava dataset with imbalance handling.

Examples (run from the project root with venv/bin/python):
    python -m Deep_Learning.train --model resnet50 --imbalance sampler
    python -m Deep_Learning.train --model resnet50 --imbalance weights --loss focal
    python -m Deep_Learning.train --model resnet34 --imbalance oversample --val-fold 2
(``python Deep_Learning/train.py ...`` works too.)

The best free GPU is picked automatically (gpu_run.py); waits if none has 12 GB free.
Override with ``CUDA_VISIBLE_DEVICES=N python -m Deep_Learning.train ...``.

Model selection uses validation **macro-F1** (accuracy is misleading at 61% CMD).
Outputs go to Deep_Learning/runs/<run-name>/:
    best.pt, config.json, history.csv, history.png, test_report.txt, test_metrics.json,
    test_probs.npy, confusion_matrix.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # allow `python Deep_Learning/train.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "Deep_Learning"

# Pick the best free GPU automatically. Must run before `import torch`.
# Skipped if you set CUDA_VISIBLE_DEVICES yourself, e.g. CUDA_VISIBLE_DEVICES=1 python -m Deep_Learning.train
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, where gpu_run.py lives
from gpu_run import auto_select_gpu
auto_select_gpu(min_free_gb=12, wait=True)

import numpy as np
import pandas as pd
import torch

from Deep_Learning.config import NUM_CLASSES, RUNS_DIR
from Deep_Learning.data import build_dataloaders
from Deep_Learning.evaluation import plot_confusion, plot_history, text_report
from Deep_Learning.losses import build_loss
from Deep_Learning.models import AVAILABLE, build_resnet, param_groups, set_backbone_trainable
from Deep_Learning.training import evaluate, seed_everything, train_one_epoch, warmup_cosine


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="resnet50", choices=AVAILABLE)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-3, help="head LR; backbone uses lr * backbone-lr-mult")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--freeze-epochs", type=int, default=1, help="train only the head for the first N epochs")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--imbalance", default="sampler", choices=["none", "weights", "sampler", "oversample"])
    ap.add_argument("--weight-power", type=float, default=0.5, help="1 = inverse freq, 0.5 = sqrt inverse freq")
    ap.add_argument("--oversample-target", default="median", help='"max", "median" or an int')
    ap.add_argument("--loss", default="ce", choices=["ce", "focal"])
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--val-fold", type=int, default=0, help="which CV fold of splits.csv is validation")
    ap.add_argument("--split", default="grouped", choices=["grouped", "random"],
                    help="grouped (near-duplicates never cross splits) or naive random split, same sizes")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="seed of the random split, and of the --train-frac subset")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="train on this fraction of the train split (learning curve); val/test unchanged")
    ap.add_argument("--patience", type=int, default=5, help="early stopping on val macro-F1")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--tta", action="store_true", help="flip TTA at test time")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(argv)
    if args.oversample_target.isdigit():
        args.oversample_target = int(args.oversample_target)
    return args


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = True

    run_name = args.run_name or f"{args.model}_{args.imbalance}_{args.loss}_f{args.val_fold}_{time.strftime('%m%d-%H%M')}"
    out = RUNS_DIR / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"[run] {out}  device={device}  amp={use_amp}")

    # ---- data
    data = build_dataloaders(args.img_size, args.batch_size, args.num_workers, args.val_fold,
                             args.imbalance, args.weight_power, args.oversample_target, args.seed,
                             args.split, args.split_seed, args.train_frac)
    # Which images went where (needed later for the leakage analysis of random splits)
    pd.concat([d.assign(part=part)[["image_id", "label", "part"]]
               for part, d in zip(("train", "val", "test"), data.frames)]).to_csv(out / "split_ids.csv", index=False)

    # ---- loss
    weights = data.loss_weights.to(device) if data.loss_weights is not None else None
    if weights is not None:
        print("[loss] class weights:", weights.cpu().numpy().round(3))
    criterion = build_loss(args.loss, weights, args.label_smoothing, args.focal_gamma).to(device)
    eval_criterion = torch.nn.CrossEntropyLoss()   # unweighted, comparable across runs

    # ---- model / optimiser
    model = build_resnet(args.model, NUM_CLASSES, pretrained=True, dropout=args.dropout,
                         freeze_backbone=args.freeze_epochs > 0).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, args.lr, args.backbone_lr_mult, args.weight_decay))
    steps = len(data.train)
    scheduler = warmup_cosine(optimizer, int(args.warmup_epochs * steps), args.epochs * steps)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ---- train
    best_f1, best_epoch, history = -1.0, -1, []
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            set_backbone_trainable(model, True)
            print("[train] backbone unfrozen")
        t0 = time.time()
        tr_loss = train_one_epoch(model, data.train, criterion, optimizer, scheduler, scaler, device, use_amp)
        val, *_ = evaluate(model, data.val, eval_criterion, device, use_amp)
        row = {"epoch": epoch, "train_loss": tr_loss, **{f"val_{k}": v for k, v in val.items()},
               "lr_head": optimizer.param_groups[1]["lr"], "time_s": time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        improved = val["macro_f1"] > best_f1
        print(f"epoch {epoch:02d} | train_loss {tr_loss:.4f} | val_loss {val['loss']:.4f} "
              f"acc {val['acc']:.4f} bal_acc {val['balanced_acc']:.4f} macro_f1 {val['macro_f1']:.4f} "
              f"| {row['time_s']:.0f}s{'  *best*' if improved else ''}")
        if improved:
            best_f1, best_epoch = val["macro_f1"], epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val": val}, out / "best.pt")
        elif epoch - best_epoch >= args.patience:
            print(f"[train] early stopping (no val macro-F1 gain for {args.patience} epochs)")
            break
    plot_history(pd.DataFrame(history), out / "history.png")

    # ---- final evaluation on the held-out test set with the best checkpoint
    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test, labels, preds, probs = evaluate(model, data.test, eval_criterion, device, use_amp, tta=args.tta)
    report = text_report(labels, preds)
    print(f"\n[test] best epoch {best_epoch} (val macro-F1 {best_f1:.4f})\n{report}")
    (out / "test_report.txt").write_text(report)
    (out / "test_metrics.json").write_text(json.dumps({"best_epoch": best_epoch, "val_macro_f1": best_f1,
                                                       **{f"test_{k}": v for k, v in test.items()}}, indent=2))
    np.save(out / "test_probs.npy", probs)
    plot_confusion(labels, preds, out / "confusion_matrix.png")
    print(f"[done] artifacts in {out}")


if __name__ == "__main__":
    main()
