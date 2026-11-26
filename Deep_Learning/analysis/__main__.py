import argparse

from ..config import ROOT
from .imbalance_report import run

ap = argparse.ArgumentParser(description="Class-imbalance figures and tables for the paper.")
ap.add_argument("--out", default=ROOT / "reports" / "figures" / "imbalance")
ap.add_argument("--val-fold", type=int, default=0)
args = ap.parse_args()
run(args.out, args.val_fold)
