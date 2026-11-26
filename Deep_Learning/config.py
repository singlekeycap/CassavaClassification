"""Shared paths and constants."""
import json
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
DATASET_DIR = ROOT / "dataset"
IMAGE_DIR = DATASET_DIR / "train_images"
TRAIN_CSV = DATASET_DIR / "train.csv"
SPLITS_CSV = ROOT / "splits" / "splits.csv"
RUNS_DIR = PACKAGE_DIR / "runs"

NUM_CLASSES = 5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def class_names() -> list[str]:
    mapping = json.loads((DATASET_DIR / "label_num_to_disease_map.json").read_text())
    return [mapping[str(i)] for i in range(len(mapping))]


def short_class_names() -> list[str]:
    """CBB, CBSD, CGM, CMD, Healthy."""
    return [n.split("(")[-1].rstrip(")") for n in class_names()]
