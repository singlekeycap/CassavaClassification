"""Imbalance-aware metrics. Macro-F1 is the model-selection metric: always predicting
CMD already gives ~61% accuracy but only ~0.15 macro-F1."""
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score

from ..config import class_names


def compute_metrics(labels, preds) -> dict:
    return {
        "acc": accuracy_score(labels, preds),
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def text_report(labels, preds) -> str:
    return classification_report(labels, preds, target_names=class_names(), digits=4, zero_division=0)
