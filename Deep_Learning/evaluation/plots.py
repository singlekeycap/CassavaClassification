import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from ..config import short_class_names


def plot_confusion(labels, preds, path, title="Test confusion matrix (row-normalised)"):
    names = short_class_names()
    cm = confusion_matrix(labels, preds, labels=range(len(names)))
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
           xlabel="Predicted", ylabel="True", title=title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: pd.DataFrame, path):
    """Loss and validation metrics per epoch."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(history.epoch, history.train_loss, label="train loss")
    a1.plot(history.epoch, history.val_loss, label="val loss")
    a1.set(xlabel="epoch", title="Loss")
    a1.legend()
    for col, lab in (("val_acc", "accuracy"), ("val_balanced_acc", "balanced acc"), ("val_macro_f1", "macro-F1")):
        a2.plot(history.epoch, history[col], label=lab)
    a2.set(xlabel="epoch", title="Validation metrics")
    a2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
