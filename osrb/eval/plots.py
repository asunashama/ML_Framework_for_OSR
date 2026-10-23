import os
from typing import Dict, List, Optional
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mltools import calculate_confusion_matrix, plot_confusion_matrix


def plot_open_set_confusion_matrix(
    num_known: int,
    known_targets: np.ndarray,
    known_preds: np.ndarray,
    known_is_accepted: np.ndarray,
    unknown_preds: np.ndarray,
    unknown_is_accepted: np.ndarray,
    save_path: str,
    class_labels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Generate and save a (K + 1) x (K + 1) row-normalized confusion matrix (C1..CK + Unknown).
    Reuses mltools.calculate_confusion_matrix and mltools.plot_confusion_matrix.
    """
    n_classes = num_known + 1
    if class_labels is None or len(class_labels) != n_classes:
        display_labels = [f"C{i + 1}" for i in range(num_known)] + ["Unknown"]
    else:
        display_labels = list(class_labels)

    n_k = len(known_targets)
    n_u = len(unknown_preds)
    total = n_k + n_u

    Y = np.zeros((total, n_classes), dtype=np.float32)
    Y_hat = np.zeros((total, n_classes), dtype=np.float32)

    for i in range(n_k):
        t = int(known_targets[i])
        p = int(known_preds[i]) if bool(known_is_accepted[i]) else num_known
        Y[i, t] = 1.0
        Y_hat[i, p] = 1.0

    for j in range(n_u):
        idx = n_k + j
        p = int(unknown_preds[j]) if bool(unknown_is_accepted[j]) else num_known
        Y[idx, num_known] = 1.0
        Y_hat[idx, p] = 1.0

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    confnorm, _, _ = calculate_confusion_matrix(Y, Y_hat, display_labels)
    plot_confusion_matrix(cm=confnorm, labels=display_labels, save_filename=save_path)
    return confnorm


def plot_training_curves(history: Dict[str, List[float]], save_path: str) -> None:
    """Save training and validation accuracy / loss curves."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    epochs = range(1, len(history.get("val_accuracy", [])) + 1)
    if not epochs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    if "accuracy" in history and "val_accuracy" in history:
        axes[0].plot(epochs, history["accuracy"], label="Train Acc", color="tab:red", linewidth=1.5)
        axes[0].plot(epochs, history["val_accuracy"], label="Val Acc", color="tab:blue", linewidth=1.5)
        axes[0].set_title("Closed-Set Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

    for key in ("loss", "entropy", "regularization", "mse_loss"):
        if key in history and len(history[key]) == len(epochs):
            axes[1].plot(epochs, history[key], label=key, linewidth=1.5)
    axes[1].set_title("Training Loss Components")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    save_path: str,
    threshold: float = 0.0,
    title: str = "Open-Set Confidence Score Distribution",
) -> None:
    """Plot histogram of known vs. unknown confidence scores."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    ax.hist(known_scores, bins=50, alpha=0.6, density=True, label="Known (Test)", color="tab:blue")
    ax.hist(unknown_scores, bins=50, alpha=0.6, density=True, label="Unknown (Test)", color="tab:orange")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.2f})")
    ax.set_title(title)
    ax.set_xlabel("Calibrated Open-Set Score (< 0 Rejected as Unknown)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
