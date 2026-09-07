"""
Visualization Utilities for Oral Cancer Screening using Vision Transformers.

Generates:
1. Training and Validation Loss & Accuracy curves.
2. Formatted Confusion Matrix plot.
3. Prediction sample visual grid with risk badges.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/server execution
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image

from ml.vit_model import CLASS_NAMES


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: str = "outputs/training_curves.png",
):
    """
    Plot train/val loss and train/val accuracy across training epochs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f172a")

    # Loss plot
    ax1.set_facecolor("#1e293b")
    ax1.plot(epochs, history["train_loss"], color="#38bdf8", label="Train Loss", linewidth=2.2)
    ax1.plot(epochs, history["val_loss"], color="#f43f5e", label="Val Loss", linewidth=2.2, linestyle="--")
    ax1.set_title("Cross-Entropy Loss vs. Epochs", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", color="#94a3b8", fontsize=11)
    ax1.set_ylabel("Loss", color="#94a3b8", fontsize=11)
    ax1.tick_params(colors="#cbd5e1")
    ax1.grid(True, color="#334155", linestyle=":", alpha=0.6)
    ax1.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    # Accuracy plot
    ax2.set_facecolor("#1e293b")
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], color="#34d399", label="Train Accuracy (%)", linewidth=2.2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], color="#fbbf24", label="Val Accuracy (%)", linewidth=2.2, linestyle="--")
    ax2.set_title("Classification Accuracy vs. Epochs", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", color="#94a3b8", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", color="#94a3b8", fontsize=11)
    ax2.tick_params(colors="#cbd5e1")
    ax2.grid(True, color="#334155", linestyle=":", alpha=0.6)
    ax2.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualization] Saved training curves to {output_path}")


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = CLASS_NAMES,
    output_path: str = "outputs/confusion_matrix.png",
):
    """
    Plot and save confusion matrix with raw counts and normalized percentages.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#cbd5e1")

    # Short display labels for matrix ticks
    short_labels = ["Normal", "Low Risk", "High Risk"]
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(short_labels, color="#f8fafc", fontsize=11, fontweight="medium")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(short_labels, color="#f8fafc", fontsize=11, fontweight="medium")

    # Annotate matrix cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_color = "white" if count > thresh else "#0f172a"
            ax.text(
                j, i, f"{count}\n({pct:.1f}%)",
                horizontalalignment="center",
                verticalalignment="center",
                color=text_color,
                fontsize=11,
                fontweight="bold"
            )

    ax.set_title("Confusion Matrix (Test Evaluation)", color="#f8fafc", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Class", color="#94a3b8", fontsize=11, labelpad=10)
    ax.set_ylabel("True Class", color="#94a3b8", fontsize=11, labelpad=10)
    ax.tick_params(colors="#cbd5e1")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualization] Saved confusion matrix to {output_path}")
