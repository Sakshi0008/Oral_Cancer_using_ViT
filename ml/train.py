"""
Fine-Tuning Pipeline for Vision Transformer on Oral Lesion Risk Classification.

Usage:
    python ml/train.py --epochs 10 --batch-size 8 --lr 1e-4 --data-dir data/sample
"""

import os
import argparse
import time
import json
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.vit_model import OralViTClassifier, CLASS_NAMES, NUM_CLASSES
from ml.preprocessing import get_train_transforms, get_inference_transforms
from ml.dataset import OralLesionDataset, load_dataset_from_directory, generate_sample_dataset
from ml.visualization import plot_training_curves


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> (float, float):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = total_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> (float, float):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    val_loss = total_loss / max(1, total)
    val_acc = correct / max(1, total)
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Train Vision Transformer for Oral Cancer Screening")
    parser.add_argument("--data-dir", type=str, default="data/sample", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model checkpoint")
    parser.add_argument("--curves-path", type=str, default="outputs/training_curves.png", help="Curves image path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OralViT Training] Utilizing device: {device}")

    # Ensure dataset exists
    if not os.path.exists(args.data_dir) or len(os.listdir(args.data_dir)) == 0:
        print(f"[OralViT Training] {args.data_dir} not found. Generating sample starter dataset...")
        args.data_dir = generate_sample_dataset()

    train_samples, val_samples, test_samples = load_dataset_from_directory(args.data_dir)
    print(f"[OralViT Training] Dataset loaded: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples")

    train_dataset = OralLesionDataset(train_samples, transform=get_train_transforms())
    val_dataset = OralLesionDataset(val_samples, transform=get_inference_transforms())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate model
    print("[OralViT Training] Initializing Vision Transformer (ViT-B/16)...")
    model = OralViTClassifier(num_classes=NUM_CLASSES, pretrained=True, dropout_rate=0.3, freeze_backbone=True)
    model = model.to(device)

    # Class weighting to address clinical imbalance if present
    train_labels = [s[1] for s in train_samples]
    class_counts = [max(1, train_labels.count(c)) for c in range(NUM_CLASSES)]
    total_samples = len(train_labels)
    class_weights = [total_samples / (NUM_CLASSES * count) for count in class_counts]
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(args.output_dir, "best_vit_model.pth")

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print("\n--- Starting Model Fine-Tuning ---")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(round(t_loss, 4))
        history["train_acc"].append(round(t_acc, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["val_acc"].append(round(v_acc, 4))

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc * 100:.1f}% | "
              f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc * 100:.1f}%")

        # Save checkpoint if best validation accuracy achieved
        if v_acc >= best_val_acc:
            best_val_acc = v_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": v_acc,
                "val_loss": v_loss,
                "class_names": CLASS_NAMES,
            }, best_checkpoint_path)

    elapsed = time.time() - start_time
    print(f"\n[OralViT Training] Completed in {elapsed:.1f}s. Best Val Acc: {best_val_acc * 100:.1f}%")
    print(f"[OralViT Training] Checkpoint saved at: {best_checkpoint_path}")

    # Plot and save training curves
    plot_training_curves(history, output_path=args.curves_path)

    # Save history json
    with open("outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
