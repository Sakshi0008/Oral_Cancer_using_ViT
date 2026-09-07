"""
Model Evaluation Script for Vision Transformer Oral Cancer Screening.

Calculates:
- Overall Accuracy
- Macro & Weighted Precision, Recall, F1-Score
- Per-class Classification Metrics
- Confusion Matrix Plot (outputs/confusion_matrix.png)
- Metrics Summary JSON (outputs/metrics.json)
- Sample prediction visualizations (outputs/sample_predictions/)

Academic Note:
These represent computer vision research metrics on academic datasets, NOT clinical validation.
"""

import os
import argparse
import json
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from PIL import Image, ImageDraw, ImageFont

from ml.vit_model import build_vit_model, CLASS_NAMES, NUM_CLASSES
from ml.preprocessing import get_inference_transforms
from ml.dataset import OralLesionDataset, load_dataset_from_directory, generate_sample_dataset
from ml.visualization import plot_confusion_matrix


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> (List[int], List[int], List[np.ndarray], List[str]):
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_probs: List[np.ndarray] = []
    image_paths: List[str] = []

    with torch.no_grad():
        for images, labels, pids in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_probs.extend(probs)

    return y_true, y_pred, y_probs


def save_sample_visual_predictions(
    test_samples: list,
    y_true: List[int],
    y_pred: List[int],
    y_probs: List[np.ndarray],
    output_dir: str = "outputs/sample_predictions",
    max_samples: int = 6,
):
    os.makedirs(output_dir, exist_ok=True)
    count = min(max_samples, len(test_samples))

    for i in range(count):
        img_path, true_idx, pid = test_samples[i]
        pred_idx = y_pred[i]
        prob = y_probs[i][pred_idx]

        orig = Image.open(img_path).convert("RGB").resize((256, 256))
        canvas = Image.new("RGB", (256, 320), color=(15, 23, 42))
        canvas.paste(orig, (0, 0))

        draw = ImageDraw.Draw(canvas)
        is_correct = (true_idx == pred_idx)
        badge_color = (52, 211, 153) if is_correct else (244, 63, 94)

        short_true = ["Normal", "Low Risk", "High Risk"][true_idx]
        short_pred = ["Normal", "Low Risk", "High Risk"][pred_idx]

        draw.text((8, 262), f"True: {short_true}", fill=(203, 213, 225))
        draw.text((8, 280), f"Pred: {short_pred} ({prob * 100:.1f}%)", fill=badge_color)
        draw.text((8, 298), f"Patient: {pid}", fill=(148, 163, 184))

        out_file = os.path.join(output_dir, f"sample_pred_{i:02d}.png")
        canvas.save(out_file)

    print(f"[Evaluation] Saved {count} sample prediction cards to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vision Transformer on Oral Cancer Screening")
    parser.add_argument("--data-dir", type=str, default="data/sample", help="Dataset directory")
    parser.add_argument("--checkpoint", type=str, default="models/best_vit_model.pth", help="Model checkpoint")
    parser.add_argument("--output-json", type=str, default="outputs/metrics.json", help="Metrics JSON output")
    parser.add_argument("--cm-path", type=str, default="outputs/confusion_matrix.png", help="Confusion matrix image")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluation] Utilizing device: {device}")

    # Ensure dataset exists
    if not os.path.exists(args.data_dir) or len(os.listdir(args.data_dir)) == 0:
        args.data_dir = generate_sample_dataset()

    _, _, test_samples = load_dataset_from_directory(args.data_dir)
    if len(test_samples) == 0:
        # If test set is small or empty, use val set for demonstration
        _, test_samples, _ = load_dataset_from_directory(args.data_dir)

    print(f"[Evaluation] Evaluating on {len(test_samples)} test samples...")
    test_dataset = OralLesionDataset(test_samples, transform=get_inference_transforms())
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Load model
    model = build_vit_model(checkpoint_path=args.checkpoint, device=device, pretrained=True)
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    prec_class, rec_class, f1_class, support_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    per_class_metrics = {}
    for idx, name in enumerate(CLASS_NAMES):
        per_class_metrics[name] = {
            "precision": round(float(prec_class[idx]), 4) if idx < len(prec_class) else 0.0,
            "recall": round(float(rec_class[idx]), 4) if idx < len(rec_class) else 0.0,
            "f1_score": round(float(f1_class[idx]), 4) if idx < len(f1_class) else 0.0,
            "support": int(support_class[idx]) if idx < len(support_class) else 0,
        }

    metrics = {
        "accuracy": round(float(acc), 4),
        "accuracy_pct": f"{acc * 100:.2f}%",
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "total_test_samples": len(y_true),
        "per_class_metrics": per_class_metrics,
        "classes": CLASS_NAMES,
        "model_architecture": "Vision Transformer (ViT-B/16)",
        "disclaimer": "Metrics represent research-model evaluation and do NOT denote clinical diagnostic efficacy.",
    }

    print("\n" + "=" * 55)
    print("      RESEARCH EVALUATION PERFORMANCE REPORT")
    print("=" * 55)
    print(f"Overall Accuracy : {metrics['accuracy_pct']}")
    print(f"Macro F1-Score   : {metrics['macro_f1']}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']}")
    print("-" * 55)
    for c_name, c_vals in per_class_metrics.items():
        print(f"[{c_name}]")
        print(f"  Prec: {c_vals['precision']} | Rec: {c_vals['recall']} | F1: {c_vals['f1_score']} (N={c_vals['support']})")
    print("=" * 55)

    # Save metrics JSON
    with open(args.output_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Evaluation] Metrics written to {args.output_json}")

    # Plot Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, output_path=args.cm_path)

    # Save visual sample predictions
    save_sample_visual_predictions(test_samples, y_true, y_pred, y_probs)


if __name__ == "__main__":
    main()
