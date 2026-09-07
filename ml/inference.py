"""
Standalone CLI Inference Pipeline for Oral Cancer Screening using Vision Transformers.

Performs:
1. Image validation & preprocessing
2. Monte Carlo Dropout prediction & uncertainty quantification
3. Vision Transformer Grad-CAM explainability heatmap & overlay
4. Controlled clinical screening summary generation

Usage:
    python ml/inference.py --image path/to/image.jpg
"""

import os
import argparse
import json
import torch
from PIL import Image

from ml.vit_model import build_vit_model, CLASS_NAMES
from ml.preprocessing import preprocess_for_inference
from ml.uncertainty import MonteCarloDropoutEstimator
from ml.explainability import generate_explanation_suite
from ml.clinical_explanation import ControlledClinicalExplainer


def run_inference(
    image_path: str,
    checkpoint_path: str = "models/best_vit_model.pth",
    num_mc_passes: int = 15,
    save_overlay_path: str = "outputs/sample_predictions/latest_inference.png",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Utilizing device: {device}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # 1. Preprocessing & Validation
    input_tensor, validated_pil = preprocess_for_inference(image_path, device=device)

    # 2. Build model
    model = build_vit_model(checkpoint_path=checkpoint_path, device=device, pretrained=True)

    # 3. Monte Carlo Dropout Uncertainty Estimation
    mc_estimator = MonteCarloDropoutEstimator(model, num_passes=num_mc_passes)
    uncertainty_results = mc_estimator.estimate(input_tensor, device=device)

    pred_class = uncertainty_results["predicted_class"]
    pred_idx = uncertainty_results["predicted_index"]
    confidence = uncertainty_results["confidence"]
    uncertainty_level = uncertainty_results["uncertainty_level"]

    # 4. ViT Grad-CAM Explainability
    explanation_suite = generate_explanation_suite(
        model=model,
        input_tensor=input_tensor,
        original_pil=validated_pil,
        target_class_idx=pred_idx,
        device=device,
    )

    # 5. Controlled Clinical Summary
    explainer = ControlledClinicalExplainer()
    clinical_summary = explainer.generate_explanation(
        predicted_class=pred_class,
        confidence=confidence,
        uncertainty_level=uncertainty_level,
        primary_region=explanation_suite["primary_region"],
        probabilities=uncertainty_results["probabilities"],
    )

    # Combine into unified response
    report = {
        "status": "success",
        "predicted_risk": pred_class,
        "confidence": confidence,
        "confidence_pct": uncertainty_results["confidence_pct"],
        "uncertainty_score": uncertainty_results["uncertainty_score"],
        "uncertainty_level": uncertainty_level,
        "probabilities": uncertainty_results["probabilities"],
        "primary_visual_region": explanation_suite["primary_region"],
        "clinical_summary": clinical_summary,
        "disclaimer": clinical_summary["disclaimer"],
    }

    # Save output visualization
    os.makedirs(os.path.dirname(save_overlay_path), exist_ok=True)
    # Decode overlay base64 to save file
    import base64
    raw_b64 = explanation_suite["overlay_base64"].split(",")[1]
    with open(save_overlay_path, "wb") as f:
        f.write(base64.b64decode(raw_b64))

    print("\n" + "=" * 55)
    print("      ORAL LESION SCREENING PREDICTION REPORT")
    print("=" * 55)
    print(f"Risk Category     : {pred_class}")
    print(f"Model Confidence  : {uncertainty_results['confidence_pct']}")
    print(f"Uncertainty Level : {uncertainty_level} (Score: {uncertainty_results['uncertainty_score']})")
    print(f"Primary Visual Reg: {explanation_suite['primary_region']}")
    print("-" * 55)
    print("Class Probabilities:")
    for c_name, p_val in uncertainty_results["probabilities"].items():
        print(f"  • {c_name}: {p_val * 100:.1f}%")
    print("-" * 55)
    print("Clinical Recommendation:")
    print(f"  {clinical_summary['clinical_recommendation']}")
    print("=" * 55)
    print(f"[Inference] Saved Grad-CAM overlay to: {save_overlay_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run ViT inference with MC Dropout and Grad-CAM")
    parser.add_argument("--image", type=str, required=True, help="Path to input oral image")
    parser.add_argument("--checkpoint", type=str, default="models/best_vit_model.pth", help="Path to checkpoint")
    parser.add_argument("--passes", type=int, default=15, help="Number of Monte Carlo forward passes")
    parser.add_argument("--output", type=str, default="outputs/sample_predictions/latest_inference.png", help="Overlay output path")
    args = parser.parse_args()

    run_inference(args.image, args.checkpoint, args.passes, args.output)


if __name__ == "__main__":
    main()
