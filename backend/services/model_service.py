"""
Model Service managing Vision Transformer lifecycle, MC Dropout, and XAI generation.
"""

import os
import io
from typing import Optional, Dict, Any
from PIL import Image
import torch

from ml.vit_model import build_vit_model, CLASS_NAMES, OralViTClassifier, NUM_CLASSES
from ml.preprocessing import preprocess_for_inference, ImageValidationError
from ml.uncertainty import MonteCarloDropoutEstimator
from ml.explainability import generate_explanation_suite
from ml.clinical_explanation import ControlledClinicalExplainer

CHECKPOINT_PATH = "models/best_vit_model.pth"


class OralScreeningService:
    """
    Service wrapper for inference, explainability, and uncertainty quantification.
    """

    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[OralViTClassifier] = None
        self.explainer = ControlledClinicalExplainer()
        self._load_model()

    def _load_model(self):
        """Load model with checkpoint if available, otherwise with pretrained ViT weights."""
        print(f"[ModelService] Initializing Vision Transformer on device: {self.device}")
        has_checkpoint = os.path.exists(self.checkpoint_path)
        if has_checkpoint:
            print(f"[ModelService] Loading trained checkpoint from: {self.checkpoint_path}")
        else:
            print("[ModelService] Trained checkpoint not found; starting in Pretrained Demo Mode.")

        self.model = build_vit_model(
            checkpoint_path=self.checkpoint_path if has_checkpoint else None,
            device=self.device,
            pretrained=True,
            dropout_rate=0.3,
        )
        self.model.eval()

    def reload_checkpoint(self) -> bool:
        """Reload weights after a fresh training run."""
        self._load_model()
        return os.path.exists(self.checkpoint_path)

    def predict_image(
        self,
        image_bytes: bytes,
        num_mc_passes: int = 15,
        alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """
        End-to-end inference handler for uploaded image bytes.
        """
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ImageValidationError(f"Invalid image file format: {str(e)}")

        # 1. Preprocessing & validation
        input_tensor, validated_pil = preprocess_for_inference(pil_image, device=self.device)

        # 2. Monte Carlo Dropout Uncertainty
        mc_estimator = MonteCarloDropoutEstimator(self.model, num_passes=num_mc_passes)
        uncertainty_res = mc_estimator.estimate(input_tensor, device=self.device)

        pred_class = uncertainty_res["predicted_class"]
        pred_idx = uncertainty_res["predicted_index"]
        confidence = uncertainty_res["confidence"]
        uncertainty_level = uncertainty_res["uncertainty_level"]

        # 3. Vision Transformer Grad-CAM Explainability
        explanation_suite = generate_explanation_suite(
            model=self.model,
            input_tensor=input_tensor,
            original_pil=validated_pil,
            target_class_idx=pred_idx,
            alpha=alpha,
            device=self.device,
        )

        # 4. Controlled Clinical Explanation
        clinical_summary = self.explainer.generate_explanation(
            predicted_class=pred_class,
            confidence=confidence,
            uncertainty_level=uncertainty_level,
            primary_region=explanation_suite["primary_region"],
            probabilities=uncertainty_res["probabilities"],
        )

        return {
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "probabilities": uncertainty_res["probabilities"],
            "confidence": confidence,
            "confidence_pct": uncertainty_res["confidence_pct"],
            "uncertainty_score": uncertainty_res["uncertainty_score"],
            "uncertainty_level": uncertainty_level,
            "primary_region": explanation_suite["primary_region"],
            "original_image": explanation_suite["original_base64"],
            "explanation_heatmap": explanation_suite["heatmap_base64"],
            "explanation_overlay": explanation_suite["overlay_base64"],
            "clinical_summary": clinical_summary,
            "disclaimer": clinical_summary["disclaimer"],
        }


# Global singleton instance
_screening_service_instance: Optional[OralScreeningService] = None


def get_screening_service() -> OralScreeningService:
    global _screening_service_instance
    if _screening_service_instance is None:
        _screening_service_instance = OralScreeningService()
    return _screening_service_instance
