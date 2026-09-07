"""
Monte Carlo Dropout Uncertainty Estimation for Vision Transformer.

Quantifies epistemic (model) uncertainty by performing multiple stochastic
forward passes with active Dropout during inference.

Statistical Metrics Computed:
- Mean Predictive Probabilities across N passes
- Standard Deviation per class
- Normalized Predictive Entropy (0.0 to 1.0)
- Categorical Uncertainty Level (Low / Moderate / High)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F

from ml.vit_model import CLASS_NAMES, OralViTClassifier


class MonteCarloDropoutEstimator:
    """
    Performs Monte Carlo Dropout sampling on a trained OralViTClassifier.
    """

    def __init__(self, model: OralViTClassifier, num_passes: int = 15):
        """
        Args:
            model: OralViTClassifier instance with dropout in head.
            num_passes: Number of stochastic forward passes (default: 15).
        """
        self.model = model
        self.num_passes = max(2, num_passes)

    @torch.no_grad()
    def estimate(
        self,
        input_tensor: torch.Tensor,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Perform N stochastic forward passes on a single image tensor.
        
        Args:
            input_tensor: Tensor of shape (1, 3, 224, 224).
            device: Computing device.
            
        Returns:
            Dictionary containing:
                - predicted_class: Name of highest-probability class
                - predicted_index: Index (0, 1, or 2)
                - probabilities: Dict mapping each class name to mean probability
                - class_std: Dict mapping class to standard deviation
                - confidence: Mean probability of top class (0.0 to 1.0)
                - confidence_pct: Percentage string (e.g. "86.4%")
                - entropy: Normalized predictive entropy [0, 1]
                - uncertainty_score: Combined uncertainty metric
                - uncertainty_level: "Low", "Moderate", or "High"
                - raw_pass_probabilities: Matrix of shape (N, num_classes)
        """
        if device is not None:
            input_tensor = input_tensor.to(device)
            self.model.to(device)

        # Activate Dropout while keeping LayerNorm in eval mode
        self.model.enable_mc_dropout()

        pass_probs: List[np.ndarray] = []

        for _ in range(self.num_passes):
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pass_probs.append(probs)

        # Matrix of shape (num_passes, num_classes)
        pass_matrix = np.array(pass_probs)

        # 1. Mean probabilities
        mean_probs = np.mean(pass_matrix, axis=0)

        # 2. Standard deviation across stochastic passes
        std_probs = np.std(pass_matrix, axis=0)

        # 3. Top predicted class
        pred_idx = int(np.argmax(mean_probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(mean_probs[pred_idx])

        # 4. Normalized Predictive Entropy: H(p) / log(C)
        num_classes = len(CLASS_NAMES)
        eps = 1e-12
        entropy = -np.sum(mean_probs * np.log(mean_probs + eps))
        max_entropy = np.log(num_classes)
        norm_entropy = float(entropy / max_entropy)
        norm_entropy = np.clip(norm_entropy, 0.0, 1.0)

        # 5. Combined uncertainty measure (weighted blend of top-class std and normalized entropy)
        top_std = float(std_probs[pred_idx])
        uncertainty_score = float(0.5 * norm_entropy + 0.5 * (top_std * 2.0))
        uncertainty_score = round(min(1.0, max(0.0, uncertainty_score)), 4)

        # 6. Categorical rating
        if uncertainty_score < 0.30:
            uncertainty_level = "Low"
        elif uncertainty_score < 0.60:
            uncertainty_level = "Moderate"
        else:
            uncertainty_level = "High"

        # Restore standard evaluation mode
        self.model.eval()

        return {
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "probabilities": {
                name: round(float(mean_probs[i]), 4)
                for i, name in enumerate(CLASS_NAMES)
            },
            "class_std": {
                name: round(float(std_probs[i]), 4)
                for i, name in enumerate(CLASS_NAMES)
            },
            "confidence": round(confidence, 4),
            "confidence_pct": f"{confidence * 100:.1f}%",
            "entropy": round(norm_entropy, 4),
            "uncertainty_score": uncertainty_score,
            "uncertainty_level": uncertainty_level,
            "num_mc_passes": self.num_passes,
            "disclaimer": "Confidence and uncertainty scores reflect model distribution spread and must NOT be construed as medical certainty.",
        }
