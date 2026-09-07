"""
Automated Unit and Pipeline Verification Suite for Oral Cancer Screening Prototype.
"""

import unittest
import numpy as np
from PIL import Image
import torch

from ml.vit_model import OralViTClassifier, CLASS_NAMES
from ml.preprocessing import validate_image, preprocess_for_inference, ImageValidationError
from ml.uncertainty import MonteCarloDropoutEstimator
from ml.explainability import ViTGradCAM, generate_explanation_suite
from ml.clinical_explanation import ControlledClinicalExplainer


class TestOralScreeningPipeline(unittest.TestCase):

    def setUp(self):
        # Create synthetic RGB test image (256x256)
        arr = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        self.test_img = Image.fromarray(arr)

        # Initialize base model (pretrained=False for rapid unit testing without downloading weights again)
        self.model = OralViTClassifier(num_classes=3, pretrained=False, dropout_rate=0.3)
        self.model.eval()

    def test_image_validation_valid(self):
        validated = validate_image(self.test_img)
        self.assertEqual(validated.mode, "RGB")
        self.assertEqual(validated.size, (256, 256))

    def test_image_validation_too_small(self):
        tiny = Image.new("RGB", (32, 32), color="pink")
        with self.assertRaises(ImageValidationError):
            validate_image(tiny, min_dimension=64)

    def test_image_preprocessing_shape(self):
        tensor, _ = preprocess_for_inference(self.test_img)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_vit_forward_pass(self):
        dummy_tensor = torch.randn(2, 3, 224, 224)
        logits = self.model(dummy_tensor)
        self.assertEqual(logits.shape, (2, 3))

    def test_mc_dropout_uncertainty(self):
        mc = MonteCarloDropoutEstimator(self.model, num_passes=5)
        dummy_tensor = torch.randn(1, 3, 224, 224)
        res = mc.estimate(dummy_tensor)

        self.assertIn(res["predicted_class"], CLASS_NAMES)
        self.assertIn(res["uncertainty_level"], ["Low", "Moderate", "High"])
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertTrue(0.0 <= res["uncertainty_score"] <= 1.0)

        # Verify probabilities sum close to 1.0
        prob_sum = sum(res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)

    def test_vit_gradcam(self):
        grad_cam = ViTGradCAM(self.model)
        dummy_tensor = torch.randn(1, 3, 224, 224)
        try:
            cam = grad_cam.generate_heatmap(dummy_tensor, target_class_idx=1)
            self.assertEqual(cam.shape, (224, 224))
            self.assertTrue(np.all(cam >= 0.0) and np.all(cam <= 1.0))
        finally:
            grad_cam.remove_hooks()

    def test_clinical_explanation_guardrails(self):
        explainer = ControlledClinicalExplainer()
        explanation = explainer.generate_explanation(
            predicted_class="High Risk of Malignant Transformation",
            confidence=0.88,
            uncertainty_level="Low",
            primary_region="Central mucosal area",
        )

        text = " ".join([
            explanation["finding_summary"],
            explanation["clinical_recommendation"],
            explanation["visual_explanation"],
        ]).lower()

        # Check that forbidden phrases are NOT present
        self.assertNotIn("you have oral cancer", text)
        self.assertNotIn("cancer confirmed", text)
        self.assertNotIn("no cancer detected", text)

        # Check required disclaimer presence
        self.assertIn("disclaimer", explanation)
        self.assertIn("preliminary research decision support", explanation["disclaimer"].lower())


if __name__ == "__main__":
    unittest.main()
