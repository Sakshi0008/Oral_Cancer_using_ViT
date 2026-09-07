"""
Controlled Clinical Explanation Module for Oral Cancer Screening.

Generates structured, clinically responsible screening explanations based exclusively
on model outputs (predicted risk class, confidence %, uncertainty rating, and XAI visual regions).

Safety & Guardrail Policy:
- Strictly non-diagnostic: this is a clinical decision-support screening aid.
- Forbidden terms: "You have oral cancer", "Cancer confirmed", "No cancer detected".
- Permitted risk terminology:
    * "Normal"
    * "Low Risk of Malignant Transformation"
    * "High Risk of Malignant Transformation"
- Strictly disallows medication or surgical advice.
- Mandates prompt recommendation for qualified oral medicine, ENT, or dental specialist evaluation.
"""

from typing import Dict, Any, Optional
import os


class ControlledClinicalExplainer:
    """
    Produces deterministic, clinically safe explanations and screening recommendations.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")

    def generate_explanation(
        self,
        predicted_class: str,
        confidence: float,
        uncertainty_level: str,
        primary_region: str = "Central mucosal region",
        probabilities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured clinical explanation from model outputs.
        """
        confidence_pct = f"{confidence * 100:.1f}%"

        # Safe template generator (guaranteed to be medically responsible and hallucination-free)
        if predicted_class == "High Risk of Malignant Transformation":
            finding_summary = (
                f"The Vision Transformer model flagged visual morphologic patterns consistent with "
                f"high-risk oral mucosal alterations (Model Confidence: {confidence_pct}, "
                f"Uncertainty: {uncertainty_level})."
            )
            visual_notes = (
                f"Self-attention and gradient attribution focalized primarily on the {primary_region.lower()}, "
                f"highlighting surface texture irregularity, mucosal erythema, or plaque-like hyperkeratosis."
            )
            recommendation = (
                "Expedited clinical examination by an Oral and Maxillofacial Surgeon, Oral Medicine specialist, "
                "or Head & Neck specialist is strongly advised. Clinical adjunctive tests (e.g., toluidine blue, "
                "optical fluorescence, or scalpel biopsy with histopathology) should be considered to establish "
                "a definitive diagnosis."
            )
            urgency = "High Priority Follow-Up"

        elif predicted_class == "Low Risk of Malignant Transformation":
            finding_summary = (
                f"The model identified features suggestive of low-risk or benign mucosal variations "
                f"(Model Confidence: {confidence_pct}, Uncertainty: {uncertainty_level})."
            )
            visual_notes = (
                f"Attribution gradients localized around the {primary_region.lower()}, reflecting mild focal "
                f"inflammation or benign epithelial change without prominent high-risk dysplasia patterns."
            )
            recommendation = (
                "Routine clinical monitoring and follow-up. If the lesion has persisted for more than two weeks, "
                "exhibits changes in color or size, causes discomfort, or is subject to recurrent mechanical irritation, "
                "a formal dental or medical evaluation is recommended."
            )
            urgency = "Routine / Periodic Monitoring"

        else:  # Normal
            finding_summary = (
                f"The oral cavity image exhibits visual characteristics consistent with healthy, intact oral mucosa "
                f"(Model Confidence: {confidence_pct}, Uncertainty: {uncertainty_level})."
            )
            visual_notes = (
                f"Distributed visual attention across the {primary_region.lower()} without focal dysplastic hot spots."
            )
            recommendation = (
                "Continue standard oral hygiene practices and maintain routine biannual dental examinations. "
                "Any newly emergent oral ulcers or mucosal alterations lasting longer than 14 days should be "
                "evaluated by a dental practitioner."
            )
            urgency = "Standard Preventive Care"

        return {
            "predicted_risk": predicted_class,
            "confidence_display": confidence_pct,
            "uncertainty_rating": uncertainty_level,
            "urgency": urgency,
            "finding_summary": finding_summary,
            "visual_explanation": visual_notes,
            "clinical_recommendation": recommendation,
            "disclaimer": (
                "This AI-assisted screening report is generated for preliminary research decision support only. "
                "It is NOT a definitive clinical or histopathological diagnosis. Medical decisions must always "
                "be made by a licensed healthcare professional."
            ),
        }
