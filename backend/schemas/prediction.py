"""
Pydantic Schemas for Oral Cancer Screening API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    device: str
    cuda_available: bool
    model_loaded: bool
    checkpoint_exists: bool
    checkpoint_path: str


class ClinicalExplanationSchema(BaseModel):
    predicted_risk: str
    confidence_display: str
    uncertainty_rating: str
    urgency: str
    finding_summary: str
    visual_explanation: str
    clinical_recommendation: str
    disclaimer: str


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    probabilities: Dict[str, float]
    confidence: float
    confidence_pct: str
    uncertainty_score: float
    uncertainty_level: str
    primary_region: str
    original_image: str
    explanation_heatmap: str
    explanation_overlay: str
    clinical_summary: ClinicalExplanationSchema
    disclaimer: str


class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    backbone: str
    input_resolution: List[int]
    patch_size: int
    num_classes: int
    classes: List[str]
    dropout_rate: float
    explainability_method: str
    uncertainty_method: str
    device: str


class MetricsResponse(BaseModel):
    available: bool
    accuracy: Optional[float] = None
    accuracy_pct: Optional[str] = None
    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    macro_f1: Optional[float] = None
    weighted_f1: Optional[float] = None
    total_test_samples: Optional[int] = None
    per_class_metrics: Optional[Dict[str, Any]] = None
    confusion_matrix_url: Optional[str] = None
    training_curves_url: Optional[str] = None
    message: Optional[str] = None
    disclaimer: str
