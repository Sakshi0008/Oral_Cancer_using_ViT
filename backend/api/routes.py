"""
API Routes for Oral Cancer Screening Prototype.
"""

import os
import json
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
import torch

from backend.schemas.prediction import (
    HealthResponse,
    PredictionResponse,
    ModelInfoResponse,
    MetricsResponse,
)
from backend.services.model_service import get_screening_service, CHECKPOINT_PATH
from ml.preprocessing import ImageValidationError
from ml.vit_model import CLASS_NAMES

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    service = get_screening_service()
    has_checkpoint = os.path.exists(CHECKPOINT_PATH)
    return HealthResponse(
        status="operational",
        device=str(service.device),
        cuda_available=torch.cuda.is_available(),
        model_loaded=service.model is not None,
        checkpoint_exists=has_checkpoint,
        checkpoint_path=CHECKPOINT_PATH,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    service = get_screening_service()
    return ModelInfoResponse(
        model_name="ViT-B/16 Oral Lesion Risk Classifier",
        architecture="Vision Transformer",
        backbone="torchvision.models.vit_b_16",
        input_resolution=[224, 224, 3],
        patch_size=16,
        num_classes=len(CLASS_NAMES),
        classes=CLASS_NAMES,
        dropout_rate=0.3,
        explainability_method="ViT Grad-CAM (Final LayerNorm Token Gradient Attribution)",
        uncertainty_method="Monte Carlo Dropout (Stochastic Forward Sampling)",
        device=str(service.device),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_oral_image(
    file: UploadFile = File(...),
    mc_passes: int = Query(15, ge=2, le=50, description="Monte Carlo Dropout passes"),
    alpha: float = Query(0.5, ge=0.1, le=0.9, description="Grad-CAM overlay alpha blend"),
):
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: JPEG, PNG, WEBP, BMP.",
        )

    try:
        contents = await file.read()
        service = get_screening_service()
        result = service.predict_image(contents, num_mc_passes=mc_passes, alpha=alpha)
        return result
    except ImageValidationError as ive:
        raise HTTPException(status_code=422, detail=str(ive))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    metrics_path = "outputs/metrics.json"
    cm_path = "outputs/confusion_matrix.png"
    curves_path = "outputs/training_curves.png"

    disclaimer = "Metrics are computed on research evaluation splits and do NOT constitute clinical validation."

    if not os.path.exists(metrics_path):
        return MetricsResponse(
            available=False,
            message="Model evaluation will appear after training. Run 'python ml/train.py' followed by 'python ml/evaluate.py'.",
            disclaimer=disclaimer,
        )

    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)

        return MetricsResponse(
            available=True,
            accuracy=data.get("accuracy"),
            accuracy_pct=data.get("accuracy_pct"),
            macro_precision=data.get("macro_precision"),
            macro_recall=data.get("macro_recall"),
            macro_f1=data.get("macro_f1"),
            weighted_f1=data.get("weighted_f1"),
            total_test_samples=data.get("total_test_samples"),
            per_class_metrics=data.get("per_class_metrics"),
            confusion_matrix_url="/outputs/confusion_matrix.png" if os.path.exists(cm_path) else None,
            training_curves_url="/outputs/training_curves.png" if os.path.exists(curves_path) else None,
            disclaimer=disclaimer,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {str(e)}")


@router.post("/reload-checkpoint")
def reload_checkpoint():
    service = get_screening_service()
    loaded = service.reload_checkpoint()
    return {"reloaded": loaded, "checkpoint_exists": os.path.exists(CHECKPOINT_PATH)}
