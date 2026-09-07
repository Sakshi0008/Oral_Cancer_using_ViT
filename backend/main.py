"""
FastAPI Entry Point for Oral Cancer Screening using Vision Transformers.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router

app = FastAPI(
    title="Oral Cancer Screening Decision-Support API",
    description=(
        "Vision Transformer (ViT-B/16) screening service for oral cavity lesion risk classification "
        "with Monte Carlo Dropout uncertainty estimation and transformer-compatible Grad-CAM."
    ),
    version="1.0.0",
)

# Enable CORS for local Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local prototyping
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure outputs directory exists and mount as static file directory
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Include prediction and info routes
app.include_router(router)


@app.get("/")
def root():
    return {
        "title": "Oral Cancer Screening Prototype API",
        "description": "Preliminary oral-lesion risk assessment using Vision Transformers with Explainable AI.",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "model_info": "/model-info",
            "metrics": "/metrics",
            "docs": "/docs",
        },
        "disclaimer": (
            "This system provides AI-assisted preliminary screening and is not a medical diagnosis. "
            "Professional medical evaluation is required for clinical diagnosis."
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
