# Oral Cancer Screening Using Vision Transformers with Explainable AI

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2BCUDA-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React + Vite](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB.svg)](https://vitejs.dev/)
[![License: Academic Research](https://img.shields.io/badge/License-Academic%20Research-green.svg)]()

> **B.Tech CSE (AIML) Major Project • Healthcare Decision-Support Prototype**  
> **Author:** Sakshi Aswale  
> **Notice:** This system provides AI-assisted preliminary screening decision-support and is **NOT** a clinical diagnosis.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Aim](#3-aim)
4. [Objectives](#4-objectives)
5. [System Architecture](#5-system-architecture)
6. [Technology Stack](#6-technology-stack)
7. [Dataset](#7-dataset)
8. [Dataset Sources](#8-dataset-sources)
9. [Dataset Label Mapping](#9-dataset-label-mapping)
10. [Installation & Setup](#10-installation--setup)
11. [Training Instructions](#11-training-instructions)
12. [Evaluation Instructions](#12-evaluation-instructions)
13. [Backend API Instructions](#13-backend-api-instructions)
14. [Frontend Application Instructions](#14-frontend-application-instructions)
15. [Vision Transformer Architecture](#15-vision-transformer-architecture)
16. [Explainable AI (ViT Grad-CAM)](#16-explainable-ai-vit-grad-cam)
17. [Uncertainty Quantification (MC Dropout)](#17-uncertainty-quantification-mc-dropout)
18. [Controlled Clinical Explanation Module](#18-controlled-clinical-explanation-module)
19. [Visual Interface & Result Dashboard](#19-visual-interface--result-dashboard)
20. [Limitations](#20-limitations)
21. [Ethical Considerations](#21-ethical-considerations)
22. [Medical Disclaimer](#22-medical-disclaimer)
23. [Future Work](#23-future-work)

---

## 1. Project Overview

Oral cancer, primarily Oral Squamous Cell Carcinoma (OSCC), represents one of the most prevalent head and neck malignancies worldwide. Late-stage diagnosis remains the leading contributor to poor 5-year survival rates (<50%), whereas early detection of Oral Potentially Malignant Disorders (OPMDs) increases survival outcomes significantly. 

This project presents an academic research prototype for **preliminary oral cavity lesion risk screening**. It leverages a pretrained **Vision Transformer (ViT-B/16)** fine-tuned via transfer learning, coupled with **Monte Carlo Dropout** for epistemic uncertainty quantification, and **Transformer-compatible Grad-CAM** for visual attention attribution. The end-to-end prototype includes a high-performance **FastAPI** backend and an interactive **React + Vite** clinical dashboard.

---

## 2. Problem Statement

Conventional oral screening in outpatient settings relies heavily on visual inspection and palpation by clinicians, which suffers from inter-observer variability, subtle early-stage dysplastic presentation, and limited specialist availability in resource-constrained rural clinics. Furthermore, contemporary deep learning prototypes often exhibit two fatal flaws:
1. **Black-box predictions**: Clinicians cannot verify which mucosal regions influenced the classification.
2. **Overconfident erroneous predictions**: Conventional models generate high softmax confidences even on out-of-distribution or ambiguous lesions.

There is a critical need for an interpretable, uncertainty-aware screening system that categorizes lesions into scientifically defensible risk tiers rather than claiming definitive automated cancer diagnosis.

---

## 3. Aim

To design, develop, and evaluate an explainable, uncertainty-aware computer vision prototype utilizing Vision Transformers for preliminary risk stratification of oral cavity photographic images.

---

## 4. Objectives

1. Implement a robust **image preprocessing and quality validation pipeline** conforming to ViT specifications ($224 \times 224$ RGB, ImageNet normalization).
2. Adapt a pretrained **Vision Transformer (ViT-B/16)** backbone with a custom 3-class classification head incorporating Dropout ($p=0.3$).
3. Fine-tune the architecture using transfer learning and patient-wise data splits (70% train, 15% val, 15% test).
4. Implement **Monte Carlo Dropout** to compute predictive distributions, epistemic variance, and normalized entropy over $N=15$ stochastic forward passes.
5. Formulate a **ViT-compatible Grad-CAM** attribution module by backpropagating target-class gradients to the final Transformer Encoder LayerNorm tokens.
6. Design a **Controlled Clinical Explanation Module** with strict screening guardrails that contextualizes model predictions without making independent diagnoses.
7. Deploy a **FastAPI** REST service and a **React + Vite** dashboard with real-time prediction, interactive heatmap opacity blending, and real test evaluation metrics.

---

## 5. System Architecture

```
                                [ Oral Image Upload ]
                                         │
                                         ▼
                            [ Preprocessing & Validation ]
                            • Dimension & format checks
                            • 224x224 RGB normalization
                                         │
                                         ▼
                             [ Vision Transformer (ViT) ]
                           (12 Encoder Blocks, 768 Hidden Dim)
                                         │
                     ┌───────────────────┴───────────────────┐
                     ▼                                       ▼
        [ Monte Carlo Dropout ]                  [ ViT-Compatible Grad-CAM ]
        • N=15 Stochastic Passes                 • Final LayerNorm token gradients
        • Mean Consensus Probabilities           • Reshape 196 tokens -> 14x14 grid
        • Epistemic Variance & Entropy           • Bilinear upsampling to 224x224
                     │                                       │
                     └───────────────────┬───────────────────┘
                                         ▼
                             [ 3-Class Risk Prediction ]
                             • Normal
                             • Low Risk of Malignant Transformation
                             • High Risk of Malignant Transformation
                                         │
                                         ▼
                        [ Controlled Clinical Explanation ]
                        • Deterministic safe template rules
                        • Morphological focal note & urgency
                        • Mandatory screening disclaimer
                                         │
                     ┌───────────────────┴───────────────────┐
                     ▼                                       ▼
            [ FastAPI REST API ]                   [ React + Vite Dashboard ]
            • /predict, /explain                   • Interactive Opacity Slider
            • /metrics, /health                    • Uncertainty Gauges & Plots
```

---

## 6. Technology Stack

- **Machine Learning & Deep Learning**: Python 3.11, PyTorch 2.6 (with CUDA 12.4 acceleration), torchvision, NumPy, Pandas, scikit-learn.
- **Computer Vision & Image Processing**: OpenCV (cv2), Pillow (PIL), Matplotlib.
- **API Backend**: FastAPI, Uvicorn, Pydantic, Python-Multipart.
- **Frontend Dashboard**: React 19, Vite, Lucide-React, Vanilla CSS Design System with custom tokens.

---

## 7. Dataset

The system supports standard oral cavity clinical photographic benchmarks. Images encompass healthy oral mucosa, benign/reactive oral lesions (aphthous ulcers, oral lichen planus, fibromas), oral potentially malignant disorders (leukoplakia, erythroplakia, oral submucous fibrosis), and suspected oral squamous cell carcinoma (OSCC).

---

## 8. Dataset Sources

1. **Mendeley Data Oral Cancer Dataset**:
   - Reference: Rahman et al., *Oral Cancer Identification from Photographic Images*
   - Official Source: [https://data.mendeley.com/datasets/f6bpvrv48v/1](https://data.mendeley.com/datasets/f6bpvrv48v/1)
2. **Kaggle Oral Cancer (Lips and Tongue) Dataset**:
   - Official Source: [https://www.kaggle.com/datasets/shivam17299/oral-cancer-lips-and-tongue-images](https://www.kaggle.com/datasets/shivam17299/oral-cancer-lips-and-tongue-images)

A verified starter synthetic clinical dataset is automatically generated in `data/sample/` for immediate pipeline testing and demonstration.

---

## 9. Dataset Label Mapping

To ensure clinical defensibility, external labels are strictly mapped into the three project categories:

| Original Dataset Label | Clinical Condition Examples | Project Risk Category | Rationale |
| :--- | :--- | :--- | :--- |
| **Normal / Healthy** | Intact oral mucosa, gingiva, tongue | **Normal** | Baseline intact oral epithelium. |
| **Benign / Reactive** | Aphthous stomatitis, geographic tongue, fibroma, mild lichen planus | **Low Risk of Malignant Transformation** | Common benign mucosal variations with low statistical transformation risk. |
| **OPMD / Dysplasia / OSCC** | Homogeneous/speckled leukoplakia, erythroplakia, suspected OSCC | **High Risk of Malignant Transformation** | Documented histological dysplastic risks requiring expedited specialist biopsy. |

---

## 10. Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js v18+ and npm
- (Optional) NVIDIA GPU with CUDA drivers for acceleration

### Step 1: Clone and Navigate
```bash
cd "c:\Users\Sakshi Aswale\OneDrive\Desktop\Major pro"
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## 11. Training Instructions

Fine-tune the Vision Transformer on the oral lesion dataset:

```bash
# Fine-tune ViT model (defaults to data/sample if external dataset is not yet present)
python ml/train.py --epochs 8 --batch-size 8 --lr 1e-4 --data-dir data/sample
```

**Training Process:**
- Freezes early ViT blocks and trains the custom classification head and top transformer layers.
- Automatically calculates class-weighting penalties for imbalance.
- Checkpoints the best model based on validation performance to `models/best_vit_model.pth`.
- Generates loss and accuracy curves at `outputs/training_curves.png`.

---

## 12. Evaluation Instructions

Evaluate the trained checkpoint on the held-out test split:

```bash
python ml/evaluate.py --checkpoint models/best_vit_model.pth --data-dir data/sample
```

**Evaluation Outputs:**
- Overall Accuracy, Macro F1, Weighted F1, and per-class metrics.
- Saves machine-readable metrics to `outputs/metrics.json`.
- Plots and saves the test split confusion matrix to `outputs/confusion_matrix.png`.
- Generates visual sample prediction cards in `outputs/sample_predictions/`.

---

## 13. Backend API Instructions

Start the FastAPI backend service:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

- API Documentation (Swagger UI): `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`
- Model Info: `http://localhost:8000/model-info`
- Real Metrics: `http://localhost:8000/metrics`

---

## 14. Frontend Application Instructions

Start the Vite development server:

```bash
cd frontend
npm run dev
```

Open your browser at `http://localhost:5173`.

---

## 15. Vision Transformer Architecture

- **Backbone**: Vision Transformer (`vit_b_16`, torchvision).
- **Patch Resolution**: $16 \times 16$ pixels ($14 \times 14 = 196$ spatial patches + 1 `[CLS]` token = 197 tokens).
- **Embedding Dimension**: $D = 768$.
- **Transformer Encoder**: 12 Multi-Head Self-Attention layers with 12 heads each.
- **Classification Head**:
  $$\text{Head}(x) = \text{Linear}(256, 3) \circ \text{Dropout}(p=0.3) \circ \text{ReLU} \circ \text{Linear}(768, 256)$$
  The insertion of `nn.Dropout(p=0.3)` inside the classification head is essential for Monte Carlo Dropout uncertainty sampling.

---

## 16. Explainable AI (ViT Grad-CAM)

Traditional Grad-CAM relies on 2D convolutional feature maps. In Vision Transformers, feature representations exist as 1D sequence tokens.

Our ViT-compatible Grad-CAM implementation:
1. Registers forward and backward hooks on the final Transformer Encoder block (`vit.encoder.layers[-1].ln_1`).
2. Computes the gradient $\frac{\partial y_c}{\partial A}$ of the target risk score $y_c$ with respect to the sequence activation tokens $A \in \mathbb{R}^{197 \times 768}$.
3. Discards the non-spatial `[CLS]` token at index 0, retaining the 196 patch tokens.
4. Performs channel-wise global average pooling of gradients to obtain token importance weights $\alpha$.
5. Applies $\text{ReLU}$ to isolate features with positive attribution toward class $c$.
6. Reshapes the 196 tokens into a $14 \times 14$ spatial grid.
7. Applies bilinear interpolation upsampling from $14 \times 14 \to 224 \times 224$.
8. Generates both a Jet/Turbo colormap heatmap and an alpha-blended overlay with an interactive opacity slider in the frontend.

**UI Caption:**
> "Highlighted regions indicate image areas that influenced the model prediction. They do not represent a medical diagnosis."

---

## 17. Uncertainty Quantification (MC Dropout)

Standard softmax point predictions often produce overconfident probabilities on unrepresentative or noisy clinical images.

**Monte Carlo Dropout Formulation:**
1. During inference, dropout layers remain active while LayerNorm remains in evaluation mode (`model.enable_mc_dropout()`).
2. The model executes $N=15$ stochastic forward passes for a single oral image:
   $$\bar{p}_c = \frac{1}{N} \sum_{i=1}^N p_{i, c}$$
3. Epistemic variance is measured via class standard deviations $\sigma_c$.
4. Predictive uncertainty is summarized using normalized Shannon Entropy:
   $$H(p) = -\frac{1}{\ln(3)} \sum_{c=1}^3 \bar{p}_c \ln(\bar{p}_c + \epsilon)$$
5. The result is categorized into **Low**, **Moderate**, or **High** uncertainty tiers.

---

## 18. Controlled Clinical Explanation Module

Predictions are routed through a strictly controlled explanation layer that maps structured model outputs into medical decision-support text.

**Clinical Safety Rules:**
- Rejects definitive words: "You have oral cancer", "Cancer confirmed", "No cancer detected".
- Uses sanctioned terminology: "High Risk of Malignant Transformation", "Low Risk of Malignant Transformation", "Normal".
- Recommends clinical examination by an Oral & Maxillofacial Surgeon, Oral Medicine specialist, or ENT specialist when dysplastic patterns are flagged.
- Recommends biopsy/histopathology as the sole definitive diagnostic standard.

---

## 19. Visual Interface & Result Dashboard

The web interface is structured into three unified views:
- **Screening**: Image drag-and-drop, quick-load clinical demo samples, inference settings (MC passes, opacity), risk badge, probability breakdown bars, confidence/uncertainty meters, and side-by-side ViT Grad-CAM comparison with a live opacity slider.
- **Model Dashboard**: Live quantitative metrics (Accuracy, Macro F1, Weighted F1), per-class precision/recall/F1 table, test confusion matrix viewer, and training curves viewer.
- **ViT Architecture**: Interactive architectural diagrams, mathematical formulations, and XAI methodology.

---

## 20. Limitations

1. **Dataset Representation**: Training photographic sets may not cover every skin phototype, illumination variation, or rare oral mucosal disorder.
2. **2D Surface Limitation**: Photographs capture surface mucosal coloration and texture but cannot evaluate submucosal induration or depth of invasion.
3. **Biopsy Requirement**: Image classification cannot replace scalpel biopsy, tissue histology, or immunohistochemistry.

---

## 21. Ethical Considerations

- Patient data privacy must be respected: datasets must be fully de-identified and anonymized.
- The prototype is designed as an assistive clinical triage aid, never an autonomous decision-maker.
- Healthcare providers retain complete diagnostic authority.

---

## 22. Medical Disclaimer

> **IMPORTANT MEDICAL NOTICE:**  
> This system is developed strictly for academic research and preliminary screening decision support. It does **NOT** provide a medical diagnosis and should never replace physical examination, biopsy, or histopathological evaluation by a qualified healthcare professional.

---

## 23. Future Work

1. Integration of multimodal inputs (patient tobacco/areca nut history, clinical symptoms).
2. Exploration of Vision-Language Models (e.g. BiomedCLIP, LLaVA-Med) for rich clinical dialog generation.
3. Multi-center clinical validation across diverse oral oncology clinics.
