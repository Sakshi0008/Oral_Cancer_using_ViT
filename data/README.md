# Oral Lesion Datasets & Label Mapping Guide

This project is built for AI-assisted oral cavity screening using Vision Transformers. Due to licensing and size considerations, large clinical datasets are not directly committed to version control. Instead, this guide details recommended public benchmarks, download links, expected directory layouts, and scientifically defensible label mappings.

---

## 1. Supported Public Datasets

### A. Mendeley Data Oral Cancer Dataset
- **Authors**: Rahman et al.
- **Reference**: *Oral Cancer (Lips and Tongue Images) / Histopathology & Clinical Photos*
- **Source Link**: [Mendeley Data - Oral Cancer Identification](https://data.mendeley.com/datasets/f6bpvrv48v/1)
- **Description**: Clinical images categorized into non-cancerous lesions, precancerous lesions, and oral squamous cell carcinoma (OSCC).

### B. Kaggle Oral Cavity Cancer & OPMD Benchmarks
- **Reference**: Oral Potentially Malignant Disorders (OPMD) and Normal Oral Cavity Image Sets
- **Source Link**: [Kaggle Oral Cancer Screening Dataset](https://www.kaggle.com/datasets/shivam17299/oral-cancer-lips-and-tongue-images)

---

## 2. Scientifically Defensible Label Mapping

The project enforces **exactly three risk classes** to prevent misleading definitive diagnoses:
1. `Normal`
2. `Low Risk of Malignant Transformation`
3. `High Risk of Malignant Transformation`

| Original Dataset Label | Clinical Condition Examples | Project Target Risk Category | Rationale |
| :--- | :--- | :--- | :--- |
| **Normal / Healthy Oral Mucosa** | Healthy buccal mucosa, tongue, gingiva, palate | **Normal** | Baseline normal tissue without inflammatory or dysplastic change. |
| **Benign / Non-Dysplastic Lesions** | Aphthous stomatitis, geographic tongue, mild lichen planus, fibroma | **Low Risk of Malignant Transformation** | Common benign or reactive lesions with minimal transformation probability. |
| **OPMD with Dysplasia / OSCC** | Homogeneous/speckled leukoplakia, erythroplakia, oral submucous fibrosis, suspected carcinoma | **High Risk of Malignant Transformation** | Lesions carrying documented histological transformation risks requiring urgent specialist evaluation. |

---

## 3. Expected Directory Layout

Place any full downloaded dataset in `data/processed/` following this structure:

```
data/
├── README.md
├── sample/                      <-- Pre-packaged sample verification images
│   ├── Normal/
│   ├── Low_Risk/
│   └── High_Risk/
└── processed/                   <-- Full dataset for training
    ├── train/
    │   ├── Normal/
    │   ├── Low_Risk/
    │   └── High_Risk/
    ├── val/
    │   ├── Normal/
    │   ├── Low_Risk/
    │   └── High_Risk/
    └── test/
        ├── Normal/
        ├── Low_Risk/
        └── High_Risk/
```

Patient-wise splitting is strongly advised (70% train, 15% validation, 15% test) to prevent cross-partition patient leakage.
