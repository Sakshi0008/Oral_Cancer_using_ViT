"""
Dataset Management and Patient-Wise Splitting for Oral Cancer Screening.

Features:
- PyTorch Dataset supporting folder structure and metadata CSV
- Scientifically defensible label mapping from external benchmarks (Mendeley, Kaggle)
- Patient-wise train/val/test splitting to prevent data leakage
- Built-in clinical sample dataset generator for immediate verification & demo runs
"""

import os
import shutil
import random
from typing import List, Tuple, Dict, Optional, Callable
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ml.preprocessing import get_train_transforms, get_inference_transforms
from ml.vit_model import CLASS_NAMES

# Canonical label mapping dictionary
LABEL_MAP: Dict[str, int] = {
    "Normal": 0,
    "Low Risk of Malignant Transformation": 1,
    "High Risk of Malignant Transformation": 2,
    # External dataset aliases mapping
    "normal": 0,
    "healthy": 0,
    "benign": 1,
    "low_risk": 1,
    "lichen_planus": 1,
    "aphthous": 1,
    "cancer": 2,
    "malignant": 2,
    "oscc": 2,
    "high_risk": 2,
    "leukoplakia": 2,
    "erythroplakia": 2,
    "opmd": 2,
}


class OralLesionDataset(Dataset):
    """
    PyTorch Dataset for Oral Lesion Images.
    Supports directory layout or explicit list of (filepath, label, patient_id) tuples.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int, str]],
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            samples: List of (image_path, class_index, patient_id).
            transform: PyTorch transforms to apply.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label, patient_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, patient_id


def load_dataset_from_directory(
    root_dir: str,
    split_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    random_seed: int = 42,
) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]], List[Tuple[str, int, str]]]:
    """
    Scan a directory with class subfolders and perform patient-aware or stratified split.
    Expected folder structure:
        root_dir/
            Normal/
            Low_Risk/ (or Low Risk of Malignant Transformation)
            High_Risk/ (or High Risk of Malignant Transformation)
    """
    random.seed(random_seed)
    train_samples, val_samples, test_samples = [], [], []

    # Map subdirectories to canonical class index
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        normalized_key = folder_name.lower().replace(" ", "_").replace("-", "_")
        target_idx = None
        for key, idx in LABEL_MAP.items():
            if key in normalized_key:
                target_idx = idx
                break

        if target_idx is None:
            # Check for direct class name match
            if folder_name in CLASS_NAMES:
                target_idx = CLASS_NAMES.index(folder_name)
            else:
                continue

        # Collect images in this class
        image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]

        # Group by patient ID if encoded in filename (e.g., "P012_lesion1.jpg")
        patient_groups: Dict[str, List[str]] = {}
        for f in image_files:
            base = os.path.basename(f)
            # Default: treat prefix before '_' as patient_id if present
            pid = base.split("_")[0] if "_" in base else base
            patient_groups.setdefault(pid, []).append(f)

        patients = list(patient_groups.keys())
        random.shuffle(patients)

        n_total = len(patients)
        n_train = max(1, int(n_total * split_ratio[0]))
        n_val = max(1, int(n_total * split_ratio[1]))

        train_pts = set(patients[:n_train])
        val_pts = set(patients[n_train:n_train + n_val])
        test_pts = set(patients[n_train + n_val:])

        for pid, files in patient_groups.items():
            for f in files:
                sample = (f, target_idx, pid)
                if pid in train_pts:
                    train_samples.append(sample)
                elif pid in val_pts:
                    val_samples.append(sample)
                else:
                    test_samples.append(sample)

    return train_samples, val_samples, test_samples


def generate_sample_dataset(base_dir: str = "data/sample", samples_per_class: int = 15) -> str:
    """
    Generate realistic synthetic oral cavity images for testing and demonstration.
    
    Creates:
    - Normal mucosal images (smooth pink mucosal background with fine capillary vascularity)
    - Low-risk lesion images (focal erythematous zone with small aphthous ulceration)
    - High-risk lesion images (irregular leukoplakic plaques, speckled erythroplakia with dysplastic borders)
    
    Each sample receives a synthetic patient ID to validate patient-wise splitting.
    """
    os.makedirs(base_dir, exist_ok=True)
    class_dirs = {
        0: os.path.join(base_dir, "Normal"),
        1: os.path.join(base_dir, "Low_Risk"),
        2: os.path.join(base_dir, "High_Risk"),
    }

    for path in class_dirs.values():
        os.makedirs(path, exist_ok=True)

    np.random.seed(42)

    for cls_idx, folder in class_dirs.items():
        for i in range(samples_per_class):
            pid = f"PAT{cls_idx * 100 + (i // 2):03d}"
            filename = f"{pid}_sample_{i:02d}.png"
            filepath = os.path.join(folder, filename)

            # Skip if already exists
            if os.path.exists(filepath):
                continue

            # Base oral mucosal background (warm pink/mucosa tones)
            w, h = 256, 256
            base_r = int(np.random.normal(205, 12))
            base_g = int(np.random.normal(110, 10))
            base_b = int(np.random.normal(120, 10))

            img_arr = np.zeros((h, w, 3), dtype=np.uint8)
            img_arr[:, :, 0] = np.clip(base_r + np.random.normal(0, 8, (h, w)), 140, 245)
            img_arr[:, :, 1] = np.clip(base_g + np.random.normal(0, 8, (h, w)), 60, 160)
            img_arr[:, :, 2] = np.clip(base_b + np.random.normal(0, 8, (h, w)), 70, 170)

            img = Image.fromarray(img_arr).filter(ImageFilter.GaussianBlur(radius=1.5))
            draw = ImageDraw.Draw(img)

            # Add vascular capillaries / oral mucosal texture
            for _ in range(5):
                start_x, start_y = np.random.randint(20, 230), np.random.randint(20, 230)
                end_x, end_y = start_x + np.random.randint(-30, 30), start_y + np.random.randint(-30, 30)
                draw.line([(start_x, start_y), (end_x, end_y)], fill=(170, 40, 50), width=1)

            cx, cy = np.random.randint(90, 160), np.random.randint(90, 160)

            if cls_idx == 1:
                # Low risk: small, focal, circular aphthous ulcer / benign lichenoid striae
                radius = np.random.randint(18, 30)
                # Outer red inflammatory halo
                draw.ellipse(
                    [(cx - radius - 6, cy - radius - 6), (cx + radius + 6, cy + radius + 6)],
                    fill=(180, 50, 60),
                )
                # Yellowish-white fibrinous pseudomembrane center
                draw.ellipse(
                    [(cx - radius, cy - radius), (cx + radius, cy + radius)],
                    fill=(235, 230, 190),
                )

            elif cls_idx == 2:
                # High risk: irregular speckled leukoplakia / erythroplakic plaque with indurated borders
                for _ in range(4):
                    ox, oy = cx + np.random.randint(-20, 20), cy + np.random.randint(-20, 20)
                    r1, r2 = np.random.randint(25, 45), np.random.randint(25, 45)
                    # Intense dark red velvety erythroplakic base
                    draw.ellipse([(ox - r1, oy - r2), (ox + r1, oy + r2)], fill=(150, 25, 35))
                    # Irregular white keratin plaques
                    draw.polygon(
                        [
                            (ox - 10, oy - 15),
                            (ox + 15, oy - 10),
                            (ox + 20, oy + 12),
                            (ox - 8, oy + 18),
                        ],
                        fill=(240, 238, 235),
                    )

            img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
            img.save(filepath, "PNG")

    print(f"[OralDataset] Sample dataset ready at {base_dir}")
    return base_dir
