"""
Image Preprocessing Pipeline for Vision Transformer Oral Cancer Screening.

Includes:
- Robust format and dimension validation
- ViT-compliant input transformation (224x224 RGB, ImageNet normalization)
- Training augmentations (rotation, flip, color jitter, affine)
- Deterministic inference preprocessing
- Denormalization helper for visualization
"""

from typing import Tuple, Optional, Union
from PIL import Image, ImageStat
import numpy as np
import torch
from torchvision import transforms

# ViT standard input dimensions & statistics
VIT_INPUT_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageValidationError(Exception):
    """Raised when an uploaded image fails validation criteria."""
    pass


def validate_image(
    image: Union[Image.Image, str],
    min_dimension: int = 64,
    max_dimension: int = 8192
) -> Image.Image:
    """
    Validate oral cavity image format, dimensions, and basic quality.
    
    Args:
        image: PIL Image object or path to image.
        min_dimension: Minimum acceptable width/height in pixels.
        max_dimension: Maximum acceptable width/height in pixels.
        
    Returns:
        Validated RGB PIL Image.
        
    Raises:
        ImageValidationError: If validation checks fail.
    """
    if isinstance(image, str):
        try:
            pil_img = Image.open(image)
        except Exception as e:
            raise ImageValidationError(f"Cannot open image file: {str(e)}")
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        raise ImageValidationError("Expected PIL Image or filepath string.")

    # Check format / dimensions
    width, height = pil_img.size
    if width < min_dimension or height < min_dimension:
        raise ImageValidationError(
            f"Image resolution too low ({width}x{height}). Minimum required: {min_dimension}x{min_dimension}."
        )
    if width > max_dimension or height > max_dimension:
        raise ImageValidationError(
            f"Image resolution exceeds supported maximum ({width}x{height})."
        )

    # Convert to RGB (handles RGBA, Palette, Grayscale)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Basic quality verification: check variance / non-blank
    stat = ImageStat.Stat(pil_img)
    variance = stat.var
    if all(v < 1.0 for v in variance):
        raise ImageValidationError("Image appears blank or has virtually zero visual variance.")

    return pil_img


def get_train_transforms(input_size: Tuple[int, int] = VIT_INPUT_SIZE) -> transforms.Compose:
    """
    Construct data augmentation transforms for training.
    
    Augmentations are clinically safe:
    - Random horizontal and vertical flips
    - Small angle rotations (up to 15 degrees)
    - Moderate color jitter (brightness & contrast variance in photography)
    - Random resized crop to improve lesion location invariance
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(input_size: Tuple[int, int] = VIT_INPUT_SIZE) -> transforms.Compose:
    """
    Construct deterministic transforms for evaluation and inference.
    No stochastic augmentations are applied.
    """
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_for_inference(
    image: Union[Image.Image, str],
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Full inference preprocessing pipeline:
    Validates image, applies deterministic transforms, and adds batch dimension.
    
    Returns:
        tensor: Tensor of shape (1, 3, 224, 224) ready for model forward pass.
        validated_pil: The validated RGB PIL image (for display/visualization).
    """
    validated_pil = validate_image(image)
    transform = get_inference_transforms()
    tensor = transform(validated_pil).unsqueeze(0)  # Shape: (1, 3, 224, 224)
    
    if device is not None:
        tensor = tensor.to(device)
        
    return tensor, validated_pil


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalized PyTorch image tensor (C, H, W) back to numpy RGB image (H, W, C) in [0, 1].
    Useful for overlaying explainability heatmaps.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    np_img = tensor.detach().cpu().numpy().transpose((1, 2, 0))
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    np_img = std * np_img + mean
    return np.clip(np_img, 0.0, 1.0)
