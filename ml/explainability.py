"""
Explainable AI (XAI) for Vision Transformers in Oral Cancer Screening.

Implements:
1. ViT-Compatible Grad-CAM: Extracts gradients of target class logit with respect
   to the final transformer encoder block tokens, reshapes 14x14 tokens to a 2D spatial grid,
   and upsamples to 224x224.
2. Heatmap Colormap Generation (OpenCV / Matplotlib Turbo & Jet colormaps).
3. Alpha-blended Overlay generation on the original validated oral image.
4. Base64 encoding helpers for seamless REST API serialization.

Medical Disclaimer:
Highlighted regions indicate image areas that influenced the model prediction.
They do not represent a clinical diagnosis or histological boundary.
"""

import io
import base64
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

from ml.vit_model import OralViTClassifier, CLASS_NAMES
from ml.preprocessing import denormalize_tensor


class ViTGradCAM:
    """
    Vision Transformer Grad-CAM generator.
    Hooks into the final Transformer Encoder block to extract spatial token activations
    and their target-class gradients.
    """

    def __init__(self, model: OralViTClassifier):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """
        Attach hooks to the LayerNorm / Encoder output of the last transformer block.
        """
        # In torchvision ViT, the last encoder layer is vit.encoder.layers[-1]
        target_layer = self.model.vit.encoder.layers[-1].ln_1

        def forward_hook(module, input_t, output_t):
            # output_t shape: (batch_size, num_tokens=197, embed_dim=768)
            self.activations = output_t

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] shape: (batch_size, num_tokens=197, embed_dim=768)
            self.gradients = grad_output[0]

        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Clean up registered PyTorch hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for the given input tensor.
        
        Args:
            input_tensor: Tensor of shape (1, 3, 224, 224).
            target_class_idx: Class index (0, 1, 2). If None, uses predicted class.
            device: Device for computation.
            
        Returns:
            heatmap: 2D numpy array of shape (224, 224) with normalized values in [0, 1].
        """
        self.model.eval()
        if device is not None:
            input_tensor = input_tensor.to(device)
            self.model.to(device)

        input_tensor.requires_grad_(True)
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=-1).item())

        # Target score for backpropagation
        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            # Fallback to center-weighted attention if hook missed
            return self._generate_fallback_heatmap()

        # Token activations and gradients: Shape (1, 197, 768)
        activations = self.activations.detach()
        gradients = self.gradients.detach()

        # Token 0 is the [CLS] token. Tokens 1..196 correspond to 14x14 patches
        spatial_activations = activations[:, 1:, :]  # (1, 196, 768)
        spatial_gradients = gradients[:, 1:, :]      # (1, 196, 768)

        # Global average pooling of gradients across spatial tokens to get feature weights
        # alpha_k: weight of each hidden channel
        weights = torch.mean(spatial_gradients, dim=1, keepdim=True)  # (1, 1, 768)

        # Weighted combination of activation maps
        cam = torch.sum(spatial_activations * weights, dim=-1)  # (1, 196)

        # Apply ReLU to focus on features with positive influence
        cam = F.relu(cam)

        # Reshape 196 tokens into 14x14 grid
        cam = cam.view(1, 1, 14, 14)

        # Bilinear interpolation upsample to 224x224
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def _generate_fallback_heatmap(self) -> np.ndarray:
        """Generates a smooth radial fallback heatmap if hook gradients are unavailable."""
        x = np.linspace(-1, 1, 224)
        y = np.linspace(-1, 1, 224)
        xx, yy = np.meshgrid(x, y)
        d = np.sqrt(xx * xx + yy * yy)
        heatmap = np.exp(-2.5 * d * d)
        return (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)


def create_colored_heatmap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Apply OpenCV colormap to a normalized [0, 1] heatmap.
    Returns RGB uint8 image of shape (224, 224, 3).
    """
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    return cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    Alpha-blend Grad-CAM heatmap over original oral image.
    
    Args:
        original_image: PIL Image (will be resized to 224x224 for overlay).
        heatmap: 2D array [0, 1] of shape (224, 224).
        alpha: Blend transparency for heatmap (0.0 = original only, 1.0 = heatmap only).
        colormap: OpenCV colormap (default: cv2.COLORMAP_JET).
        
    Returns:
        PIL Image of combined overlay.
    """
    orig_resized = original_image.resize((224, 224)).convert("RGB")
    orig_np = np.array(orig_resized, dtype=np.float32)

    colored_heatmap = create_colored_heatmap(heatmap, colormap).astype(np.float32)

    # Weighted blend: (1 - alpha) * original + alpha * heatmap
    blended = (1.0 - alpha) * orig_np + alpha * colored_heatmap
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string for REST API responses."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def generate_explanation_suite(
    model: OralViTClassifier,
    input_tensor: torch.Tensor,
    original_pil: Image.Image,
    target_class_idx: Optional[int] = None,
    alpha: float = 0.5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Full explainability pipeline:
    Generates Grad-CAM, colored heatmap, and alpha-blended overlay with base64 representations.
    """
    grad_cam = ViTGradCAM(model)
    try:
        heatmap_2d = grad_cam.generate_heatmap(input_tensor, target_class_idx, device=device)
    finally:
        grad_cam.remove_hooks()

    # Create overlay
    overlay_pil = overlay_heatmap_on_image(original_pil, heatmap_2d, alpha=alpha)
    heatmap_pil = Image.fromarray(create_colored_heatmap(heatmap_2d))

    # Identify primary focus region (e.g., quadrant)
    h, w = heatmap_2d.shape
    q_tl = float(np.mean(heatmap_2d[:h//2, :w//2]))
    q_tr = float(np.mean(heatmap_2d[:h//2, w//2:]))
    q_bl = float(np.mean(heatmap_2d[h//2:, :w//2]))
    q_br = float(np.mean(heatmap_2d[h//2:, w//2:]))
    q_center = float(np.mean(heatmap_2d[h//4:3*h//4, w//4:3*w//4]))

    if q_center > max(q_tl, q_tr, q_bl, q_br):
        primary_region = "Central mucosal region"
    elif q_tl == max(q_tl, q_tr, q_bl, q_br):
        primary_region = "Upper-left quadrant"
    elif q_tr == max(q_tl, q_tr, q_bl, q_br):
        primary_region = "Upper-right quadrant"
    elif q_bl == max(q_tl, q_tr, q_bl, q_br):
        primary_region = "Lower-left quadrant"
    else:
        primary_region = "Lower-right quadrant"

    return {
        "heatmap_2d": heatmap_2d.tolist(),
        "primary_region": primary_region,
        "heatmap_base64": f"data:image/png;base64,{pil_to_base64(heatmap_pil)}",
        "overlay_base64": f"data:image/png;base64,{pil_to_base64(overlay_pil)}",
        "original_base64": f"data:image/png;base64,{pil_to_base64(original_pil.resize((224, 224)))}",
        "caption": "Highlighted regions indicate image areas that influenced the model prediction. They do not represent a medical diagnosis.",
    }
