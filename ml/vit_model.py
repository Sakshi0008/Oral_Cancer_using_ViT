"""
Vision Transformer (ViT) Architecture for Oral Cancer Screening.

Features:
- Pretrained ViT-B/16 backbone from torchvision
- Custom 3-class risk classification head with Dropout (p=0.3)
- Explicit support for Monte Carlo Dropout during inference
- Gradient extraction hooks for transformer-compatible Grad-CAM
- Layer-freezing controls for efficient transfer learning
"""

import os
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights

# Standard 3 Project Risk Categories
CLASS_NAMES: List[str] = [
    "Normal",
    "Low Risk of Malignant Transformation",
    "High Risk of Malignant Transformation",
]

NUM_CLASSES: int = len(CLASS_NAMES)


class OralViTClassifier(nn.Module):
    """
    Fine-tuned Vision Transformer (ViT-B/16) for 3-class Oral Lesion Risk Screening.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super(OralViTClassifier, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Load pretrained ViT-B/16
        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT
            self.vit = models.vit_b_16(weights=weights)
        else:
            self.vit = models.vit_b_16(weights=None)

        # ViT-B/16 feature dimension is 768
        in_features = self.vit.heads.head.in_features  # 768

        # Replace original single linear head with a non-linear head containing Dropout
        # This Dropout layer is pivotal for Monte Carlo Dropout uncertainty estimation
        self.vit.heads = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256, self.num_classes),
        )

        if freeze_backbone:
            self._freeze_early_layers()

    def _freeze_early_layers(self, unfreeze_last_blocks: int = 2):
        """
        Freeze initial patch embedding and early transformer blocks.
        Allows only the top transformer blocks and head to be fine-tuned.
        """
        # Freeze patch embedding and position embeddings
        self.vit.conv_proj.requires_grad_(False)
        self.vit.class_token.requires_grad_(False)
        if hasattr(self.vit, "encoder") and hasattr(self.vit.encoder, "pos_embedding"):
            self.vit.encoder.pos_embedding.requires_grad_(False)

        # Freeze transformer encoder layers except the last n blocks
        encoder_layers = self.vit.encoder.layers
        total_layers = len(encoder_layers)
        for i, layer in enumerate(encoder_layers):
            if i < total_layers - unfreeze_last_blocks:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        return self.vit(x)

    def enable_mc_dropout(self):
        """
        Configure model for Monte Carlo Dropout inference:
        - Keeps overall model in eval mode (so BatchNorm/LayerNorm statistics remain deterministic)
        - Specifically sets all Dropout layers to training mode (so activations are stochastically dropped)
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def get_last_encoder_layer(self) -> nn.Module:
        """
        Returns the final Transformer Encoder Block (encoder.layers[-1]).
        Used to attach gradient and activation hooks for ViT Grad-CAM.
        """
        return self.vit.encoder.layers[-1]


def build_vit_model(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
) -> OralViTClassifier:
    """
    Factory function to initialize the model and optionally load saved checkpoint weights.
    """
    model = OralViTClassifier(
        num_classes=NUM_CLASSES,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            # If checkpoint contains metadata wrapper or raw state dict
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            else:
                model.load_state_dict(state_dict)
            print(f"[OralViT] Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"[OralViT] Warning: Failed to load checkpoint {checkpoint_path}: {e}")
            print("[OralViT] Falling back to pretrained/initialized weights.")

    if device is not None:
        model = model.to(device)

    return model
