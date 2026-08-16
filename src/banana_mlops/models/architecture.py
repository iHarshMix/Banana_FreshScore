"""Model architecture definition and TorchScript serialization."""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

from src.banana_mlops.utils.logger import setup_logger

logger = setup_logger("banana_mlops.models.architecture")


class BananaRipenessRegressor(nn.Module):
    """Deep CNN for continuous banana spoilage/ripeness score regression in [0.0, 1.0]."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower().strip()

        if self.backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base_model = resnet18(weights=weights)
            in_features = base_model.fc.in_features
            # Remove original classification head
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 1),
                nn.Sigmoid(),
            )
        elif self.backbone_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = efficientnet_b0(weights=weights)
            in_features = base_model.classifier[1].in_features
            self.backbone = base_model.features
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. Supported: 'resnet18', 'efficientnet_b0'"
            )

        if freeze_backbone:
            self.set_backbone_trainable(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning single continuous score in [0.0, 1.0]."""
        features = self.backbone(x)
        output = self.head(features)
        return output.squeeze(-1)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = trainable
        status = "unfrozen (trainable)" if trainable else "frozen"
        logger.info(f"Backbone layers are now {status}.")


def export_torchscript(
    model: nn.Module,
    save_path: str = "models/production_model.pt",
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu",
) -> Path:
    """Trace PyTorch model with sample input and serialize to optimized TorchScript (.pt).

    Args:
        model: Trained PyTorch model.
        save_path: Path to write serialized .pt file.
        input_shape: Shape of dummy input tensor.
        device: Device on which tracing is conducted ('cpu' for production serving).

    Returns:
        Path to saved TorchScript model.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_shape, device=device)

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(out_file))

    logger.info(f"Successfully exported TorchScript model to {out_file}.")
    return out_file
