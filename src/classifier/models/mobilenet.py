import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetClassifier(nn.Module):
    """
    A unified MobileNet-based classifier supporting V2 and V3 architectures.

    This model loads a pretrained MobileNet backbone (V2 or V3-Large) and
    replaces the classification head with a lightweight custom layer.

    Architecture:
        - Backbone: MobileNetV2 or MobileNetV3-Large (ImageNet pretrained)
        - Feature Extractor: frozen by default
        - Head: Dropout → Linear(num_classes)
        - Input:  (B, 3, H, W)
        - Output: (B, num_classes)

    Args:
        num_classes (int): Number of output classes. Default is 1.
        dropout_p (float): Dropout probability for regularization. Default is 0.5.
        model_type (str): One of ['mobilenet_v2', 'mobilenet_v3']. Default is 'mobilenet_v3'.
        freeze_backbone (bool): Whether to freeze pretrained feature extractor. Default is True.

    Example:
        >>> model = MobileNetClassifier(num_classes=2, model_type="mobilenet_v3")
    """

    def __init__(
        self,
        num_classes: int = 1,
        dropout_p: float = 0.5,
        model_type: str = "mobilenet_v3",
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.model_type = model_type.lower()

        # Load pretrained backbone
        if self.model_type == "mobilenet_v2":
            self.backbone = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
            )
            in_features = self.backbone.last_channel

        elif self.model_type == "mobilenet_v3":
            self.backbone = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            )
            in_features = self.backbone.classifier[0].in_features

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Optionally freeze pretrained backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MobileNet backbone and custom classification head.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Output logits tensor of shape (B, num_classes).
        """
        return self.backbone(x)
