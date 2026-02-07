import torch
import torch.nn as nn
import torchvision.models as models


class VGGClassifier(nn.Module):
    """
    A lightweight classifier built on top of a pretrained VGG16 backbone.

    Architecture:
        - Backbone: VGG16 (ImageNet pretrained, frozen)
        - Input: (3, 360, 360)
        - Flattened feature size: 512 × 11 × 11
        - Head: Linear(512*11*11 → 256) → Dropout → Linear(256 → num_classes)

    Args:
        num_classes (int): Number of output classes. Default is 1.
        dropout_p (float): Dropout probability for regularization. Default is 0.5.
    """

    def __init__(self, num_classes: int = 1, dropout_p: float = 0.5):
        super().__init__()

        # Load pretrained VGG16 backbone (ImageNet)
        self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Freeze all convolutional layers to retain pretrained feature extraction
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Define classifier head
        input_dim_flatten = 512 * 11 * 11
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim_flatten, 256),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the frozen VGG16 backbone and the custom classifier.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, 360, 360).

        Returns:
            torch.Tensor: Output logits of shape (B, num_classes).
        """
        # Extract convolutional features
        features = self.backbone.features(x)

        # Pass through custom classification layers
        logits = self.classifier(features)
        return logits
