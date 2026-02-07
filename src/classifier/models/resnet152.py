import torch
import torch.nn as nn
import torchvision.models as models


class ResNet152Classifier(nn.Module):
    """
    A custom classifier built on top of a pretrained ResNet-152 backbone.

    Architecture:
        - Backbone: ResNet-152 (ImageNet pretrained)
        - Feature Extractor: frozen by default
        - Head: AdaptiveAvgPool2d → Flatten → Linear(num_classes)
        - Input:  (B, 3, 360, 360)
        - Output: (B, num_classes)

    Args:
        num_classes (int): Number of target classes. Default is 1.
        freeze_backbone (bool): Whether to freeze the backbone weights. Default is True.
    """

    def __init__(self, num_classes: int = 1, freeze_backbone: bool = True):
        super().__init__()

        # Load pretrained ResNet-152 backbone
        self.backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)

        # Remove the default avgpool and fc layers for customization
        in_features = self.backbone.fc.in_features
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # Optionally freeze backbone weights
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Define custom classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(start_dim=1),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward propagation through the ResNet backbone and custom head.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Output logits tensor of shape (B, num_classes).
        """
        # Extract features through ResNet convolutional layers
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)

        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)

        # Pass through classifier head
        logits = self.classifier(features)
        return logits
