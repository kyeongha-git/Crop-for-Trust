import torch
import torch.nn as nn
import torchvision.models as models
import timm  # pip install timm

class MobileNetClassifier(nn.Module):
    """
    A unified MobileNet-based classifier supporting V1, V2, and V3 architectures.
    
    Ensures FAIR COMPARISON by using ImageNet pre-trained weights for all versions.
    
    Architecture:
        - Backbone: MobileNetV1 (timm), V2/V3 (torchvision)
        - Head: Dropout -> Linear(num_classes)
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
        
        if self.model_type == "mobilenet_v1":
            self.backbone = timm.create_model('mobilenetv1_100', pretrained=True)
            in_features = self.backbone.classifier.in_features

        elif self.model_type == "mobilenet_v2":
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

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        new_head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

        self.backbone.classifier = new_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns raw logits. Use BCEWithLogitsLoss for training.
        """
        return self.backbone(x)