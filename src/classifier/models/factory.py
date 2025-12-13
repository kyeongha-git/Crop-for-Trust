import torch.nn as nn

from src.classifier.models.mobilenet import MobileNetClassifier
from src.classifier.models.resnet152 import ResNet152Classifier
from src.classifier.models.vgg16 import VGGClassifier


def get_model(
    model_name: str,
    num_classes: int = 2,
    dropout_p: float = 0.5,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Return a classification model instance based on the given name.

    Supports: VGG16, ResNet152, MobileNetV2, MobileNetV3.

    Args:
        model_name (str): Model identifier.
        num_classes (int): Number of output classes.
        dropout_p (float): Dropout probability (if applicable).
        freeze_backbone (bool): Freeze pretrained weights.

    Returns:
        nn.Module: Initialized model.
    """
    model_key = model_name.strip().lower()

    MODEL_MAP = {
        "vgg": ("VGG16", VGGClassifier, {"dropout_p": dropout_p}),
        "vgg16": ("VGG16", VGGClassifier, {"dropout_p": dropout_p}),
        "resnet": (
            "ResNet152",
            ResNet152Classifier,
            {"freeze_backbone": freeze_backbone},
        ),
        "resnet152": (
            "ResNet152",
            ResNet152Classifier,
            {"freeze_backbone": freeze_backbone},
        ),
        "mobilenetv2": (
            "MobileNetV2",
            MobileNetClassifier,
            {
                "dropout_p": dropout_p,
                "model_type": "mobilenet_v2",
                "freeze_backbone": freeze_backbone,
            }
        ),
        "mobilenet_v2": (
            "MobileNetV2",
            MobileNetClassifier,
            {
                "dropout_p": dropout_p,
                "model_type": "mobilenet_v2",
                "freeze_backbone": freeze_backbone,
            },
        ),
        "mobilenetv3": (
            "MobileNetV3",
            MobileNetClassifier,
            {
                "dropout_p": dropout_p,
                "model_type": "mobilenet_v3",
                "freeze_backbone": freeze_backbone,
            },
        ),
        "mobilenet_v3": (
            "MobileNetV3",
            MobileNetClassifier,
            {
                "dropout_p": dropout_p,
                "model_type": "mobilenet_v3",
                "freeze_backbone": freeze_backbone,
            },
        ),
        "mobilenet": (
            "MobileNetV2",
            MobileNetClassifier,
            {
                "dropout_p": dropout_p,
                "model_type": "mobilenet_v2",
                "freeze_backbone": freeze_backbone,
            },
        ),
    }

    if model_key not in MODEL_MAP:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Supported options: {list(MODEL_MAP.keys())}"
        )

    model_label, model_class, extra_kwargs = MODEL_MAP[model_key]
    print(f"Using {model_label} backbone")

    if "resnet" in model_key and "dropout_p" in extra_kwargs:
        extra_kwargs.pop("dropout_p", None)

    return model_class(num_classes=num_classes, **extra_kwargs)
