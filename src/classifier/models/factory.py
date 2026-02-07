#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
factory.py

Model factory module for instantiating classification architectures.
Supports dynamic creation of VGG, ResNet, and MobileNet variants with
configurable hyperparameters.
"""

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
    Factory function to instantiate classification models.

    Matches the provided model name string to the corresponding class and
    initializes it with the specified hyperparameters.

    Args:
        model_name (str): Identifier for the model architecture (case-insensitive).
        num_classes (int): Number of output classes.
        dropout_p (float): Dropout probability (used in VGG/MobileNet heads).
        freeze_backbone (bool): If True, freezes pretrained backbone weights.

    Returns:
        nn.Module: The initialized PyTorch model.

    Raises:
        ValueError: If 'model_name' is not found in the supported registry.
    """
    model_key = model_name.strip().lower()

    # Registry mapping: key -> (Display Name, Model Class, Model-specific Args)
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
        "mobilenetv1": (
            "MobileNetV1",
            MobileNetClassifier,
            {
                "dropout_p": dropout_p,
                "model_type": "mobilenet_v1",
                "freeze_backbone": freeze_backbone,
            }
        ),
        "mobilenet_v1": (
            "MobileNetV1",
            MobileNetClassifier,
            {
                "dropout_p": dropout_p,
                "model_type": "mobilenet_v1",
                "freeze_backbone": freeze_backbone,
            }
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

    # Clean up arguments not supported by specific architectures (e.g., ResNet)
    if "resnet" in model_key and "dropout_p" in extra_kwargs:
        extra_kwargs.pop("dropout_p", None)

    return model_class(num_classes=num_classes, **extra_kwargs)