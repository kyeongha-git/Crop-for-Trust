#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_preprocessing.py
---------------------
Centralized preprocessing manager for image classification models.

This module provides model-specific preprocessing and augmentation pipelines.
It standardizes image resizing, normalization, and optional augmentation
based on the target architecture.

Supported models:
- VGG
- ResNet
- MobileNet (v2, v3)
"""

from typing import Dict, Tuple

from torchvision import transforms


class DataPreprocessor:
    """
    Handles preprocessing and data augmentation for image classification models.

    Provides normalization settings and transformation pipelines tailored
    to each model type, ensuring consistency across training and evaluation.

    Example:
        preprocessor = DataPreprocessor(img_size=(360, 360))
        train_tf = preprocessor.get_transform("resnet", mode="train")
        eval_tf = preprocessor.get_transform("resnet", mode="eval")
    """

    _NORMALIZATION_MAP: Dict[
        str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ] = {
        "vgg": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "vgg16": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "resnet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "resnet152": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "mobilenet_v1": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        "mobilenet_v2": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        "mobilenet_v3": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    }

    def __init__(
        self,
        img_size: Tuple[int, int] = (360, 360),
        augment_translate: float = 0.2,
        augment_scale: Tuple[float, float] = (0.8, 1.2),
    ):
        """
        Initialize preprocessing settings.

        Args:
            img_size (Tuple[int, int]): Target resize dimensions.
            augment_translate (float): Maximum translation ratio for random affine.
            augment_scale (Tuple[float, float]): Range for random scaling.
        """
        self.img_size = img_size
        self.augment_translate = augment_translate
        self.augment_scale = augment_scale

    # -----------------------------
    # Core Augmentation
    # -----------------------------
    def _augmentation(self) -> transforms.RandomAffine:
        """
        Define a lightweight random affine transformation.

        Returns:
            transforms.RandomAffine: Configured affine transform for augmentation.
        """
        return transforms.RandomAffine(
            degrees=0,
            translate=(self.augment_translate, self.augment_translate),
            scale=self.augment_scale,
            fill=0,
        )

    def _compose(self, mean, std, augment: bool = False) -> transforms.Compose:
        """
        Build a sequential preprocessing pipeline.

        Args:
            mean (Tuple[float]): Normalization mean values.
            std (Tuple[float]): Normalization standard deviations.
            augment (bool): Whether to include random augmentations.

        Returns:
            transforms.Compose: Composed preprocessing pipeline.
        """
        ops = [transforms.Resize(self.img_size)]
        if augment:
            ops.append(self._augmentation())
        ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        return transforms.Compose(ops)

    # -----------------------------
    # Public API
    # -----------------------------
    def get_transform(self, model_name: str, mode: str = "train") -> transforms.Compose:
        """
        Return the preprocessing pipeline for the specified model and mode.

        Args:
            model_name (str): Model architecture name
                (e.g., 'vgg', 'resnet', 'mobilenet_v2', etc.)
            mode (str): One of ['train', 'eval']. Enables augmentation in 'train' mode.

        Returns:
            transforms.Compose: Model-specific preprocessing pipeline.
        """
        if model_name not in self._NORMALIZATION_MAP:
            raise ValueError(f"❌ Unsupported model: {model_name}")

        mean, std = self._NORMALIZATION_MAP[model_name]
        augment = mode.lower() == "train"
        return self._compose(mean, std, augment=augment)
