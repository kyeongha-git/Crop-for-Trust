#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_preprocessing.py

Centralized preprocessing manager for image classification models.
Standardizes input normalization and augmentation pipelines across different architectures.
"""

from typing import Dict, Tuple

from torchvision import transforms


class DataPreprocessor:
    """
    Manages data preprocessing and augmentation strategies.
    Ensures input consistency by applying model-specific normalization statistics.
    """

    # ImageNet normalization statistics for supported architectures
    _NORMALIZATION_MAP: Dict[
        str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ] = {
        "vgg": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "vgg16": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "resnet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "resnet152": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "mobilenet_v1": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        "mobilenetv2": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        "mobilenet_v2": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        "mobilenetv3": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        "mobilenet_v3": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    }

    def __init__(
        self,
        img_size: Tuple[int, int] = (360, 360),
        augment_translate: float = 0.2,
        augment_scale: Tuple[float, float] = (0.8, 1.2),
    ) -> None:
        """
        Args:
            img_size (Tuple[int, int]): Target resolution (height, width).
            augment_translate (float): Max translation factor for random affine.
            augment_scale (Tuple[float, float]): Scaling range for random affine.
        """
        self.img_size = img_size
        self.augment_translate = augment_translate
        self.augment_scale = augment_scale

    # -----------------------------
    # Augmentation Logic
    # -----------------------------

    def _augmentation(self) -> transforms.RandomAffine:
        """Returns the configured random affine transformation."""
        return transforms.RandomAffine(
            degrees=0,
            translate=(self.augment_translate, self.augment_translate),
            scale=self.augment_scale,
            fill=0,
        )

    def _compose(
        self, 
        mean: Tuple[float, float, float], 
        std: Tuple[float, float, float], 
        augment: bool = False
    ) -> transforms.Compose:
        """
        Builds the sequential preprocessing pipeline.

        Args:
            mean (Tuple[float, ...]): Normalization mean.
            std (Tuple[float, ...]): Normalization std.
            augment (bool): If True, applies random augmentations.

        Returns:
            transforms.Compose: The complete transformation pipeline.
        """
        ops = [transforms.Resize(self.img_size)]
        
        if augment:
            ops.append(self._augmentation())
            
        ops.extend([
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ])
        
        return transforms.Compose(ops)

    def get_transform(self, model_name: str, mode: str = "train") -> transforms.Compose:
        """
        Retrieves the model-specific transform.

        Args:
            model_name (str): Key for the normalization map (e.g., 'resnet', 'mobilenet_v2').
            mode (str): Execution mode ('train' enables augmentation, 'eval' does not).

        Returns:
            transforms.Compose: Configured transform.

        Raises:
            ValueError: If the model architecture is not supported.
        """
        if model_name not in self._NORMALIZATION_MAP:
            raise ValueError(f"Unsupported model: {model_name}")

        mean, std = self._NORMALIZATION_MAP[model_name]
        augment = mode.lower() == "train"
        
        return self._compose(mean, std, augment=augment)