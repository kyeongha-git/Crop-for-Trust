#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cnn_data_loader.py
-------------------
Unified data loading module for image classification tasks.

This module provides:
- A PyTorch-compatible Dataset class for structured image folders.
- Automatic label mapping between string and integer IDs.
- A DataLoader helper for efficient mini-batch creation.
- Support for standard dataset splits (train / valid / test).
- Easy integration with custom datasets following a class-folder structure.
"""

import os
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ============================================================
# Utility Functions
# ============================================================
def list_image_paths(
    root_dir: str, exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
) -> List[Tuple[str, str]]:
    """
    Collect all image file paths and their corresponding class names from a directory.

    Args:
        root_dir (str): Root directory containing class subfolders.
        exts (Tuple[str]): Valid image file extensions.

    Returns:
        List[Tuple[str, str]]: List of (image_path, class_name) pairs.
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Data path not found: {root_dir}")

    image_label_pairs = []
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            if filename.lower().endswith(exts):
                image_label_pairs.append(
                    (os.path.join(class_path, filename), class_name)
                )
    if not image_label_pairs:
        raise ValueError(f"No images found under {root_dir}")
    return image_label_pairs


def build_label_mappings(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create bidirectional mappings between class names and integer labels.

    Args:
        labels (List[str]): List of class names.

    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: (label_to_idx, idx_to_label)
    """
    unique_labels = sorted(set(labels))
    label_to_idx = {name: idx for idx, name in enumerate(unique_labels)}
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


class ClassificationDataset(Dataset):
    """
    Dataset class for image classification tasks.

    Expects a folder structure like:
        data/
        ├── train/
        │   ├── class1/
        │   └── class2/
        ├── valid/
        └── test/

    Args:
        input_dir (str): Dataset root path (e.g., 'data/original').
        split (str): Data split ('train', 'valid', or 'test').
        transform (callable, optional): Torch transform for preprocessing.
        verbose (bool): Whether to print dataset summary.
    """

    def __init__(
        self,
        input_dir: str,
        split: str = "train",
        transform=None,
        verbose: bool = True,
    ):
        self.root_dir = os.path.join(input_dir, split)
        self.transform = transform

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Split path not found: {self.root_dir}")

        image_label_pairs = list_image_paths(self.root_dir)
        if not image_label_pairs:
            raise RuntimeError(f"No images found in {self.root_dir}")

        self.image_paths, self.labels = zip(*image_label_pairs)
        self.label_to_idx, self.idx_to_label = build_label_mappings(self.labels)

        if verbose:
            self._log_summary(input_dir, split)

    def __len__(self) -> int:
        """Return the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Load and return an image and its corresponding label index.

        Returns:
            (Tensor, int): Transformed image tensor and label index.
        """
        img_path = self.image_paths[idx]
        label_name = self.labels[idx]
        label = self.label_to_idx[label_name]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label

    def _log_summary(self, input_dir: str, split: str) -> None:
        """Print dataset information to console."""
        print(f"Loaded dataset from {input_dir}/{split}")
        print(f" - Samples: {len(self.image_paths)}")
        print(f" - Classes: {self.label_to_idx}")


# ============================================================
# Helper Functions
# ============================================================
def create_dataloader(
    input_dir: str,
    split: str = "train",
    transform=None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    verbose: bool = False,
) -> DataLoader:
    """
    Build a PyTorch DataLoader for a given dataset split.

    Args:
        input_dir (str): Dataset root path.
        split (str): 'train', 'valid', or 'test'.
        transform (callable, optional): Torch transform for preprocessing.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle training data.
        num_workers (int): Number of background workers.
        verbose (bool): Print dataset summary if True.

    Returns:
        DataLoader: Configured PyTorch DataLoader object.
    """
    dataset = ClassificationDataset(
        input_dir=input_dir,
        split=split,
        transform=transform,
        verbose=verbose,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader