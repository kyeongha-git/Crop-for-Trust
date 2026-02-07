#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cnn_data_loader.py

Unified data loading module for image classification.

This module implements a PyTorch-compatible Dataset and DataLoader, handling:
- Recursive image file discovery.
- Automatic label encoding (string to integer).
- Data transformation and batching.
"""

import os
from typing import Dict, List, Tuple, Union

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
    Recursively collects image paths and class names from a directory.

    Args:
        root_dir (str): Root directory containing class subfolders.
        exts (Tuple[str, ...]): Valid image extensions to collect.

    Returns:
        List[Tuple[str, str]]: A list of (image_path, class_name) tuples.

    Raises:
        FileNotFoundError: If root_dir does not exist.
        ValueError: If no images are found in the directory.
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Data path not found: {root_dir}")

    image_label_pairs = []
    
    # Iterate through class folders
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
    Generates bidirectional mappings between class names and integer indices.

    Args:
        labels (List[str]): List of all class name occurrences.

    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: (name_to_idx, idx_to_name) mappings.
    """
    unique_labels = sorted(set(labels))
    label_to_idx = {name: idx for idx, name in enumerate(unique_labels)}
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


class ClassificationDataset(Dataset):
    """
    Custom Dataset for loading classification data from a folder structure.

    Expected Structure:
        root/
        ├── class_A/
        ├── class_B/
        └── ...
    """

    def __init__(
        self,
        input_dir: str,
        split: str = "train",
        transform: Union[transforms.Compose, None] = None,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            input_dir (str): Base path to the dataset.
            split (str): Subdirectory for the split (e.g., 'train', 'valid').
            transform (callable, optional): Image transformation pipeline.
            verbose (bool): If True, prints dataset statistics on init.
        """
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
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[any, int]:
        """
        Fetches the sample and label at the given index.

        Returns:
            Tuple[Tensor, int]: (transformed_image, label_index)
        """
        img_path = self.image_paths[idx]
        label_name = self.labels[idx]
        label = self.label_to_idx[label_name]

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label

    def _log_summary(self, input_dir: str, split: str) -> None:
        """Internal helper to print dataset details."""
        print(f"Loaded dataset from {input_dir}/{split}")
        print(f" - Samples: {len(self.image_paths)}")
        print(f" - Classes: {self.label_to_idx}")


# ============================================================
# DataLoader Builder
# ============================================================

def create_dataloader(
    input_dir: str,
    split: str = "train",
    transform: Union[transforms.Compose, None] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    verbose: bool = False,
) -> DataLoader:
    """
    Factory function to create a configured DataLoader.

    Args:
        input_dir (str): Dataset root directory.
        split (str): Data split to load ('train', 'valid', 'test').
        transform (callable, optional): Preprocessing transforms.
        batch_size (int): Mini-batch size.
        shuffle (bool): Whether to shuffle the data (default: True for train).
        num_workers (int): Number of subprocesses for data loading.
        verbose (bool): Verbosity flag.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
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