#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
split_dataset.py

Module for partitioning datasets into training, validation, and test subsets.
Maintains directory structure while distributing images based on configured ratios.
"""

import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logging import get_logger


# ------------------------------------------------------------------------------
# Core Functions
# ------------------------------------------------------------------------------

def get_images(class_path: Path) -> List[Path]:
    """
    Retrieves a sorted list of valid image files from a specific class directory.

    Args:
        class_path (Path): Path to the class directory.

    Returns:
        List[Path]: Sorted list of image file paths.
    """
    valid_ext = (".jpg", ".jpeg", ".png")
    return sorted(
        [p for p in class_path.iterdir() if p.is_file() and p.suffix.lower() in valid_ext]
    )


def make_splits(
    images: List[Path], 
    train_ratio: float, 
    valid_ratio: float, 
    seed: int = 42
) -> Dict[str, List[Path]]:
    """
    Partitions a list of images into train, validation, and test sets.

    

    Args:
        images (List[Path]): List of all image paths.
        train_ratio (float): Proportion of data for training.
        valid_ratio (float): Proportion of data for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        Dict[str, List[Path]]: Dictionary mapping split names to file lists.
    """
    random.seed(seed)
    shuffled_images = list(images)
    random.shuffle(shuffled_images)

    total = len(shuffled_images)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train_set = shuffled_images[:train_end]
    valid_set = shuffled_images[train_end:valid_end]
    test_set = shuffled_images[valid_end:]

    # Heuristic handling for extremely small datasets (e.g., in demo mode)
    if 0 < total < 5:
        if not valid_set and len(train_set) > 0:
            valid_set = [train_set.pop()]
        
        if not train_set and len(images) > 0:
             train_set = [images[0]]

    return {"train": train_set, "valid": valid_set, "test": test_set}


def copy_images(
    class_name: str,
    class_path: Path,
    output_dir: Path,
    splits: Dict[str, List[Path]],
    logger: Any,
) -> None:
    """
    Copies images from the source to the destination split directories.

    Args:
        class_name (str): Name of the class (category).
        class_path (Path): Source directory for the class.
        output_dir (Path): Root directory for the split dataset.
        splits (Dict[str, List[Path]]): Mapping of split names to file lists.
        logger (Any): Logger instance.
    """
    for split_name, files in splits.items():
        split_dir = output_dir / split_name / class_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for src_path in files:
            dst_path = split_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            
        logger.info(f"Copied {len(files):>4} → {split_name}/{class_name}")


def split_dataset(
    data_dir: Path,
    output_dir: Path,
    split_cfg: Dict[str, float],
    seed: int = 42,
    logger: Optional[Any] = None,
) -> None:
    """
    Orchestrates the dataset splitting process for all classes.

    

    Args:
        data_dir (Path): Input directory containing class subfolders.
        output_dir (Path): Destination directory for the split dataset.
        split_cfg (Dict[str, float]): Configuration containing split ratios.
        seed (int): Random seed.
        logger (Optional[Any]): Logger instance.
    """
    if logger is None:
        logger = get_logger("split_dataset")

    train_ratio = split_cfg.get("train_ratio", 0.8)
    valid_ratio = split_cfg.get("valid_ratio", 0.1)
    test_ratio = split_cfg.get("test_ratio", 0.1)
    
    # Normalize ratios if they do not sum to 1.0 due to float precision
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        logger.warning("Ratios do not sum to 1. Normalizing...")
        total = train_ratio + valid_ratio + test_ratio
        train_ratio /= total
        valid_ratio /= total
        test_ratio /= total

    logger.info(f"Starting dataset split: {data_dir}")
    logger.info(f" - Output Dir: {output_dir}")
    logger.info(
        f" - Ratios: train={train_ratio:.2f}, valid={valid_ratio:.2f}, test={test_ratio:.2f}"
    )

    # Identify class directories, excluding output dir and hidden files
    categories = []
    for d in data_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("."):
            continue
        if d.resolve() == output_dir.resolve():
            continue  
        categories.append(d.name)

    categories.sort()

    if not categories:
        logger.warning(f"No class folders found in {data_dir}")
        return

    # Process each class
    for class_name in categories:
        class_path = data_dir / class_name
        images = get_images(class_path)
        
        if not images:
            logger.warning(f"No images found in {class_name}. Skipping.")
            continue

        splits = make_splits(images, train_ratio, valid_ratio, seed)
        copy_images(class_name, class_path, output_dir, splits, logger)

        logger.info(
            f"[{class_name}] "
            f"train={len(splits['train'])}, "
            f"valid={len(splits['valid'])}, "
            f"test={len(splits['test'])}"
        )

    logger.info("Dataset splitting complete.")