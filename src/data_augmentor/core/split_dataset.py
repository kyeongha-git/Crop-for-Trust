#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
split_dataset.py
-----------------
Module for splitting a dataset into train / valid / test sets.

"""

import random
import shutil
from pathlib import Path
from typing import Dict, List

from utils.logging import get_logger


# ============================================================
# 🔹 Core Functions
# ============================================================
def get_images(class_path: Path) -> List[Path]:
    """Return a sorted list of image files from a given class folder."""
    valid_ext = (".jpg", ".jpeg", ".png")
    return sorted([p for p in class_path.iterdir() if p.suffix.lower() in valid_ext])


def make_splits(
    images: List[Path], train_ratio: float, valid_ratio: float, seed: int = 42
) -> Dict[str, List[Path]]:
    """
    Split a list of image paths into train/valid/test subsets.

    Args:
        images (List[Path]): All image paths.
        train_ratio (float): Ratio for training set.
        valid_ratio (float): Ratio for validation set.
        seed (int): Random seed for reproducibility.

    Returns:
        Dict[str, List[Path]]: Mapping of split names to image lists.
    """
    random.seed(seed)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    return {
        "train": images[:train_end],
        "valid": images[train_end:valid_end],
        "test": images[valid_end:],
    }


def copy_images(
    class_name: str,
    class_path: Path,
    output_dir: Path,
    splits: Dict[str, List[Path]],
    logger,
) -> None:
    """
    Copy split images into their respective folders.

    Example target structure:
        output_dir/
            ├── train/class_name/
            ├── valid/class_name/
            └── test/class_name/
    """
    for split_name, files in splits.items():
        split_dir = output_dir / split_name / class_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for src_path in files:
            dst_path = split_dir / src_path.name
            shutil.copy2(src_path, dst_path)
        logger.info(f"📦 Copied {len(files):>4} → {split_name}/{class_name}")


def split_dataset(
    data_dir: Path,
    output_dir: Path,
    split_cfg: Dict[str, float],
    seed: int = 42,
    logger=None,
) -> None:
    """
    Perform dataset splitting for each class folder.

    Automatically divides images into train/valid/test
    based on provided ratios in `config.yaml`.

    Args:
        data_dir (Path): Root dataset directory containing class subfolders.
        output_dir (Path): Destination directory for split datasets.
        split_cfg (Dict[str, float]): Split ratios for train/valid/test.
        seed (int): Random seed for reproducibility.
        logger: Optional logger instance.
    """
    if logger is None:
        logger = get_logger("split_dataset")

    train_ratio = split_cfg.get("train_ratio", 0.8)
    valid_ratio = split_cfg.get("valid_ratio", 0.1)
    test_ratio = split_cfg.get("test_ratio", 0.1)
    assert (
        abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6
    ), "Train/Valid/Test ratios must sum to 1."

    logger.info(f"📁 Starting dataset split: {data_dir}")
    logger.info(f" - Output Dir: {output_dir}")
    logger.info(
        f" - Ratios: train={train_ratio}, valid={valid_ratio}, test={test_ratio}"
    )

    categories = [d.name for d in data_dir.iterdir() if d.is_dir()]
    if not categories:
        logger.warning(f"⚠️ No class folders found in {data_dir}")
        return

    for class_name in categories:
        class_path = data_dir / class_name
        images = get_images(class_path)
        if not images:
            logger.warning(f"[⚠️] No images found in {class_name}. Skipping.")
            continue

        splits = make_splits(images, train_ratio, valid_ratio, seed)
        copy_images(class_name, class_path, output_dir, splits, logger)

        logger.info(
            f"[{class_name}] ✅ "
            f"train={len(splits['train'])}, "
            f"valid={len(splits['valid'])}, "
            f"test={len(splits['test'])}"
        )

    logger.info("✅ Dataset splitting complete!")
