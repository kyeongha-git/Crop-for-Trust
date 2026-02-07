#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
augment_dataset.py

Core module for automated data augmentation and class balancing.

This script detects class imbalances in the training set and generates synthetic
samples using geometric and photometric transformations to achieve a balanced distribution.
"""

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image, ImageEnhance

from utils.logging import get_logger

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def list_images(folder: Path) -> List[Path]:
    """
    Retrieves a sorted list of valid image files from a directory.

    Args:
        folder (Path): Target directory path.

    Returns:
        List[Path]: List of image file paths.
    """
    return [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamps a value to the range [lo, hi]."""
    return max(lo, min(hi, v))


# ------------------------------------------------------------------------------
# Augmentation Primitives
# ------------------------------------------------------------------------------

def random_resized_crop(
    img: Image.Image,
    scale: Tuple[float, float] = (0.9, 1.0),
    ratio: Tuple[float, float] = (0.95, 1.05),
    trials: int = 10,
) -> Image.Image:
    """
    Applies Random Resized Crop (RRC).

    Attempts to crop a random region with a random aspect ratio and resize it
    back to the original dimensions. Falls back to a center crop if constraints
    are not met within the given trials.

    Args:
        img (Image.Image): Input PIL Image.
        scale (Tuple[float, float]): Range of scale of the area of the crop.
        ratio (Tuple[float, float]): Range of aspect ratio of the crop.
        trials (int): Number of attempts before falling back to center crop.

    Returns:
        Image.Image: Cropped and resized image.
    """
    w, h = img.size
    area = w * h

    for _ in range(trials):
        target_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        new_ratio = math.exp(random.uniform(*log_ratio))
        
        new_w = int(round(math.sqrt(target_area * new_ratio)))
        new_h = int(round(math.sqrt(target_area / new_ratio)))
        
        if 0 < new_w <= w and 0 < new_h <= h:
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            crop = img.crop((left, top, left + new_w, top + new_h))
            return crop.resize((w, h), Image.BICUBIC)

    # Fallback: Center crop
    s = clamp(scale[0], 0.0, 1.0)
    new_w, new_h = int(w * s), int(h * s)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    crop = img.crop((left, top, left + new_w, top + new_h))
    return crop.resize((w, h), Image.BICUBIC)


def random_hflip(img: Image.Image, p: float = 0.5) -> Image.Image:
    """
    Randomly flips the image horizontally.

    Args:
        img (Image.Image): Input image.
        p (float): Probability of flipping.

    Returns:
        Image.Image: Flipped or original image.
    """
    return img.transpose(Image.FLIP_LEFT_RIGHT) if random.random() < p else img


def random_rotate(img: Image.Image, max_deg: float = 10) -> Image.Image:
    """
    Rotates the image by a random degree within [-max_deg, max_deg].

    Args:
        img (Image.Image): Input image.
        max_deg (float): Maximum rotation angle in degrees.

    Returns:
        Image.Image: Rotated image.
    """
    deg = random.uniform(-max_deg, max_deg)
    return img.rotate(deg, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))


def random_translate(img: Image.Image, max_ratio: float = 0.04) -> Image.Image:
    """
    Translates the image randomly.

    Args:
        img (Image.Image): Input image.
        max_ratio (float): Fraction of width/height for maximum translation.

    Returns:
        Image.Image: Translated image.
    """
    w, h = img.size
    tx = int(random.uniform(-max_ratio, max_ratio) * w)
    ty = int(random.uniform(-max_ratio, max_ratio) * h)
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0, tx, 0, 1, ty),
        resample=Image.BICUBIC,
        fillcolor=(0, 0, 0),
    )


def color_jitter(
    img: Image.Image,
    b_range: Tuple[float, float] = (0.9, 1.1),
    c_range: Tuple[float, float] = (0.9, 1.15),
    s_range: Tuple[float, float] = (0.9, 1.15),
    hue_delta: float = 0.03,
) -> Image.Image:
    """
    Randomly adjusts brightness, contrast, saturation, and hue.

    Args:
        img (Image.Image): Input image.
        b_range (Tuple[float, float]): Brightness factor range.
        c_range (Tuple[float, float]): Contrast factor range.
        s_range (Tuple[float, float]): Saturation factor range.
        hue_delta (float): Maximum hue shift.

    Returns:
        Image.Image: Color-jittered image.
    """
    if random.random() < 0.9:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(*b_range))
    if random.random() < 0.9:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(*c_range))
    if random.random() < 0.9:
        img = ImageEnhance.Color(img).enhance(random.uniform(*s_range))
    
    # Hue adjustment using HSV conversion
    if hue_delta > 0 and random.random() < 0.5:
        img = img.convert("HSV")
        arr = np.array(img).astype(np.uint8)
        h_shift = int((random.uniform(-hue_delta, hue_delta)) * 255)
        arr[..., 0] = (arr[..., 0].astype(int) + h_shift) % 255
        img = Image.fromarray(arr, "HSV").convert("RGB")
    
    return img


# ------------------------------------------------------------------------------
# Augmentation Pipeline
# ------------------------------------------------------------------------------

def augment_pipeline(img: Image.Image, aug_cfg: Dict) -> Image.Image:
    """
    Sequentially applies configured augmentations to the image.

    Args:
        img (Image.Image): Original image.
        aug_cfg (Dict): Configuration dictionary containing augmentation parameters.

    Returns:
        Image.Image: Augmented image.
    """
    # 1. Geometric transformations
    img = random_resized_crop(
        img,
        scale=tuple(aug_cfg.get("random_resized_crop", {}).get("scale", (0.9, 1.0))),
        ratio=tuple(aug_cfg.get("random_resized_crop", {}).get("ratio", (0.95, 1.05))),
    )
    img = random_hflip(img, p=aug_cfg.get("random_hflip_p", 0.5))
    img = random_rotate(img, max_deg=aug_cfg.get("random_rotate_deg", 10))
    img = random_translate(img, max_ratio=aug_cfg.get("random_translate_ratio", 0.04))

    # 2. Photometric transformations
    img = color_jitter(
        img,
        b_range=tuple(aug_cfg.get("brightness_range", (0.9, 1.1))),
        c_range=tuple(aug_cfg.get("contrast_range", (0.9, 1.15))),
        s_range=tuple(aug_cfg.get("saturation_range", (0.9, 1.15))),
        hue_delta=aug_cfg.get("hue_delta", 0.03),
    )
    return img


# ------------------------------------------------------------------------------
# Class Balancing Logic
# ------------------------------------------------------------------------------

def _augment_until_equal(
    src_dir: Path, 
    target_count: int, 
    aug_cfg: Dict, 
    seed: int = 42, 
    logger: Optional[Any] = None
) -> None:
    """
    Generates synthetic images until the class count reaches target_count.

    Args:
        src_dir (Path): Directory of the class to augment.
        target_count (int): Desired total number of images.
        aug_cfg (Dict): Augmentation configuration.
        seed (int): Random seed for reproducibility.
        logger (Any): Logger instance.
    """
    if logger is None:
        logger = get_logger("augment_dataset")

    random.seed(seed)
    files = list_images(src_dir)
    cur = len(files)
    save_quality = 95
    i = 0

    logger.info(f"Augmenting {src_dir.name}: {cur} -> {target_count}")

    while cur < target_count:
        src = random.choice(files)
        try:
            with Image.open(src).convert("RGB") as img:
                aug = augment_pipeline(img, aug_cfg)
                out_name = f"{src.stem}_aug_{i:05d}.jpg"
                out_path = src_dir / out_name
                
                if out_path.exists():
                    i += 1
                    continue
                
                aug.save(out_path, quality=save_quality)
                cur += 1
                i += 1
        except Exception as e:
            logger.warning(f"Skip {src.name}: {e}")

    logger.info(f"Completed! {src_dir.name} balanced to {cur} images.")


def balance_augmentation(
    root_dir: Path,
    aug_cfg: Dict,
    seed: int = 42,
    logger: Optional[Any] = None,
) -> None:
    """
    Orchestrates the class balancing process.

    Scans the training directory, determines the maximum class count,
    and augments minority classes to match the majority.

    Args:
        root_dir (Path): Dataset root directory containing a 'train' folder.
        aug_cfg (Dict): Augmentation configuration.
        seed (int): Random seed.
        logger (Any): Logger instance.
    """
    if logger is None:
        logger = get_logger("augment_dataset")

    train_dir = root_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")

    # Discover class directories
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]

    if len(class_dirs) < 2:
        logger.warning("Less than two classes detected. Augmentation skipped.")
        return

    # Analyze class distribution
    class_counts = {d.name: len(list_images(d)) for d in class_dirs}

    for cls, cnt in class_counts.items():
        logger.info(f"[Count] train/{cls} = {cnt}")

    target_count = max(class_counts.values())
    logger.info(f"Target count for balancing: {target_count}")

    # Augment minority classes
    for class_dir in class_dirs:
        cur_count = class_counts[class_dir.name]

        if cur_count >= target_count:
            continue

        _augment_until_equal(
            src_dir=class_dir,
            target_count=target_count,
            aug_cfg=aug_cfg,
            seed=seed,
            logger=logger,
        )