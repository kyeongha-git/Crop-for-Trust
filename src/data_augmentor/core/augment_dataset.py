#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
augment_dataset.py
------------------
Core module for automated data augmentation and class balancing.

This module detects class imbalance between `train/repair` and `train/replace`
and performs augmentation based on parameters defined in `config.yaml`.

Usage:
    from data_augmentor.core.augment_dataset import balance_augmentation
"""

import math
import random
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageEnhance

from utils.logging import get_logger

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# Utility Functions
# ============================================================
def list_images(folder: Path):
    """Return a list of valid image files within the given folder."""
    return [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]


def clamp(v, lo, hi):
    """Clamp a numeric value to the given range."""
    return max(lo, min(hi, v))


# ============================================================
# Augmentation Primitives
# ============================================================
def random_resized_crop(
    img: Image.Image, scale=(0.9, 1.0), ratio=(0.95, 1.05), trials=10
):
    """Apply a random resized crop maintaining original output size."""
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

    # Fallback: center crop if all random attempts fail
    s = clamp(scale[0], 0.0, 1.0)
    new_w, new_h = int(w * s), int(h * s)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    crop = img.crop((left, top, left + new_w, top + new_h))
    return crop.resize((w, h), Image.BICUBIC)


def random_hflip(img: Image.Image, p=0.5):
    """Randomly apply horizontal flip."""
    return img.transpose(Image.FLIP_LEFT_RIGHT) if random.random() < p else img


def random_rotate(img: Image.Image, max_deg=10):
    """Apply random small rotation within ±max_deg."""
    deg = random.uniform(-max_deg, max_deg)
    return img.rotate(deg, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))


def random_translate(img: Image.Image, max_ratio=0.04):
    """Randomly translate image within a given ratio of its width/height."""
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
    b_range=(0.9, 1.1),
    c_range=(0.9, 1.15),
    s_range=(0.9, 1.15),
    hue_delta=0.03,
):
    """Randomly apply brightness, contrast, saturation, and hue adjustments."""
    if random.random() < 0.9:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(*b_range))
    if random.random() < 0.9:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(*c_range))
    if random.random() < 0.9:
        img = ImageEnhance.Color(img).enhance(random.uniform(*s_range))
    if hue_delta > 0 and random.random() < 0.5:
        img = img.convert("HSV")
        arr = np.array(img).astype(np.uint8)
        h_shift = int((random.uniform(-hue_delta, hue_delta)) * 255)
        arr[..., 0] = (arr[..., 0].astype(int) + h_shift) % 255
        img = Image.fromarray(arr, "HSV").convert("RGB")
    return img


# ============================================================
# Augmentation Pipeline
# ============================================================
def augment_pipeline(img: Image.Image, aug_cfg: Dict) -> Image.Image:
    """
    Apply config-defined augmentation operations sequentially.

    Each operation’s parameters are read from `config.yaml`.
    """
    img = random_resized_crop(
        img,
        scale=tuple(aug_cfg.get("random_resized_crop", {}).get("scale", (0.9, 1.0))),
        ratio=tuple(aug_cfg.get("random_resized_crop", {}).get("ratio", (0.95, 1.05))),
    )
    img = random_hflip(img, p=aug_cfg.get("random_hflip_p", 0.5))
    img = random_rotate(img, max_deg=aug_cfg.get("random_rotate_deg", 10))
    img = random_translate(img, max_ratio=aug_cfg.get("random_translate_ratio", 0.04))
    img = color_jitter(
        img,
        b_range=tuple(aug_cfg.get("brightness_range", (0.9, 1.1))),
        c_range=tuple(aug_cfg.get("contrast_range", (0.9, 1.15))),
        s_range=tuple(aug_cfg.get("saturation_range", (0.9, 1.15))),
        hue_delta=aug_cfg.get("hue_delta", 0.03),
    )
    return img


# ============================================================
# Class Balancing by Augmentation
# ============================================================
def _augment_until_equal(
    src_dir: Path, target_count: int, aug_cfg: Dict, seed: int = 42, logger=None
):
    """
    Augment images in the given folder until reaching target_count.

    Randomly samples from existing images, applies augmentations,
    and saves new images with incremented suffixes.
    """
    if logger is None:
        logger = get_logger("augment_dataset")

    random.seed(seed)
    files = list_images(src_dir)
    cur = len(files)
    save_quality = 95
    i = 0

    logger.info(f"Augmenting {src_dir.name}: {cur} → {target_count}")

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


def balance_augmentation(root_dir: Path, aug_cfg: Dict, seed: int = 42, logger=None):
    """
    Detect class imbalance between `train/repair` and `train/replace`
    and automatically augment the smaller class until both are balanced.
    """
    if logger is None:
        logger = get_logger("augment_dataset")

    train_dir = root_dir / "train"
    train_repair = train_dir / "repair"
    train_replace = train_dir / "replace"

    if not train_repair.exists() or not train_replace.exists():
        raise FileNotFoundError(
            f"Missing directories: {train_repair} or {train_replace}"
        )

    repair_count = len(list_images(train_repair))
    replace_count = len(list_images(train_replace))

    logger.info(f"[Counts] train/repair={repair_count}, train/replace={replace_count}")

    if repair_count == replace_count:
        logger.info("Classes are already balanced.")
        return

    smaller_dir = train_repair if repair_count < replace_count else train_replace
    target_count = max(repair_count, replace_count)

    logger.info(
        f"Class '{smaller_dir.name}' is smaller. ({len(list_images(smaller_dir))} → {target_count})"
    )
    _augment_until_equal(smaller_dir, target_count, aug_cfg, seed=seed, logger=logger)
