#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_data_processing.py
----------------------
Unit and integration tests for the dataset splitting and augmentation modules.

Modules tested:
- src.data_augmentor.core.split_dataset
- src.data_augmentor.core.augment_dataset

Test Goals:
- Verify correct split ratios and folder structures for train/valid/test sets
- Ensure deterministic behavior given fixed random seeds
- Confirm graceful handling of empty or invalid class folders
- Validate correctness of augmentation primitives and pipelines
"""

import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops

from src.data_augmentor.core.augment_dataset import (augment_pipeline,
                                                     balance_augmentation,
                                                     clamp, color_jitter,
                                                     list_images, random_hflip,
                                                     random_resized_crop,
                                                     random_rotate,
                                                     random_translate)
from src.data_augmentor.core.split_dataset import split_dataset

# ============================================================
# Fixtures for Split Dataset Tests
# ============================================================


@pytest.fixture
def dummy_dataset(tmp_path):
    """
    Create a dummy dataset containing 'repair' and 'replace' folders
    with 10 text-based fake images each.
    """
    base = tmp_path
    for cls in ["repair", "replace"]:
        cls_dir = base / cls
        cls_dir.mkdir(parents=True)
        for i in range(10):
            (cls_dir / f"img_{i}.jpg").write_text(f"fake image {i}")
    return base


@pytest.fixture
def dummy_split_config():
    """Return a simple split ratio configuration for dataset splitting."""
    return {"train_ratio": 0.6, "valid_ratio": 0.2, "test_ratio": 0.2}


# ============================================================
# Split Dataset Unit Tests
# ============================================================


def test_split_dataset_creates_splits(dummy_dataset, dummy_split_config, tmp_path):
    """
    Ensure split_dataset() creates proper train/valid/test folders,
    and each class has the expected number of samples.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    split_dataset(
        data_dir=dummy_dataset,
        output_dir=output_dir,
        split_cfg=dummy_split_config,
        seed=42,
    )

    # Verify class folders exist for each split
    for split in ["train", "valid", "test"]:
        for cls in ["repair", "replace"]:
            target_dir = output_dir / split / cls
            assert target_dir.exists(), f"Missing folder: {target_dir}"

    # Count number of images per split
    def count_images(split):
        return sum(
            len(list((output_dir / split / cls).glob("*.jpg")))
            for cls in ["repair", "replace"]
        )

    total_imgs = sum(
        len(list((dummy_dataset / cls).glob("*.jpg"))) for cls in ["repair", "replace"]
    )
    train_imgs = count_images("train")
    valid_imgs = count_images("valid")
    test_imgs = count_images("test")

    expected_train = int(total_imgs * dummy_split_config["train_ratio"])
    expected_valid = int(total_imgs * dummy_split_config["valid_ratio"])
    expected_test = total_imgs - expected_train - expected_valid

    # Quantitative comparison
    assert (
        train_imgs == expected_train
    ), f"Train split mismatch: {train_imgs} vs {expected_train}"
    assert (
        valid_imgs == expected_valid
    ), f"Valid split mismatch: {valid_imgs} vs {expected_valid}"
    assert (
        test_imgs == expected_test
    ), f"Test split mismatch: {test_imgs} vs {expected_test}"

    # Total consistency check
    assert total_imgs == train_imgs + valid_imgs + test_imgs, "Split total mismatch"


def test_split_dataset_reproducibility(dummy_dataset, dummy_split_config, tmp_path):
    """
    Verify that dataset splitting is deterministic when using the same random seed.
    """
    output_1 = tmp_path / "out1"
    output_2 = tmp_path / "out2"

    split_dataset(dummy_dataset, output_1, dummy_split_config, seed=123)
    split_dataset(dummy_dataset, output_2, dummy_split_config, seed=123)

    train_1 = sorted((output_1 / "train" / "repair").iterdir())
    train_2 = sorted((output_2 / "train" / "repair").iterdir())

    assert [p.name for p in train_1] == [p.name for p in train_2]


def test_split_dataset_empty_class(tmp_path, dummy_split_config, caplog):
    """Ensure split_dataset() handles empty class folders or missing images gracefully."""
    empty_class_dir = tmp_path / "repair"
    empty_class_dir.mkdir()
    (tmp_path / "replace").mkdir()

    split_dataset(tmp_path, tmp_path / "output", dummy_split_config, seed=1)

    # Match exact log messages from current split_dataset.py
    assert any(
        phrase in caplog.text
        for phrase in [
            "No images found",  # when class folder exists but empty
            "No class folders found",  # when no subfolders exist at all
        ]
    ), f"Expected warning log not found. Got:\n{caplog.text}"


def test_split_dataset_invalid_ratios(dummy_dataset, tmp_path):
    """
    Ensure invalid ratio configurations raise an AssertionError.
    """
    bad_cfg = {"train_ratio": 0.5, "valid_ratio": 0.5, "test_ratio": 0.2}
    with pytest.raises(AssertionError):
        split_dataset(dummy_dataset, tmp_path / "out", bad_cfg)


# ============================================================
# Augmentation Tests
# ============================================================
@pytest.fixture
def temp_dir():
    """Create a temporary directory fixture for augmentation tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image(temp_dir):
    """
    Create a small 64×64 RGB gradient image for augmentation tests.
    Produces a left-to-right asymmetric color pattern for visual difference.
    """
    img_path = temp_dir / "sample.jpg"
    img = Image.new("RGB", (64, 64))
    for x in range(64):
        for y in range(64):
            color = (x * 4, y * 2, 128)
            img.putpixel((x, y), color)
    img.save(img_path)
    return img_path


# ============================================================
# Helper Utilities
# ============================================================
def _img_to_array(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a NumPy array."""
    return np.asarray(img, dtype=np.uint8).copy()


def _images_different(img1: Image.Image, img2: Image.Image) -> bool:
    """Return True if two images differ (using bounding box of pixel difference)."""
    diff = ImageChops.difference(img1, img2)
    return diff.getbbox() is not None


def count_images_in_dir(directory: Path) -> int:
    """Count image files (jpg/png/jpeg) in the given directory."""
    return len(
        [
            f
            for f in directory.glob("*")
            if f.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ]
    )


# ============================================================
# Unit Tests — Augmentation Primitives
# ============================================================
def test_list_images_filters_correctly(temp_dir):
    """Ensure list_images() returns only image files with correct extensions."""
    (temp_dir / "a.jpg").touch()
    (temp_dir / "b.png").touch()
    (temp_dir / "c.txt").touch()
    files = list_images(temp_dir)
    assert all(f.suffix.lower() in [".jpg", ".png"] for f in files)
    assert len(files) == 2


def test_clamp_behaves_correctly():
    """Verify clamp() properly restricts values within [min, max]."""
    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(50, 0, 10) == 10


def test_random_resized_crop_changes_composition(sample_image):
    """Ensure random_resized_crop() modifies spatial composition while preserving size."""
    img = Image.open(sample_image)
    cropped = random_resized_crop(img)
    assert cropped.size == img.size
    assert _images_different(img, cropped)


def test_random_hflip_flips_horizontally(sample_image):
    """Ensure random_hflip() correctly mirrors the image horizontally when p=1.0."""
    img = Image.open(sample_image)
    arr = _img_to_array(img)
    arr[:, :32, :] = 0
    img = Image.fromarray(arr)
    flipped = random_hflip(img, p=1.0)
    assert np.array_equal(_img_to_array(flipped)[:, 0, :], arr[:, -1, :])


def test_random_rotate_rotates_pixels(sample_image):
    """Ensure random_rotate() changes pixel arrangement significantly."""
    img = Image.open(sample_image)
    arr_before = _img_to_array(img)
    rotated = random_rotate(img, max_deg=45)
    arr_after = _img_to_array(rotated)
    assert np.mean(arr_before != arr_after) > 0.05


def test_random_translate_shifts_image_content(sample_image):
    """Ensure random_translate() shifts pixel content by a visible amount."""
    img = Image.open(sample_image)
    arr_before = _img_to_array(img)
    translated = random_translate(img, max_ratio=0.2)
    arr_after = _img_to_array(translated)
    assert np.mean(arr_before != arr_after) > 0.05


def test_color_jitter_modifies_pixel_values(sample_image):
    """Ensure color_jitter() alters brightness/contrast/saturation statistics."""
    random.seed(42)
    np.random.seed(42)
    img = Image.open(sample_image)
    arr_before = _img_to_array(img)
    out = color_jitter(img)
    arr_after = _img_to_array(out)
    mean_diff = abs(arr_before.mean() - arr_after.mean())
    assert mean_diff > 0.5, f"Color jitter effect too small (Δmean={mean_diff:.3f})"


def test_augment_pipeline_combines_all(sample_image):
    """Verify augment_pipeline() executes a full augmentation chain successfully."""
    img = Image.open(sample_image)
    dummy_cfg = {
        "random_resized_crop": {"scale": [0.9, 1.0], "ratio": [0.95, 1.05]},
        "random_hflip_p": 0.5,
        "random_rotate_deg": 15,
        "random_translate_ratio": 0.05,
        "brightness_range": [0.9, 1.1],
        "contrast_range": [0.9, 1.15],
        "saturation_range": [0.9, 1.15],
        "hue_delta": 0.03,
    }
    result = augment_pipeline(img, dummy_cfg)
    assert isinstance(result, Image.Image)
    assert result.size == img.size


def test_balance_augmentation_balances_classes(temp_dir):
    """Ensure balance_augmentation() upsamples minority classes to achieve balance."""
    train_repair = temp_dir / "train" / "repair"
    train_replace = temp_dir / "train" / "replace"
    train_repair.mkdir(parents=True, exist_ok=True)
    train_replace.mkdir(parents=True, exist_ok=True)

    # repair: 5 images, replace: 2 images
    for i in range(5):
        Image.new("RGB", (32, 32), (128, 128, 128)).save(
            train_repair / f"repair_{i}.jpg"
        )
    for i in range(2):
        Image.new("RGB", (32, 32), (128, 128, 128)).save(
            train_replace / f"replace_{i}.jpg"
        )

    dummy_cfg = {"random_hflip_p": 1.0}
    balance_augmentation(root_dir=temp_dir, aug_cfg=dummy_cfg, seed=123)

    repair_count = count_images_in_dir(train_repair)
    replace_count = count_images_in_dir(train_replace)
    assert repair_count == replace_count


def test_balance_augmentation_handles_equal_case(temp_dir):
    """Ensure no augmentation occurs if class distributions are already balanced."""
    train_repair = temp_dir / "train" / "repair"
    train_replace = temp_dir / "train" / "replace"
    train_repair.mkdir(parents=True, exist_ok=True)
    train_replace.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        img = Image.new("RGB", (32, 32), (128, 128, 128))
        img.save(train_repair / f"repair_{i}.jpg")
        img.save(train_replace / f"replace_{i}.jpg")

    dummy_cfg = {"random_hflip_p": 1.0}
    before = count_images_in_dir(train_repair)
    balance_augmentation(root_dir=temp_dir, aug_cfg=dummy_cfg, seed=42)
    after = count_images_in_dir(train_repair)
    assert before == after
