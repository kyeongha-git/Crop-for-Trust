#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_annotation_cleaner_pipeline.py
-----------------------------------
Comprehensive unit and integration tests for the AnnotationCleaner subsystem.

This file includes clean, self-contained tests for:
1️⃣ ImagePadder  → padding and metadata generation
2️⃣ CleanAnnotation → Gemini API mock image cleaning
3️⃣ RestoreCropper → ROI restoration from metadata

All tests are written in a clear, minimal, and reproducible format.
Each test verifies both file-level results and metadata integrity.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
from PIL import Image

from src.annotation_cleaner.core.clean_annotation import CleanAnnotation
from src.annotation_cleaner.core.image_padding import ImagePadder
from src.annotation_cleaner.core.restore_crop import RestoreCropper


# ============================================================
# Helper Functions
# ============================================================
def create_test_image(path: Path, size=(512, 512), color=(128, 128, 128)):
    """Create a simple RGB test image and save it to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path)
    return path


def setup_test_environment(tmp_path: Path):
    """Prepare a test environment with repair/replace subfolders and images."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    (input_dir / "repair").mkdir(parents=True)
    (input_dir / "replace").mkdir(parents=True)

    # Create small/large images for both categories
    create_test_image(input_dir / "repair" / "small_repair.jpg", size=(512, 512))
    create_test_image(input_dir / "repair" / "large_repair.jpg", size=(1200, 1200))
    create_test_image(input_dir / "replace" / "small_replace.jpg", size=(512, 512))
    create_test_image(input_dir / "replace" / "large_replace.jpg", size=(1400, 1400))

    return input_dir, output_dir


def load_metadata(meta_path: Path):
    """Load metadata JSON file and return its contents."""
    if not meta_path.exists():
        raise AssertionError(f"Metadata file missing: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_fake_padding_metadata(meta_path: Path, image_files):
    """Generate a fake padding_info.json file for testing RestoreCropper."""
    metadata = {
        file.name: {
            "orig_size": [512, 512],
            "pad_info": {"top": 256, "left": 256, "bottom": 256, "right": 256},
        }
        for file in image_files
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


# ============================================================
# Tests for ImagePadder
# ============================================================
def test_image_padder_run_creates_expected_outputs(tmp_path):
    """ImagePadder should pad small images and copy large ones."""
    input_dir, output_dir = setup_test_environment(tmp_path)

    padder = ImagePadder(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        categories=["repair", "replace"],
        target_size=1024,
    )
    padder.run()

    for category in ["repair", "replace"]:
        verify_category_output(output_dir, category)


def verify_category_output(output_dir: Path, category: str):
    """Verify padding results and metadata correctness per category."""
    out_dir = output_dir / category
    meta_path = out_dir / "padding_info.json"

    # Folder and output existence
    assert out_dir.exists(), f"{category} output directory missing"
    assert (out_dir / f"small_{category}.jpg").exists(), f"small_{category}.jpg missing"
    assert (out_dir / f"large_{category}.jpg").exists(), f"large_{category}.jpg missing"
    assert meta_path.exists(), f"{category} metadata file missing"

    # Validate metadata
    meta = load_metadata(meta_path)
    small_file = f"small_{category}.jpg"
    large_file = f"large_{category}.jpg"

    assert small_file in meta, f"{small_file} not in metadata"
    assert "pad_info" in meta[small_file], f"{small_file} missing pad_info"
    assert large_file not in meta, f"{large_file} should not have metadata"


# ============================================================
# Tests for CleanAnnotation
# ============================================================
def mock_gemini_client(tmp_image: Path):
    """Create a mock Gemini client that returns fake image bytes."""
    mock_client = MagicMock()

    with open(tmp_image, "rb") as f:
        fake_bytes = f.read()

    mock_inline_part = MagicMock()
    mock_inline_part.inline_data.data = fake_bytes
    mock_content = MagicMock()
    mock_content.parts = [mock_inline_part]
    mock_candidate = MagicMock()
    mock_candidate.content = mock_content
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]

    mock_client.models.generate_content.return_value = mock_response
    return mock_client


def test_generate_clean_image_creates_output(tmp_path):
    """_generate_clean_image should save the generated image file."""
    input_img = create_test_image(tmp_path / "input.jpg")
    output_img = tmp_path / "output.jpg"
    fake_client = mock_gemini_client(input_img)

    cleaner = CleanAnnotation(
        input_dir=str(tmp_path),
        output_dir=str(tmp_path),
        model="fake-model",
        prompt="Remove markings.",
        client=fake_client,
        test_mode=True,
    )

    success = cleaner._generate_clean_image(input_img, output_img)
    assert success is True, "_generate_clean_image returned False"
    assert output_img.exists(), "Output image not created"


def test_clean_annotation_run_creates_outputs(tmp_path):
    """run() should generate cleaned images per category."""
    input_dir, output_dir = setup_test_environment(tmp_path)
    fake_client = mock_gemini_client(input_dir / "repair" / "small_repair.jpg")

    cleaner = CleanAnnotation(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        model="fake-model",
        prompt="Remove markings.",
        categories=["repair", "replace"],
        client=fake_client,
        test_mode=True,
        test_limit=10,
    )
    cleaner.run()

    for category in ["repair", "replace"]:
        for filename in ["small", "large"]:
            img_name = f"{filename}_{category}.jpg"
            output_path = output_dir / category / img_name
            assert output_path.exists(), f"{img_name} not generated"

    fake_client.models.generate_content.assert_called()


# ============================================================
# Tests for RestoreCropper
# ============================================================
def test_restore_single_image_restores_correct_roi(tmp_path):
    """_restore_single_image should correctly crop ROI from padded image."""
    padded_img_path = tmp_path / "padded.jpg"
    img = np.full((1024, 1024, 3), 255, np.uint8)
    cv2.imwrite(str(padded_img_path), img)

    meta = {
        "orig_size": [512, 512],
        "pad_info": {"top": 256, "left": 256, "bottom": 256, "right": 256},
    }
    output_path = tmp_path / "restored.jpg"

    restorer = RestoreCropper(
        input_dir=str(tmp_path),
        output_dir=str(tmp_path),
        meta_dir=str(tmp_path),
        metadata_name="padding_info.json",
    )

    success = restorer._restore_single_image(padded_img_path, meta, output_path)

    assert success is True
    assert output_path.exists(), "Restored image not found"
    restored_img = cv2.imread(str(output_path))
    assert restored_img.shape[:2] == (512, 512), "Incorrect restored image size"


def test_restore_crop_run_restores_padded_images(tmp_path):
    """run() should restore padded images for each category."""
    input_dir, output_dir = setup_test_environment(tmp_path)
    meta_dir = tmp_path / "meta"
    padded_dir = tmp_path / "generated_image_padded"

    for category in ["repair", "replace"]:
        (padded_dir / category).mkdir(parents=True)
        create_test_image(
            padded_dir / category / f"small_{category}.jpg", size=(1024, 1024)
        )
        create_test_image(
            padded_dir / category / f"large_{category}.jpg", size=(1024, 1024)
        )
        meta_path = meta_dir / category / "padding_info.json"
        create_fake_padding_metadata(meta_path, [Path(f"small_{category}.jpg")])

    restorer = RestoreCropper(
        input_dir=str(padded_dir),
        output_dir=str(output_dir),
        meta_dir=str(meta_dir),
        categories=["repair", "replace"],
        metadata_name="padding_info.json",
    )
    restorer.run()

    for category in ["repair", "replace"]:
        out_dir = output_dir / category
        restored_small = out_dir / f"small_{category}.jpg"
        restored_large = out_dir / f"large_{category}.jpg"

        assert restored_small.exists(), f"{restored_small.name} not restored"
        assert restored_large.exists(), f"{restored_large.name} not copied"

        img_small = cv2.imread(str(restored_small))
        img_large = cv2.imread(str(restored_large))
        assert img_small.shape[:2] == (
            512,
            512,
        ), f"{category}: incorrect small restore size"
        assert img_large.shape[:2] == (
            1024,
            1024,
        ), f"{category}: large image should remain full-size"
