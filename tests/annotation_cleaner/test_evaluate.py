#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_evaluator.py
-----------------
Unit and integration tests for the `Evaluator` class.

Test Goals:
1️⃣ Verify initialization from injected configuration.
2️⃣ Validate metric computation logic (mock image pairs).
3️⃣ Confirm that full-image evaluation produces correct metrics and CSV output.
4️⃣ Ensure YOLO-based crop evaluation works correctly with mocked YOLO predictions.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.annotation_cleaner.evaluate import Evaluator


# ============================================================
# Helper Functions
# ============================================================
def create_dummy_image(path: Path, color=(128, 128, 128), size=(64, 64)):
    """Create a uniform-color dummy image for testing."""
    img = np.full((*size, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


# ============================================================
# PyTest Fixture
# ============================================================
@pytest.fixture
def temp_dataset(tmp_path):
    """
    Create a dummy dataset structure for Evaluator tests.

    Structure:
    orig/
        repair/img1.jpg
        replace/img1.jpg
    gen/
        repair/img1.jpg
        replace/img1.jpg

    """
    orig_dir = tmp_path / "orig"
    gen_dir = tmp_path / "gen"
    metric_dir = tmp_path / "metrics"

    for root in [orig_dir, gen_dir]:
        for cat in ["repair", "replace"]:
            d = root / cat
            d.mkdir(parents=True, exist_ok=True)
            create_dummy_image(d / "img1.jpg", color=(100, 100, 100))
            create_dummy_image(d / "img2.jpg", color=(150, 150, 150))

    return {
        "orig_dir": orig_dir,
        "gen_dir": gen_dir,
        "metric_dir": metric_dir,
    }


# ============================================================
# Test 1: Initialization
# ============================================================
def test_evaluator_initialization(temp_dataset):
    """Evaluator should initialize correctly with injected config."""
    cfg = {
        "orig_dir": str(temp_dataset["orig_dir"]),
        "gen_dir": str(temp_dataset["gen_dir"]),
        "metric_dir": str(temp_dataset["metric_dir"]),
        "metrics": ["ssim", "l1", "edge_iou"],
        "yolo_model": "./dummy_yolo.pt",
        "imgsz": 416,
    }

    evaluator = Evaluator(**cfg)
    assert evaluator.orig_dir.exists()
    assert evaluator.gen_dir.exists()
    assert "ssim" in evaluator.metrics


# ============================================================
# Test 2: Metric Computation
# ============================================================
def test_compute_metrics_returns_values(temp_dataset):
    """_compute_metrics should return numeric metric results."""
    evaluator = Evaluator(
        orig_dir=temp_dataset["orig_dir"],
        gen_dir=temp_dataset["gen_dir"],
        metric_dir=temp_dataset["metric_dir"],
        metrics=["ssim", "l1", "edge_iou"],
        yolo_model="./dummy.pt",
        imgsz=416,
    )

    img = np.full((32, 32, 3), 127, dtype=np.uint8)
    result = evaluator._compute_metrics(img, img)

    assert isinstance(result, dict)
    assert all(k in result for k in ["SSIM", "L1", "Edge_IoU"])
    assert all(isinstance(v, float) for v in result.values())


# ============================================================
# Test 3: Full Image Evaluation
# ============================================================
def test_evaluate_full_images_creates_csv(temp_dataset):
    """Full-image evaluation should produce valid CSV output."""
    evaluator = Evaluator(
        orig_dir=temp_dataset["orig_dir"],
        gen_dir=temp_dataset["gen_dir"],
        metric_dir=temp_dataset["metric_dir"],
        metrics=["ssim", "l1"],
        yolo_model="./dummy.pt",
        imgsz=416,
    )

    save_path = temp_dataset["metric_dir"] / "metrics_full_image.csv"
    avg = evaluator.evaluate_full_images(save_path)

    assert save_path.exists(), "CSV file not created."
    assert isinstance(avg, dict)
    df = pd.read_csv(save_path)
    assert not df.empty, "Metrics CSV is empty."


# ============================================================
# Test 4: YOLO Crop Evaluation
# ============================================================
@patch("src.annotation_cleaner.evaluate.YOLO")
def test_evaluate_with_yolo_crop_uses_tempdir(mock_yolo, temp_dataset):
    """YOLO crop evaluation should execute correctly with a mocked model."""
    mock_pred = MagicMock()
    mock_pred.boxes.xyxy = torch.tensor([[0, 0, 16, 16]])
    mock_yolo.return_value.predict.return_value = [mock_pred]

    evaluator = Evaluator(
        orig_dir=temp_dataset["orig_dir"],
        gen_dir=temp_dataset["gen_dir"],
        metric_dir=temp_dataset["metric_dir"],
        metrics=["ssim", "l1"],
        yolo_model="./dummy.pt",
        imgsz=416,
    )

    save_path = temp_dataset["metric_dir"] / "metrics_yolo_crop.csv"
    avg = evaluator.evaluate_with_yolo_crop(save_path)

    assert isinstance(avg, dict)
    assert save_path.exists(), "CSV file not generated for YOLO crop metrics."

    # Ensure no leftover folders are created under gen_dir
    assert not (evaluator.gen_dir / "crops").exists()
    assert not (evaluator.gen_dir / "bboxes").exists()
