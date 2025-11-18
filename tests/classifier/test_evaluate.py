#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_classifier_evaluator.py
----------------------------
Unit and integration tests for the `Evaluator` class in `src.classifier.evaluate`.

Covers:
1️⃣ Transform generation (preprocessing)
2️⃣ Model loading from checkpoint
3️⃣ Data loading via mock dataset
4️⃣ Metrics saving and file output
5️⃣ Full run() integration (mock-based)
"""

import json
import os
from unittest.mock import MagicMock, patch

import matplotlib
import pytest
import torch

# Enable headless backend for CI/server environments
matplotlib.use("Agg")

from src.classifier.evaluate import Evaluator


# ======================================================
# Dummy Config Fixtures
# ======================================================
@pytest.fixture
def dummy_cfg(tmp_path):
    """Provide a minimal configuration structure required by Evaluator."""
    cfg = {
        "train": {
            "save_dir": str(tmp_path / "saved_models"),
            "metric_dir": str(tmp_path / "metrics"),
        },
        "wandb": {"enabled": False},  # disable wandb logging for tests
    }
    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)
    os.makedirs(cfg["train"]["metric_dir"], exist_ok=True)
    return cfg


@pytest.fixture
def evaluator(tmp_path, dummy_cfg):
    """Instantiate Evaluator with temporary directories and dummy config."""
    input_dir = str(tmp_path / "data" / "original")
    os.makedirs(input_dir, exist_ok=True)
    ev = Evaluator(
        input_dir=input_dir,
        model="vgg16",
        cfg=dummy_cfg,  # dict-based config injection
    )
    return ev


# ======================================================
# _get_transform() Tests
# ======================================================
def test_get_transform_vgg(evaluator):
    """_get_transform(): should return valid transform pipeline for VGG."""
    transform = evaluator._get_transform()
    assert transform is not None

    ops = [t.__class__.__name__ for t in transform.transforms]
    has_aug = any("Affine" in o or "Random" in o for o in ops)
    assert not has_aug, "_get_transform() should not include random augmentations"
    print(f"_get_transform() passed — ops: {ops}")


# ======================================================
# _load_model() Tests
# ======================================================
def test_load_model_file_exists(evaluator):
    """_load_model(): should successfully load model when checkpoint exists."""
    dummy_model_path = os.path.join(evaluator.save_root, "vgg16.pt")
    os.makedirs(os.path.dirname(dummy_model_path), exist_ok=True)
    torch.save({}, dummy_model_path)

    with patch("torch.nn.Module.load_state_dict", return_value=None):
        model = evaluator._load_model()

    assert model is not None
    print("_load_model() passed")


# ======================================================
# _load_data() Tests
# ======================================================
def test_load_data_returns_loader(evaluator):
    """_load_data(): should return an iterable DataLoader object."""
    transform = evaluator._get_transform()

    with patch(
        "src.classifier.evaluate.ClassificationDataset",
        return_value=[(torch.rand(3, 360, 360), torch.tensor(1))],
    ):
        loader = evaluator._load_data(transform)

    assert loader is not None and hasattr(loader, "__iter__")
    print(f"_load_data() passed — batch count: {len(loader)}")


# ======================================================
# _save_results() Tests
# ======================================================
def test_save_results_creates_files(evaluator, tmp_path):
    """_save_results(): should save metrics.json and confusion matrix image."""
    temp_metrics_dir = tmp_path / "metrics"
    evaluator.metric_root = temp_metrics_dir
    os.makedirs(temp_metrics_dir, exist_ok=True)

    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    acc, f1 = 0.85, 0.8

    evaluator._save_results(y_true, y_pred, acc, f1)

    metrics_path = os.path.join(temp_metrics_dir, "classifier", "vgg16", "metrics.json")
    cm_path = os.path.join(temp_metrics_dir, "classifier", "vgg16", "cm.png")

    assert os.path.exists(metrics_path), f"metrics.json missing: {metrics_path}"
    assert os.path.exists(cm_path), f"confusion matrix image missing: {cm_path}"

    with open(metrics_path, "r") as f:
        data = json.load(f)
    assert "accuracy" in data and "f1_score" in data
    print("_save_results() passed (metrics written to tmp_path)")


# ======================================================
# run() Integration Tests
# ======================================================
def test_run_integration_mock(evaluator, tmp_path):
    """run(): should execute the full pipeline using mocks without errors."""
    evaluator.metric_root = tmp_path / "metrics"
    os.makedirs(evaluator.metric_root, exist_ok=True)

    with patch.object(Evaluator, "_get_transform", return_value=MagicMock()):
        with patch.object(
            Evaluator,
            "_load_data",
            return_value=[(torch.rand(1, 3, 360, 360), torch.tensor([1]))],
        ):
            with patch.object(Evaluator, "_load_model", return_value=MagicMock()):
                with patch("torch.sigmoid", return_value=torch.tensor([[0.9]])):
                    acc, f1 = evaluator.run()

    assert isinstance(acc, float)
    assert isinstance(f1, float)

    metrics_json = os.path.join(
        evaluator.metric_root, "classifier", "vgg16", "metrics.json"
    )
    assert os.path.exists(metrics_json), f"metrics.json missing: {metrics_json}"
    print(f"run() integration mock passed — ACC={acc:.4f}, F1={f1:.4f}")
