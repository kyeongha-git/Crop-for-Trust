#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_darknet.py
---------------
Lightweight smoke test for the DarknetPipeline module.

Purpose:
    - Verify that `DarknetPipeline.run()` executes without raising any exceptions.
    - Confirm that it returns a valid path string pointing to a `result.json` file.
    - All heavy dependencies (trainer, evaluator, etc.) are mocked out for isolation.

This test does NOT validate internal behaviors or file I/O,
only the high-level control flow and return type.
"""

from unittest.mock import patch

import pytest

from src.yolo_cropper.models.darknet.darknet import DarknetPipeline

# ==============================================================
# Fixture: Mock Configuration
# ==============================================================
@pytest.fixture
def mock_config(tmp_path):
    """
    Create a minimal fake configuration dictionary mimicking a `config.yaml` file.

    """
    saved_model_dir = tmp_path / "saved_model" / "yolo_cropper"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "yolo_cropper": {
            "main": {
                "model_name": "yolov4",
                "input_dir": str(tmp_path / "data" / "original"),
            },
            "darknet": {
                "darknet_dir": str(tmp_path / "third_party" / "darknet"),
            },
            "dataset": {
                "saved_model_dir": str(saved_model_dir),
                "train_data_dir": str(tmp_path / "data" / "yolo_cropper"),
            },
        }
    }


# ==============================================================
# Smoke Test
# ==============================================================
def test_darknet_pipeline_runs_without_errors(tmp_path, mock_config):
    """
    Verify that `DarknetPipeline.run()` executes without errors.

    This test:
    - Mocks out all heavy submodules (MakeManager, Trainer, Evaluator, etc.)
    - Confirms that `run()` returns a valid `result.json` path string
    - Does NOT depend on actual Darknet binaries, datasets, or model files

    """

    # Patch all heavy dependencies in DarknetPipeline to prevent side effects
    with patch(
        "src.yolo_cropper.models.darknet.darknet.load_yaml_config",
        return_value=mock_config,
    ), patch("src.yolo_cropper.models.darknet.darknet.CfgManager"), patch(
        "src.yolo_cropper.models.darknet.darknet.MakeManager"
    ), patch(
        "src.yolo_cropper.models.darknet.darknet.DarknetDataPreparer"
    ), patch(
        "src.yolo_cropper.models.darknet.darknet.DarknetTrainer"
    ), patch(
        "src.yolo_cropper.models.darknet.darknet.DarknetEvaluator"
    ), patch(
        "src.yolo_cropper.models.darknet.darknet.DarknetPredictor"
    ) as MockPred, patch(
        "src.yolo_cropper.models.darknet.darknet.YOLOCropper"
    ):

        # Mock the predictor to return only output paths
        MockPred.return_value.run.return_value = (
            "outputs/json_results/yolov4/result.json",
            "outputs/json_results/predict.txt",
        )

        # Execute pipeline
        pipeline = DarknetPipeline(config_path="dummy_config.yaml")
        result = pipeline.run()

        # Assertions
        assert isinstance(
            result, str
        ), "Pipeline should return a result.json path string"
        assert result.endswith("result.json"), "Returned path must point to result.json"

        print(f"DarknetPipeline smoke test passed → {result}")
