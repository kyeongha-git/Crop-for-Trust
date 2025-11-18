#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_yolov8.py
-----------------------------
Lightweight smoke test for the `YOLOv8Pipeline` module.

Purpose:
    - Verify that `YOLOv8Pipeline.run()` executes successfully without raising exceptions,
       regardless of whether internal steps are commented out or skipped.
    - All heavy operations (training, evaluation, prediction, etc.) are mocked out.
    - This test does NOT verify model performance or internal computation.

Success Criteria:
    - No exceptions raised during pipeline execution.
    - Return value is either `None` or a dictionary.
"""

from unittest.mock import patch

import pytest

from src.yolo_cropper.models.yolov8.yolov8 import YOLOv8Pipeline

# ==============================================================
# Fixture: Mock Configuration
# ==============================================================


@pytest.fixture
def mock_yolov8_config(tmp_path):
    """
    Create a minimal fake configuration dictionary for YOLOv8.

    This mimics the expected `config.yaml` structure, including paths for
    model directories, dataset locations, and input data.
    """
    saved_model_dir = tmp_path / "saved_model" / "yolo_cropper"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "yolo_cropper": {
            "main": {
                "model_name": "yolov8s",
                "input_dir": str(tmp_path / "data" / "original"),
            },
            "yolov8": {
                "yolov8_dir": str(tmp_path / "third_party" / "yolov8"),
            },
            "dataset": {
                "saved_model_dir": str(saved_model_dir),
                "base_dir": str(tmp_path / "data" / "yolo_cropper"),
                "input_dir": str(tmp_path / "data" / "original"),
            },
        }
    }


# ==============================================================
# Smoke Test: YOLOv8 Pipeline
# ==============================================================


def test_yolov8_pipeline_runs_without_errors(tmp_path, mock_yolov8_config):
    """
    Verify that `YOLOv8Pipeline.run()` executes without raising exceptions.

    This smoke test ensures:
        - No runtime errors occur even if internal stages are commented out.
        - Return type can be `None` or a dictionary depending on mock setup.
        - Heavy submodules (training, evaluation, prediction) are fully mocked.
    """

    # Patch all heavy dependencies with lightweight mocks
    with patch(
        "src.yolo_cropper.models.yolov8.yolov8.load_yaml_config",
        return_value=mock_yolov8_config,
    ), patch("src.yolo_cropper.models.yolov8.yolov8.YOLOv8Trainer"), patch(
        "src.yolo_cropper.models.yolov8.yolov8.YOLOv8Evaluator"
    ), patch(
        "src.yolo_cropper.models.yolov8.yolov8.YOLOv8Predictor"
    ), patch(
        "src.yolo_cropper.models.yolov8.yolov8.YOLOPredictListGenerator"
    ), patch(
        "src.yolo_cropper.models.yolov8.yolov8.YOLOConverter"
    ), patch(
        "src.yolo_cropper.models.yolov8.yolov8.YOLOCropper"
    ):

        # Initialize the pipeline
        pipeline = YOLOv8Pipeline(config_path="dummy_config.yaml")

        # Run pipeline safely
        result = None
        try:
            result = pipeline.run()
        except Exception as e:
            pytest.fail(f"YOLOv8Pipeline.run() raised an unexpected exception: {e}")

        # Validate result
        assert result is None or isinstance(
            result, dict
        ), "YOLOv8Pipeline should complete successfully (return None or dict)"

        print(f"YOLOv8Pipeline smoke test passed → result: {result}")
