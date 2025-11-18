#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_yolov5.py
-----------------------------
Lightweight smoke test for the `YOLOv5Pipeline` module.

Purpose:
    - Verify that `YOLOv5Pipeline.run()` executes without raising any exceptions,
       regardless of internal step comments or skipped processes.
    - All heavy components (trainer, evaluator, predictor, converter, etc.) are mocked out.
    - This test does NOT verify accuracy, model training, or evaluation logic.

Success criteria:
    - No runtime errors are raised during pipeline execution.
    - The return type is either `None` or a dictionary (mock-compatible).
"""

from unittest.mock import patch

import pytest

from src.yolo_cropper.models.yolov5.yolov5 import YOLOv5Pipeline

# ==============================================================
# Fixture: Mock Configuration
# ==============================================================


@pytest.fixture
def mock_yolov5_config(tmp_path):
    """
    Create a minimal fake configuration dictionary for YOLOv5.

    This configuration mimics the structure of a real `config.yaml`
    and provides all required directory paths for the pipeline.

    """
    saved_model_dir = tmp_path / "saved_model" / "yolo_cropper"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "yolo_cropper": {
            "main": {
                "model_name": "yolov5",
                "input_dir": str(tmp_path / "data" / "original"),
            },
            "yolov5": {
                "yolov5_dir": str(tmp_path / "third_party" / "yolov5"),
            },
            "dataset": {
                "saved_model_dir": str(saved_model_dir),
                "train_data_dir": str(tmp_path / "data" / "yolo_cropper"),
                "input_dir": str(tmp_path / "data" / "original"),
            },
        }
    }


# ==============================================================
# Smoke Test: YOLOv5 Pipeline
# ==============================================================


def test_yolov5_pipeline_runs_without_errors(tmp_path, mock_yolov5_config):
    """
    Verify that `YOLOv5Pipeline.run()` executes successfully without raising exceptions.

    This is a smoke test:
        - Heavy dependencies such as training, evaluation, and prediction are mocked.
        - Only the pipeline's control flow and return behavior are tested.
        - The return value may be `None` or a dictionary depending on mocks.

    """

    # Patch all heavy components of YOLOv5Pipeline to lightweight mocks
    with patch(
        "src.yolo_cropper.models.yolov5.yolov5.load_yaml_config",
        return_value=mock_yolov5_config,
    ), patch("src.yolo_cropper.models.yolov5.yolov5.YOLOv5Trainer"), patch(
        "src.yolo_cropper.models.yolov5.yolov5.YOLOv5Evaluator"
    ), patch(
        "src.yolo_cropper.models.yolov5.yolov5.YOLOv5Predictor"
    ), patch(
        "src.yolo_cropper.models.yolov5.yolov5.YOLOPredictListGenerator"
    ), patch(
        "src.yolo_cropper.models.yolov5.yolov5.YOLOConverter"
    ), patch(
        "src.yolo_cropper.models.yolov5.yolov5.YOLOCropper"
    ):

        # Initialize pipeline
        pipeline = YOLOv5Pipeline(config_path="dummy_config.yaml")

        # Attempt to run the pipeline (expect no exceptions)
        result = None
        try:
            result = pipeline.run()
        except Exception as e:
            pytest.fail(f"YOLOv5Pipeline.run() raised an unexpected exception: {e}")

        # Assertions — should complete successfully
        assert result is None or isinstance(
            result, dict
        ), "YOLOv5Pipeline should complete successfully (return None or dict)"

        print(f"YOLOv5Pipeline smoke test passed → result: {result}")
