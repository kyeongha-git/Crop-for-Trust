#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_yolo_cropper_controller.py
-------------------------------
Unit tests for `YOLOCropperController`, the unified dispatcher
for YOLO-based detection pipelines.

Purpose:
    - Verify that the controller correctly dispatches to the appropriate YOLO pipeline
       (DarknetPipeline, YOLOv5Pipeline, YOLOv8Pipeline) based on `model_name`.
    - Confirm that unsupported model names raise a ValueError.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.yolo_cropper.yolo_cropper import YOLOCropperController

# ==============================================================
# 🔹 Fixture: Base Configuration
# ==============================================================


@pytest.fixture
def base_config(tmp_path):
    """
    Provide a minimal fake configuration dictionary with a default YOLO model name.

    """
    return {
        "yolo_cropper": {
            "main": {
                "model_name": "yolov5",
            }
        }
    }


# ==============================================================
# Case 1 — YOLOv2 / YOLOv4 → DarknetPipeline
# ==============================================================


@pytest.mark.parametrize("model_name", ["yolov2", "yolov4"])
def test_controller_dispatches_darknet_pipeline(model_name, base_config):
    """
    Test that YOLOv2 and YOLOv4 models are correctly dispatched to the DarknetPipeline.

    Verifies:
        - Correct import path for the Darknet pipeline
        - The pipeline is instantiated and executed
        - Returned metrics contain mAP@0.5
    """
    base_config["yolo_cropper"]["main"]["model_name"] = model_name

    with patch(
        "src.yolo_cropper.yolo_cropper.load_yaml_config", return_value=base_config
    ), patch("src.yolo_cropper.yolo_cropper.importlib.import_module") as mock_import:

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value.run.return_value = {"mAP@0.5": 0.75}
        mock_import.return_value = MagicMock(DarknetPipeline=mock_pipeline_cls)

        controller = YOLOCropperController(config_path="dummy.yaml")
        metrics = controller.run()

        mock_import.assert_called_once_with("src.yolo_cropper.models.darknet.darknet")
        mock_pipeline_cls.assert_called_once()
        assert "mAP@0.5" in metrics
        print(f"Darknet dispatch success ({model_name}) → {metrics}")


# ==============================================================
# Case 2 — YOLOv5 → YOLOv5Pipeline
# ==============================================================


def test_controller_dispatches_yolov5_pipeline(base_config):
    """
    Test that YOLOv5 is correctly dispatched to the YOLOv5Pipeline.

    Verifies:
        - Import path for YOLOv5 pipeline
        - Successful instantiation and run call
        - Return dictionary includes 'precision' key
    """
    base_config["yolo_cropper"]["main"]["model_name"] = "yolov5"

    with patch(
        "src.yolo_cropper.yolo_cropper.load_yaml_config", return_value=base_config
    ), patch("src.yolo_cropper.yolo_cropper.importlib.import_module") as mock_import:

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value.run.return_value = {"precision": 0.9}
        mock_import.return_value = MagicMock(YOLOv5Pipeline=mock_pipeline_cls)

        controller = YOLOCropperController(config_path="dummy.yaml")
        metrics = controller.run()

        mock_import.assert_called_once_with("src.yolo_cropper.models.yolov5.yolov5")
        mock_pipeline_cls.assert_called_once()
        assert "precision" in metrics
        print(f"YOLOv5 dispatch success → {metrics}")


# ==============================================================
# Case 3 — YOLOv8 (s/m/l/x) → YOLOv8Pipeline
# ==============================================================


@pytest.mark.parametrize("model_name", ["yolov8s", "yolov8m", "yolov8l", "yolov8x"])
def test_controller_dispatches_yolov8_pipeline(model_name, base_config):
    """
    Test that all YOLOv8 variants are dispatched to the YOLOv8Pipeline.

    Verifies:
        - Correct dynamic import path based on model name
        - Pipeline instantiation and execution
        - Returned metrics include mAP@0.5
    """
    base_config["yolo_cropper"]["main"]["model_name"] = model_name

    with patch(
        "src.yolo_cropper.yolo_cropper.load_yaml_config", return_value=base_config
    ), patch("src.yolo_cropper.yolo_cropper.importlib.import_module") as mock_import:

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value.run.return_value = {"mAP@0.5": 0.88}
        mock_import.return_value = MagicMock(YOLOv8Pipeline=mock_pipeline_cls)

        controller = YOLOCropperController(config_path="dummy.yaml")
        metrics = controller.run()

        mock_import.assert_called_once_with("src.yolo_cropper.models.yolov8.yolov8")
        mock_pipeline_cls.assert_called_once()
        assert "mAP@0.5" in metrics
        print(f"YOLOv8 dispatch success ({model_name}) → {metrics}")


# ==============================================================
# Case 4 — Unsupported model_name
# ==============================================================


def test_controller_raises_for_invalid_model(base_config):
    """
    Test that unsupported YOLO model names raise a ValueError.

    Example:
        model_name = "yoloX99"

    Expected:
        YOLOCropperController.run() → raises ValueError
    """
    base_config["yolo_cropper"]["main"]["model_name"] = "yoloX99"

    with patch(
        "src.yolo_cropper.yolo_cropper.load_yaml_config", return_value=base_config
    ):
        with pytest.raises(ValueError):
            controller = YOLOCropperController(config_path="dummy.yaml")
            controller.run()
