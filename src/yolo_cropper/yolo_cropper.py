#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolo_cropper.py
---------------
Unified YOLO Pipeline Controller

This module serves as the central controller for all YOLO-based
pipelines (YOLOv2, YOLOv4, YOLOv5, YOLOv8). It automatically
detects the target model type from the configuration file and
dispatches execution to the corresponding pipeline class.

Supported Pipelines:
    - YOLOv2 / YOLOv4 → DarknetPipeline
    - YOLOv5          → YOLOv5Pipeline
    - YOLOv8 (s/m/l/x) → YOLOv8Pipeline
"""

import importlib
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging


class YOLOCropperController:
    """
    Unified YOLO Cropper Controller.

    """

    def __init__(self, config_path: str = "utils/config.yaml"):
        """
        Initialize the unified YOLO Cropper controller.

        Args:
            config_path (str): Path to the configuration YAML file.
                Must contain a `yolo_cropper.main.model_name` entry
                specifying the YOLO version to use.
        """
        setup_logging("logs/yolo_cropper")
        self.logger = get_logger("yolo_cropper.Controller")

        # Load configuration
        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})

        # Model name resolution
        self.model_name = self.main_cfg.get("model_name", "yolov5").lower()
        self.logger.info(
            f"Initialized YOLO Cropper Controller ({self.model_name.upper()})"
        )

    def run(self):
        """
        Dispatch the appropriate YOLO pipeline based on `model_name`.

        Returns:
            dict | None: Evaluation metrics dictionary (if applicable)
            or None if the pipeline does not return metrics.

        """
        # Handle YOLO version mapping
        if self.model_name.startswith("yolov8"):
            module_path = "src.yolo_cropper.models.yolov8.yolov8"
            class_name = "YOLOv8Pipeline"
        elif self.model_name in ("yolov2", "yolov4"):
            module_path = "src.yolo_cropper.models.darknet.darknet"
            class_name = "DarknetPipeline"
        elif self.model_name == "yolov5":
            module_path = "src.yolo_cropper.models.yolov5.yolov5"
            class_name = "YOLOv5Pipeline"
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

        self.logger.info(f"Loading pipeline → {module_path}.{class_name}")

        # Dynamic import
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            raise ImportError(f"Failed to import module {module_path}: {e}")

        # Validate class existence
        if not hasattr(module, class_name):
            raise AttributeError(
                f"{module_path} does not define class '{class_name}'."
            )

        # Instantiate and run pipeline
        pipeline_class = getattr(module, class_name)
        pipeline = pipeline_class(config_path=str(self.config_path))

        self.logger.info(f"Running {self.model_name.upper()} pipeline...")
        metrics = pipeline.run()
        self.logger.info(f"Pipeline complete ({self.model_name.upper()})")

        return metrics


# --------------------------------------------------------
# CLI Entry Point
# --------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified YOLO Cropper Controller")
    parser.add_argument(
        "--config",
        type=str,
        default="utils/config.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()

    controller = YOLOCropperController(config_path=args.config)
    controller.run()
