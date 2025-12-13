#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolo_cropper.py
---------------
Unified YOLO Pipeline Controller

This module synchronizes config.yaml via ConfigManager
and dispatches the appropriate YOLO pipeline.
"""

import importlib
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging
from utils.config_manager import ConfigManager


class YOLOCropperController:
    """
    Unified YOLO Cropper Controller.
    """

    def __init__(self, config_path: str = "utils/config.yaml"):
        setup_logging("logs/yolo_cropper")
        self.logger = get_logger("yolo_cropper.Controller")

        self.config_path = Path(config_path)

        # ======================================================
        # 1. Synchronize & overwrite config.yaml
        # ======================================================
        self.logger.info("Synchronizing config.yaml via ConfigManager")

        cfg_manager = ConfigManager(str(self.config_path))
        updated_cfg = cfg_manager.update_paths()
        cfg_manager.save()  # overwrite config.yaml

        # ======================================================
        # 2. Reload synchronized config
        # ======================================================
        self.cfg = load_yaml_config(self.config_path)
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})

        # Single source of truth
        self.model_name = self.main_cfg.get("model_name", "yolov5").lower()

        self.logger.info(
            f"Initialized YOLO Cropper Controller ({self.model_name.upper()})"
        )

    # --------------------------------------------------
    def run(self):
        """
        Dispatch the appropriate YOLO pipeline based on model_name.
        """
        if self.model_name.startswith("yolov8"):
            module_path = "src.yolo_cropper.models.yolov8.yolov8"
            class_name = "YOLOv8Pipeline"
        elif self.model_name in ("yolov2", "yolov4"):
            module_path = "src.yolo_cropper.models.darknet.darknet"
            class_name = "DarknetPipeline"
        elif self.model_name.startswith("yolov5"):
            module_path = "src.yolo_cropper.models.yolov5.yolov5"
            class_name = "YOLOv5Pipeline"
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

        self.logger.info(f"Loading pipeline → {module_path}.{class_name}")

        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            raise ImportError(f"Failed to import module {module_path}: {e}")

        if not hasattr(module, class_name):
            raise AttributeError(
                f"{module_path} does not define class '{class_name}'."
            )

        pipeline_class = getattr(module, class_name)
        pipeline = pipeline_class(config_path=str(self.config_path))

        self.logger.info(f"Running {self.model_name.upper()} pipeline...")
        metrics = pipeline.run()
        self.logger.info(f"Pipeline complete ({self.model_name.upper()})")

        return metrics


# ======================================================
# Standalone Entrypoint
# ======================================================
def main():
    """
    Standalone entrypoint for YOLO Cropper Controller.

    Example:
        python src/yolo_cropper/yolo_cropper.py --config utils/config.yaml
    """
    parser = argparse.ArgumentParser(
        description="Standalone YOLO Cropper Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="utils/config.yaml",
        help="Path to config.yaml",
    )

    args = parser.parse_args()

    setup_logging("logs/yolo_cropper")
    logger = get_logger("yolo_cropper.main")

    logger.info("Starting standalone YOLO Cropper execution")
    logger.info(f"Using config: {args.config}")

    try:
        controller = YOLOCropperController(config_path=args.config)
        controller.run()
        logger.info("Standalone YOLO Cropper finished successfully")
    except Exception:
        logger.exception("YOLO Cropper execution failed")
        raise


if __name__ == "__main__":
    main()
