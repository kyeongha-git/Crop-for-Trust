#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified YOLOv5 Pipeline.

Orchestrates the end-to-end workflow for YOLOv5:
Training, Evaluation, Prediction, Conversion, and Cropping.
"""

import sys
from pathlib import Path
import shutil
from typing import Optional, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from src.yolo_cropper.core.converter import YOLOConverter
from src.yolo_cropper.core.cropper import YOLOCropper
from src.yolo_cropper.core.make_predict import YOLOPredictListGenerator
from src.yolo_cropper.models.yolov5.evaluate import YOLOv5Evaluator
from src.yolo_cropper.models.yolov5.predict import YOLOv5Predictor
from src.yolo_cropper.models.yolov5.train import YOLOv5Trainer
from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging


class YOLOv5Pipeline:
    """
    Manages the execution flow for the YOLOv5-based detection and cropping pipeline.
    """

    def __init__(self, config_path: str = "utils/config.yaml") -> None:
        setup_logging("logs/yolo_cropper")
        self.logger = get_logger("yolo_cropper.YOLOv5Pipeline")

        # Load Configuration
        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)

        self.global_main_cfg = self.cfg.get("main", {})
        self.demo_mode = self.global_main_cfg.get("demo", False)

        # Component configurations
        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = yolo_cropper_cfg.get("main", {})
        self.yolov5_cfg = yolo_cropper_cfg.get("yolov5", {})
        self.dataset_cfg = yolo_cropper_cfg.get("dataset", {})

        # Path setup
        self.model_name = self.main_cfg.get("model_name", "yolov5").lower()
        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()

        self.input_dir = Path(self.main_cfg.get("input_dir", "data/original")).resolve()
        self.detect_root = Path(
            self.dataset_cfg.get("detect_output_dir", "runs/detect")
        ).resolve()

        self.logger.info(f"Initialized YOLOv5 Pipeline (Model: {self.model_name.upper()})")

    # --------------------------------------------------------
    def cleanup_previous_runs(self) -> None:
        """
        Removes YOLOv5 detection outputs from previous runs.

        YOLOv5 produces multiple detection directories with prefix 'yolov5_*'
        (e.g., yolov5_repair, yolov5_replace). All of them must be removed
        to ensure a clean and reproducible execution.
        """
        if not self.detect_root.exists():
            return

        removed_dirs: List[Path] = []

        for d in self.detect_root.iterdir():
            if d.is_dir() and d.name.startswith("yolov5"):
                try:
                    shutil.rmtree(d)
                    removed_dirs.append(d)
                except Exception as e:
                    self.logger.warning(f"Failed to remove {d}: {e}")

        if removed_dirs:
            self.logger.info(
                f"Cleaned up previous YOLOv5 results ({len(removed_dirs)} dirs)"
            )
            for d in removed_dirs:
                self.logger.info(f" - {d}")
        else:
            self.logger.info("No previous YOLOv5 results to clean up")

    # --------------------------------------------------------
    def step_train(self) -> None:
        if self.demo_mode:
            self.logger.info("Demo mode enabled: Skipping training.")
            return

        self.logger.info("[Step 1] Starting Training")
        trainer = YOLOv5Trainer(config=self.cfg)
        trainer.run()

    def step_evaluate(self) -> Optional[Dict]:
        if self.demo_mode:
            self.logger.info("Demo mode enabled: Skipping evaluation.")
            return None

        self.logger.info("[Step 2] Starting Evaluation")
        evaluator = YOLOv5Evaluator(config=self.cfg)
        return evaluator.run()

    def step_make_predict(self) -> None:
        self.logger.info("[Step 3] Generating prediction list")
        maker = YOLOPredictListGenerator(config=self.cfg)
        maker.run()

    def step_predict(self) -> None:
        self.logger.info("[Step 4] Running Inference")
        predictor = YOLOv5Predictor(config=self.cfg)
        predictor.run()

    def step_converter(self) -> None:
        self.logger.info("[Step 5] Converting detections to JSON")
        converter = YOLOConverter(config=self.cfg)
        converter.run()

    def step_cropper(self) -> None:
        self.logger.info("[Step 6] Cropping images")
        cropper = YOLOCropper(config=self.cfg)
        cropper.run()

    # --------------------------------------------------------
    def run(self, save_image: bool) -> None:
        """
        Executes the full pipeline sequence.

        Args:
            save_image (bool): If True, proceeds to crop images after detection.
        """
        self.logger.info("===== Starting YOLOv5 Pipeline =====")

        self.cleanup_previous_runs()
        self.step_train()
        self.step_evaluate()
        self.step_make_predict()
        self.step_predict()
        self.step_converter()

        if save_image:
            self.step_cropper()
        else:
            self.logger.info("Skipping crop step (save_image=False)")

        self.logger.info("YOLOv5 pipeline completed successfully.")
