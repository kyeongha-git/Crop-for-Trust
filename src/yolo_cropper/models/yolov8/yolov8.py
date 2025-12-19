#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified YOLOv8 Pipeline.

Orchestrates the end-to-end workflow for YOLOv8:
Training, Evaluation, Prediction, Conversion, and Cropping.
"""

import sys
from pathlib import Path
import shutil

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from src.yolo_cropper.core.converter import YOLOConverter
from src.yolo_cropper.core.cropper import YOLOCropper
from src.yolo_cropper.core.make_predict import YOLOPredictListGenerator
from src.yolo_cropper.models.yolov8.evaluate import YOLOv8Evaluator
from src.yolo_cropper.models.yolov8.predict import YOLOv8Predictor
from src.yolo_cropper.models.yolov8.train import YOLOv8Trainer
from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging


class YOLOv8Pipeline:
    """
    Manages the execution flow for the YOLOv8-based detection and cropping pipeline.
    """

    def __init__(self, config_path: str = "utils/config.yaml") -> None:
        setup_logging("logs/yolo_cropper")
        self.logger = get_logger("yolo_cropper.YOLOv8Pipeline")

        # Load Configuration
        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)

        self.global_main_cfg = self.cfg.get("main", {})
        self.demo_mode = self.global_main_cfg.get("demo", False)

        # Component configurations
        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = yolo_cropper_cfg.get("main", {})
        self.yolov8_cfg = yolo_cropper_cfg.get("yolov8", {})
        self.train_cfg = yolo_cropper_cfg.get("train", {})
        self.dataset_cfg = yolo_cropper_cfg.get("dataset", {})

        # Path setup
        self.model_name = self.main_cfg.get("model_name", "yolov8s").lower()
        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()
        self.input_dir = Path(self.main_cfg.get("input_dir", "data/original")).resolve()
        self.detect_output_dir = (
            Path(self.dataset_cfg.get("detect_output_dir", "runs/detect"))
            / self.model_name
        )

        self.weight_path = self.saved_model_dir / f"{self.model_name}.pt"

        self.logger.info(f"Initialized YOLOv8 Pipeline (Model: {self.model_name.upper()})")

    def cleanup_previous_runs(self) -> None:
        """
        Removes output directories from previous runs to ensure a clean state.
        """
        if self.detect_output_dir.exists():
            try:
                shutil.rmtree(self.detect_output_dir)
                self.logger.info(f"Cleaned up previous results: {self.detect_output_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to remove {self.detect_output_dir}: {e}")

    def step_train(self) -> None:
        """
        Executes the training module if not in demo mode.
        """
        if self.demo_mode:
            self.logger.info("Demo mode enabled: Skipping training.")
            return

        self.logger.info("[Step 1] Starting Training")
        trainer = YOLOv8Trainer(config=self.cfg)
        trainer.run()

    def step_evaluate(self) -> None:
        """
        Executes the evaluation module to assess model performance.
        """
        if self.demo_mode:
            self.logger.info("Demo mode enabled: Skipping evaluation.")
            return

        self.logger.info("[Step 2] Starting Evaluation")
        evaluator = YOLOv8Evaluator(config=self.cfg)
        evaluator.run()

    def step_make_predict(self) -> None:
        """
        Generates the list of images to be used for prediction.
        """
        self.logger.info("[Step 3] Generating prediction list")
        maker = YOLOPredictListGenerator(config=self.cfg)
        maker.run()

    def step_predict(self) -> None:
        """
        Runs inference on the target images.
        """
        self.logger.info("[Step 4] Running Inference")
        predictor = YOLOv8Predictor(config=self.cfg)
        predictor.run()

    def step_converter(self) -> None:
        """
        Converts raw YOLO detection results into standardized JSON format.
        """
        self.logger.info("[Step 5] Converting detections to JSON")
        conv = YOLOConverter(config=self.cfg)
        conv.run()

    def step_cropper(self) -> None:
        """
        Crops images based on the detection coordinates in the JSON results.
        """
        self.logger.info("[Step 6] Cropping images")
        cropper = YOLOCropper(config=self.cfg)
        cropper.run()

    def run(self, save_image: bool) -> None:
        """
        Executes the full pipeline sequence.

        Args:
            save_image (bool): If True, proceeds to crop images after detection.
        """
        self.logger.info("===== Starting YOLOv8 Pipeline =====")
        
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

        self.logger.info("YOLOv8 pipeline completed successfully.")