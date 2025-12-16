#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolov8.py
----------
This module defines the unified YOLOv8 pipeline.

Steps:
1. Train model
2. Evaluate performance
3. Generate `predict.txt`
4. Run prediction
5. Convert detections to JSON
6. Perform ROI cropping
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
# === YOLO Submodules ===
from src.yolo_cropper.models.yolov8.train import YOLOv8Trainer
from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging


class YOLOv8Pipeline:
    """
    Unified YOLOv8 pipeline orchestrator.

    """

    def __init__(self, config_path: str = "utils/config.yaml"):
        """
        Initialize the YOLOv8 pipeline.

        Args:
            config_path (str): Path to the YAML configuration file defining
                model, dataset, and runtime parameters.
        """
        setup_logging("logs/yolo_cropper")
        self.logger = get_logger("yolo_cropper.YOLOv8Pipeline")

        # --------------------------------------------------------
        # Load Configuration
        # --------------------------------------------------------
        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)

        self.global_main_cfg = self.cfg.get("main", {})
        self.demo_mode = self.global_main_cfg.get("demo", False)

        # Shortcut configs
        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = yolo_cropper_cfg.get("main", {})
        self.yolov8_cfg = yolo_cropper_cfg.get("yolov8", {})
        self.train_cfg = yolo_cropper_cfg.get("train", {})
        self.dataset_cfg = yolo_cropper_cfg.get("dataset", {})

        # Paths
        self.model_name = self.main_cfg.get("model_name", "yolov8s").lower()
        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()
        self.train_dataset_dir = Path(
            f"{self.yolov8_cfg.get('data_yaml', 'data/yolo_cropper/yolov8/data.yaml')}"
        ).resolve()
        self.input_dir = Path(self.main_cfg.get("input_dir", "data/original")).resolve()
        self.detect_output_dir = Path(
            self.dataset_cfg.get("detect_output_dir", "runs/detect")
        ).resolve()

        # Derived paths
        self.weight_path = self.saved_model_dir / f"{self.model_name}.pt"

        # Logging info
        self.logger.info(f"Initialized YOLOv8 Pipeline ({self.model_name.upper()})")
        self.logger.info(f" - Demo Mode      : {self.demo_mode}")
        self.logger.info(f" - Config path    : {self.config_path}")
        self.logger.info(f" - Input dir      : {self.input_dir}")
        self.logger.info(f" - Saved model dir: {self.weight_path}")


    # --------------------------------------------------------
    # Step 0. Cleanup Previous Results
    # --------------------------------------------------------
    def cleanup_previous_runs(self):
        if self.detect_output_dir.exists():
            self.logger.warning(f"[CleanUp] Removing previous run results: {self.detect_output_dir}")
            try:
                shutil.rmtree(self.detect_output_dir)
            except Exception as e:
                self.logger.warning(f"[CleanUp] Failed to remove {self.detect_output_dir}: {e}")

    # --------------------------------------------------------
    # Step 1. Train
    # --------------------------------------------------------
    def step_train(self):
        if self.demo_mode:
            self.logger.info("[STEP 1] Demo mode → Skipping training")
            return

        self.logger.info("[STEP 1] Starting YOLO v8 training...")

        trainer = YOLOv8Trainer(config=self.cfg)
        trainer.run()

    # --------------------------------------------------------
    # Step 2. Evaluate
    # --------------------------------------------------------
    def step_evaluate(self):
        if self.demo_mode:
            self.logger.info("[STEP 2] Demo mode → Skipping evaluation.")
            return None

        self.logger.info("[STEP 2] Evaluating YOLOv8 model...")
        evaluator = YOLOv8Evaluator(config=self.cfg)
        evaluator.run()
        
    # --------------------------------------------------------
    # Step 3. Make predict.txt
    # --------------------------------------------------------
    def step_make_predict(self):
        self.logger.info("[STEP 3] Generating predict.txt")
        maker = YOLOPredictListGenerator(config=self.cfg)
        maker.run()

    # --------------------------------------------------------
    # Step 4. Predict
    # --------------------------------------------------------
    def step_predict(self):
        self.logger.info("[STEP 4] Running YOLOv8 prediction...")
        predictor = YOLOv8Predictor(config=self.cfg)
        predictor.run()

    # --------------------------------------------------------
    # Step 5. Converter
    # --------------------------------------------------------
    def step_converter(self):
        self.logger.info("[STEP 5] Converting YOLOv8 detects → result.json")
        conv = YOLOConverter(config=self.cfg)
        conv.run()

    # -------------------------------------------------
    # Step 6. Cropper
    # -------------------------------------------------
    def step_cropper(self):
        self.logger.info("[STEP 6] Cropping from result.json")
        cropper = YOLOCropper(config=self.cfg)
        cropper.run()

    # --------------------------------------------------------
    # Entrypoint
    # --------------------------------------------------------
    def run(self):
        self.logger.info("Running YOLOv8 Pipeline")
        self.cleanup_previous_runs()
        self.step_train()
        self.step_evaluate()
        self.step_make_predict()
        self.step_predict()
        self.step_converter()
        self.step_cropper()
        self.logger.info("\nYOLOv8 pipeline completed successfully!")
