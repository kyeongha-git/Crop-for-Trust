#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolov5.py
----------
This module defines the configuration-driven YOLOv5 pipeline.

It automates the entire workflow — from model training and evaluation
to prediction, output conversion, and cropping — under a unified structure.

Steps:
1. Train YOLOv5
2. Evaluate trained model
3. Run predictions (multi-folder support)
4. Generate `predict.txt` for cropping
5. Convert YOLOv5 `.txt` detections to unified `result.json`
6. Perform cropping from JSON results
"""

import sys
from pathlib import Path
import shutil

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
    Full YOLOv5 workflow (train → eval → predict → convert → crop)
    with demo-mode skipping for training steps.
    """

    def __init__(self, config_path: str = "utils/config.yaml"):

        setup_logging("logs/yolo_cropper")
        self.logger = get_logger("yolo_cropper.YOLOv5Pipeline")

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
        self.yolov5_cfg = yolo_cropper_cfg.get("yolov5", {})
        self.dataset_cfg = yolo_cropper_cfg.get("dataset", {})

        # Paths
        self.model_name = self.main_cfg.get("model_name", "yolov5").lower()
        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()    
        self.input_dir = Path(self.main_cfg.get("input_dir", "data/original"))
        self.detect_output_dir = (
            Path(self.dataset_cfg.get("detect_output_dir", "runs/detect"))
            / self.model_name
        )
        
        # Logging info
        self.logger.info(f"Initialized YOLOv5 Pipeline ({self.model_name.upper()})")
        self.logger.info(f" - Demo mode      : {self.demo_mode}")
        self.logger.info(f" - Config path    : {self.config_path}")
        self.logger.info(f" - Input dir      : {self.input_dir}")


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

        self.logger.info("[STEP 1] Starting YOLO v5 training...")

        trainer = YOLOv5Trainer(config=self.cfg)
        trainer.run()

    # --------------------------------------------------------
    # Step 2. Evaluate
    # --------------------------------------------------------
    def step_evaluate(self):
        if self.demo_mode:
            self.logger.info("[STEP 2] Demo mode → Skipping evaluation")
            return None

        self.logger.info("[STEP 2] Evaluating YOLOv5 model...")
        evaluator = YOLOv5Evaluator(config=self.cfg)
        metrics = evaluator.run()
        return metrics
    
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
        self.logger.info("[STEP 4] Running YOLOv5 prediction...")
        predictor = YOLOv5Predictor(config=self.cfg)
        predictor.run()

    # --------------------------------------------------------
    # Step 5. Converter
    # --------------------------------------------------------
    def step_converter(self):
        self.logger.info("[STEP 5] Converting YOLOv5 detects → result.json")
        converter = YOLOConverter(config=self.cfg)
        converter.run()

    # --------------------------------------------------------
    # Step 6. Cropper
    # --------------------------------------------------------
    def step_cropper(self):
        self.logger.info("[STEP 6] Cropping from result.json")
        cropper = YOLOCropper(config=self.cfg)
        cropper.run()

    # --------------------------------------------------------
    # Entrypoint
    # --------------------------------------------------------
    def run(self, save_image):
        self.logger.info("Running YOLOv5 Pipeline")
        self.cleanup_previous_runs()
        self.step_train()
        self.step_evaluate()
        self.step_make_predict()
        self.step_predict()
        self.step_converter()
        if save_image:
            self.step_cropper()
        else:
            self.logger.info("[save_image: False] → Cropper Skip.")

        self.logger.info("\nYOLOv5 pipeline completed successfully!")