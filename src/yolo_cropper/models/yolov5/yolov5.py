#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolov5.py
----------
This module defines the configuration-driven YOLOv5 pipeline.

It automates the entire workflow — from model training and evaluation
to prediction, output conversion, and cropping — under a unified structure.

Steps:
1️⃣ Train YOLOv5
2️⃣ Evaluate trained model
3️⃣ Run predictions (multi-folder support)
4️⃣ Generate `predict.txt` for cropping
5️⃣ Convert YOLOv5 `.txt` detections to unified `result.json`
6️⃣ Perform cropping from JSON results
"""

import sys
from pathlib import Path

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

        main_cfg = self.cfg.get("main", {})
        self.demo_mode = str(main_cfg.get("demo", "off")).lower()

        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = yolo_cropper_cfg.get("main", {})
        self.yolov5_cfg = yolo_cropper_cfg.get("yolov5", {})
        self.dataset_cfg = yolo_cropper_cfg.get("dataset", {})

        self.model_name = self.main_cfg.get("model_name", "yolov5").lower()
        self.saved_model_dir = Path(self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")).resolve()
        self.saved_weight_path = self.saved_model_dir / f"{self.model_name}.pt"

        self.input_dir = Path(self.main_cfg.get("input_dir", "data/original")).resolve()
        self.train_dataset_dir = Path(self.yolov5_cfg.get('data_yaml', 'data/yolo_cropper/yolov5/data.yaml')).resolve()

        self.logger.info(f"Initialized YOLO v5 Pipeline ({self.model_name.upper()})")
        self.logger.info(f"Demo mode         : {self.demo_mode}")
        self.logger.info(f"Training Data     : {self.train_dataset_dir}")
        self.logger.info(f"Saved weights     : {self.saved_weight_path}")
        self.logger.info(f"Input dir         : {self.input_dir}")

    # --------------------------------------------------------
    # Step 1. Train
    # --------------------------------------------------------
    def step_train(self):
        if self.demo_mode == "on":
            self.logger.info("[STEP 1] Demo mode ON → Skipping training")
            return

        self.logger.info("[STEP 1] Starting YOLO v5 training...")
        if self.saved_weight_path.exists():
            self.logger.info(f"[SKIP] Existing trained model: {self.saved_weight_path}")
            return

        trainer = YOLOv5Trainer(config=self.cfg)
        trainer.run()
        self.logger.info("Training complete")

    # --------------------------------------------------------
    # Step 2. Evaluate
    # --------------------------------------------------------
    def step_evaluate(self):
        if self.demo_mode == "on":
            self.logger.info("[STEP 2] Demo mode ON → Skipping evaluation")
            return None

        self.logger.info("[STEP 2] Running evaluation...")
        evaluator = YOLOv5Evaluator(config=self.cfg)
        metrics = evaluator.run()
        return metrics

    # --------------------------------------------------------
    # Step 3. Predict
    # --------------------------------------------------------
    def step_predict(self):
        self.logger.info("[STEP 3] Prediction…")
        predictor = YOLOv5Predictor(config=self.cfg)
        predictor.run()

    # --------------------------------------------------------
    # Step 4. Make predict.txt
    # --------------------------------------------------------
    def step_make_predict(self):
        self.logger.info("[STEP 4] Generating predict.txt")
        maker = YOLOPredictListGenerator(config=self.cfg)
        maker.run()

    # --------------------------------------------------------
    # Step 5. Convert to result.json
    # --------------------------------------------------------
    def step_converter(self):
        self.logger.info("[STEP 5] Converting YOLO outputs → result.json")
        conv = YOLOConverter(config=self.cfg)
        conv.run()

    # --------------------------------------------------------
    # Step 6. Crop from JSON
    # --------------------------------------------------------
    def step_cropper(self):
        self.logger.info("[STEP 6] Cropping from result.json")
        cropper = YOLOCropper(config=self.cfg)
        cropper.crop_from_json()

    # --------------------------------------------------------
    # Entrypoint
    # --------------------------------------------------------
    def run(self):
        self.logger.info("Running YOLOv5 Pipeline")
        self.step_train()
        self.step_evaluate()
        self.step_predict()
        self.step_make_predict()
        self.step_converter()
        self.step_cropper()

        self.logger.info("YOLOv5 pipeline completed successfully!")