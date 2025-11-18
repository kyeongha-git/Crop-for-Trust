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
    Orchestrates the complete YOLOv5 workflow in a modular, config-driven manner.

    This pipeline integrates model training, evaluation, prediction,
    post-processing, and cropping into a unified automated process.
    Each stage can be executed independently or sequentially.

    """

    def __init__(self, config_path: str = "utils/config.yaml"):
        """
        Initialize the YOLOv5 pipeline using a configuration file.

        Args:
            config_path (str, optional): Path to the YAML configuration file.
                Defaults to "utils/config.yaml".
        """
        setup_logging("logs/yolo_cropper")
        self.logger = get_logger("yolo_cropper.YOLOv5Pipeline")

        # --------------------------------------------------------
        # Load Configuration
        # --------------------------------------------------------
        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)

        # Shortcut configs
        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = yolo_cropper_cfg.get("main", {})
        self.yolov5_cfg = yolo_cropper_cfg.get("yolov5", {})
        self.dataset_cfg = yolo_cropper_cfg.get("dataset", {})

        # Paths
        self.model_name = self.main_cfg.get("model_name", "yolov5").lower()
        self.yolov5_dir = Path(
            self.yolov5_cfg.get("yolov5_dir", "third_party/yolov5")
        ).resolve()
        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()
        self.train_dataset_dir = Path(
            f"{self.yolov5_cfg.get('data_yaml', 'data/yolo_cropper/yolov5/data.yaml')}"
        ).resolve()

        self.input_dir = Path(self.main_cfg.get("input_dir", "data/original")).resolve()

        # Derived paths
        self.saved_weight_path = self.saved_model_dir / f"{self.model_name}.pt"

        # Logging info
        self.logger.info(f"Initialized YOLO v5 Pipeline ({self.model_name.upper()})")
        self.logger.info(f" - Config path    : {self.config_path}")
        self.logger.info(f" - YOLO v5 dir    : {self.yolov5_dir}")
        self.logger.info(f" - Training Dataset dir    : {self.train_dataset_dir}")
        self.logger.info(f" - Saved model dir: {self.saved_weight_path}")
        self.logger.info(f" - Input dir      : {self.input_dir}")

    # --------------------------------------------------------
    # Step 1️⃣ Train
    # --------------------------------------------------------
    def step_train(self):
        self.logger.info("[STEP 1] Starting YOLO v5 training...")
        if self.saved_weight_path.exists():
            self.logger.info(
                f"[SKIP] Found existing trained model → {self.saved_weight_path}"
            )
            return
        trainer = YOLOv5Trainer(config=self.cfg)
        trainer.run()
        self.logger.info("Training step done")

    # --------------------------------------------------------
    # Step 2️⃣ Evaluate
    # --------------------------------------------------------
    def step_evaluate(self):
        self.logger.info("[STEP 2] Evaluation starts")
        evaluator = YOLOv5Evaluator(config=self.cfg)
        metrics = evaluator.run()
        self.logger.info("Evaluation step done")
        return metrics

    # --------------------------------------------------------
    # Step 3️⃣ Predict (auto multi-folder)
    # --------------------------------------------------------
    def step_predict(self):
        self.logger.info("[STEP 3] Preparing dataset for YOLOv5...")
        predictor = YOLOv5Predictor(config=self.cfg)
        predictor.run()
        self.logger.info("Prediction step done")

    # --------------------------------------------------------
    # Step 4️⃣ Make predict.txt
    # --------------------------------------------------------
    def step_make_predict(self):
        self.logger.info("[STEP 4] Generating predict.txt")
        maker = YOLOPredictListGenerator(config=self.cfg)  # config-driven
        maker.run()
        self.logger.info("predict.txt generated")

    # --------------------------------------------------------
    # Step 5️⃣ Converter (YOLOv5 detect → unified result.json)
    # --------------------------------------------------------
    def step_converter(self):
        self.logger.info("[STEP 5] Converting YOLOv5 detects → result.json")
        conv = YOLOConverter(config=self.cfg)  # config-driven
        conv.run()
        self.logger.info("Conversion step done")

    # -------------------------------------------------
    # Step 6️⃣ Cropper (result.json 기반 ROI crop)
    # -------------------------------------------------
    def step_cropper(self):
        self.logger.info("[STEP 6] Cropping from result.json")
        cropper = YOLOCropper(config=self.cfg)  # config-driven
        cropper.crop_from_json()
        self.logger.info("Cropping step done")

    # --------------------------------------------------------
    # Unified Runner
    # --------------------------------------------------------
    def run(self):
        self.logger.info("Running YOLOv5 Pipeline")
        # self.step_train()
        # metrics = self.step_evaluate()
        self.step_predict()
        self.step_make_predict()
        self.step_converter()
        self.step_cropper()
        self.logger.info("\n🎉 YOLOv5 pipeline completed successfully!")
        # return metrics


# --------------------------------------------------------
# CLI Entry Point
# --------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO v5 Unified Pipeline Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="utils/config.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    pipeline = YOLOv5Pipeline(config_path=args.config)
    pipeline.run()
