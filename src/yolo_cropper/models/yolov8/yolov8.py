#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolov8.py
----------
This module defines the unified YOLOv8 pipeline.

Steps:
1️⃣ Train model
2️⃣ Evaluate performance
3️⃣ Generate `predict.txt`
4️⃣ Run prediction
5️⃣ Convert detections to JSON
6️⃣ Perform ROI cropping
"""


import sys
from pathlib import Path

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

        # Derived paths
        self.weight_path = self.saved_model_dir / f"{self.model_name}.pt"

        # Logging info
        self.logger.info(f"Initialized YOLO v8 Pipeline ({self.model_name.upper()})")
        self.logger.info(f" - Config path    : {self.config_path}")
        self.logger.info(f" - Training Dataset dir    : {self.train_dataset_dir}")
        self.logger.info(f" - Saved model dir: {self.weight_path}")
        self.logger.info(f" - Input dir      : {self.input_dir}")

    # --------------------------------------------------------
    # Step 1️⃣ Train
    # --------------------------------------------------------
    def step_train(self):
        self.logger.info("[STEP 1] Starting YOLO v8 training...")
        if self.weight_path.exists():
            self.logger.info(
                f"[SKIP] Found existing trained model → {self.weight_path}"
            )
            return
        trainer = YOLOv8Trainer(config=self.cfg)
        trainer.run()
        self.logger.info("Training step done")

    # --------------------------------------------------------
    # Step 2️⃣ Evaluate
    # --------------------------------------------------------
    def step_evaluate(self):
        self.logger.info("[STEP 2] Evaluation starts")
        evaluator = YOLOv8Evaluator(config=self.cfg)
        metrics = evaluator.run()
        self.logger.info("Evaluation step done")
        return metrics

    # --------------------------------------------------------
    # Step 3️⃣ Make predict.txt
    # --------------------------------------------------------
    def step_make_predict(self):
        self.logger.info("[STEP 4] Generating predict.txt")
        maker = YOLOPredictListGenerator(config=self.cfg)  # config-driven
        maker.run()
        self.logger.info("predict.txt generated")

    # --------------------------------------------------------
    # Step 4️⃣ Predict (auto multi-folder)
    # --------------------------------------------------------
    def step_predict(self):
        self.logger.info("[STEP 3] Preparing dataset for Darknet...")
        predictor = YOLOv8Predictor(config=self.cfg)
        predictor.run()
        self.logger.info("Prediction step done")

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
        self.logger.info("Running YOLOv8 Pipeline")
        self.step_train()
        # metrics = self.step_evaluate()
        self.step_make_predict()
        self.step_predict()
        self.step_converter()
        self.step_cropper()
        self.logger.info("\n🎉 YOLOv8 pipeline completed successfully!")
        # return metrics


# --------------------------------------------------------
# CLI Entry Point
# --------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO v8 Unified Pipeline Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="utils/config.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    pipeline = YOLOv8Pipeline(config_path=args.config)
    pipeline.run()
