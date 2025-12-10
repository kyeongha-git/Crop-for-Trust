#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
darknet.py
----------
This module defines a unified Darknet-based YOLOv2/YOLOv4 pipeline.

Steps:
1️⃣ cfg_manager
2️⃣ make_manager
3️⃣ data_prepare
4️⃣ train (skip if saved_model exists)
5️⃣ evaluate
6️⃣ predict
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from src.yolo_cropper.core.cropper import YOLOCropper
from src.yolo_cropper.models.darknet.evaluate import DarknetEvaluator
from src.yolo_cropper.models.darknet.predict import DarknetPredictor
from src.yolo_cropper.models.darknet.setup.cfg_manager import CfgManager
from src.yolo_cropper.models.darknet.setup.data_prepare import \
    DarknetDataPreparer
from src.yolo_cropper.models.darknet.setup.make_manager import MakeManager
from src.yolo_cropper.models.darknet.train import DarknetTrainer
from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging


class DarknetPipeline:
    """
    A unified workflow manager for Darknet-based YOLOv2/YOLOv4 training and inference.

    It uses a single configuration file to drive the entire process.

    """

    def __init__(self, config_path: str = "utils/config.yaml"):
        """
        Initialize the Darknet pipeline and load configuration parameters.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        setup_logging("logs/yolo_cropper")
        self.logger = get_logger("yolo_cropper.DarknetPipeline")

        # --------------------------------------------------------
        # Load Configuration
        # --------------------------------------------------------
        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)

        # Shortcut configs
        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = yolo_cropper_cfg.get("main", {})
        self.darknet_cfg = yolo_cropper_cfg.get("darknet", {})
        self.dataset_cfg = yolo_cropper_cfg.get("dataset", {})

        # Paths
        self.model_name = self.main_cfg.get("model_name", "yolov4").lower()
        self.darknet_dir = Path(
            self.darknet_cfg.get("darknet_dir", "third_party/darknet")
        ).resolve()
        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()
        self.base_dataset_dir = Path(
            self.dataset_cfg.get("train_data_dir", "data/yolo_cropper")
        ).resolve()
        self.train_dataset_dir = Path(
            f"{self.dataset_cfg.get('train_data_dir', 'data/yolo_cropper')}/{self.model_name}"
        ).resolve()
        self.input_dir = Path(self.main_cfg.get("input_dir", "data/original")).resolve()

        # Derived paths
        self.saved_weight_path = self.saved_model_dir / f"{self.model_name}.weights"

        # Logging info
        self.logger.info(f"Initialized DarknetPipeline ({self.model_name.upper()})")
        self.logger.info(f" - Config path    : {self.config_path}")
        self.logger.info(f" - Darknet dir    : {self.darknet_dir}")
        self.logger.info(f" - Dataset dir    : {self.base_dataset_dir}")
        self.logger.info(f" - Saved model dir: {self.saved_weight_path}")
        self.logger.info(f" - Input dir      : {self.input_dir}")

    # --------------------------------------------------------
    # Step 1️⃣ CFG Setup
    # --------------------------------------------------------
    def step_cfg_manager(self):
        self.logger.info("[STEP 1] Generating Darknet cfg...")
        cfg_mgr = CfgManager(config=self.cfg)
        self.cfg_path = cfg_mgr.generate()
        return self.cfg_path

    # --------------------------------------------------------
    # Step 2️⃣ Makefile Setup
    # --------------------------------------------------------
    def step_make_manager(self):
        self.logger.info("[STEP 2] Configuring & Building Darknet...")
        maker = MakeManager(config=self.cfg)
        maker.configure()
        maker.rebuild()
        maker.verify_darknet()

    # --------------------------------------------------------
    # Step 3️⃣ Dataset Preparation
    # --------------------------------------------------------
    def step_data_prepare(self):
        self.logger.info("[STEP 3] Preparing dataset for Darknet...")
        preparer = DarknetDataPreparer(config=self.cfg)
        preparer.prepare()

    # --------------------------------------------------------
    # Step 4️⃣ Training
    # --------------------------------------------------------
    def step_train(self):
        self.logger.info("[STEP 4] Starting Darknet training...")
        if self.saved_weight_path.exists():
            self.logger.info(
                f"[SKIP] Found existing trained model → {self.saved_weight_path}"
            )
            return
        trainer = DarknetTrainer(config=self.cfg)
        if trainer.verify_files():
            trainer.run()

    # --------------------------------------------------------
    # Step 5️⃣ Evaluation
    # --------------------------------------------------------
    def step_evaluate(self):
        self.logger.info("[STEP 5] Evaluating trained model...")
        evaluator = DarknetEvaluator(config=self.cfg)
        metrics = evaluator.run()
        self.logger.info(
            f"Evaluation complete → mAP@0.5 = {metrics.get('mAP@0.5', 0):.2f}%"
        )
        return metrics

    # --------------------------------------------------------
    # Step 6️⃣ Prediction
    # --------------------------------------------------------
    def step_predict(self):
        self.logger.info("[STEP 6] Running Darknet prediction...")
        predictor = DarknetPredictor(config=self.cfg)
        result_json, predict_txt = predictor.run()
        self.logger.info(f"Prediction complete → {result_json}")
        return result_json, predict_txt

    # --------------------------------------------------------
    # Step 7️⃣ YOLO Crop
    # --------------------------------------------------------
    def step_cropping(self):
        self.logger.info("[STEP 6] Running YOLO Cropping...")
        cropper = YOLOCropper(config=self.cfg)
        cropper.crop_from_json()

        self.logger.info("Cropping complete")

    # --------------------------------------------------------
    # Unified Runner
    # --------------------------------------------------------
    def run(self):
        self.logger.info(f"Starting Darknet Pipeline ({self.model_name.upper()})")
        self.step_cfg_manager()
        # self.step_make_manager()
        # self.step_data_prepare()
        # self.step_train()
        # self.step_evaluate()
        result_json, predict_txt = self.step_predict()
        self.step_cropping()
        self.logger.info("\n🎉 Darknet pipeline completed successfully!")
        self.logger.info(f"Result JSON  : {result_json}")
        self.logger.info(f"Predict TXT  : {predict_txt}")
        self.logger.info(f"Model Weights: {self.saved_weight_path}")
        return result_json


# --------------------------------------------------------
# CLI Entry Point
# --------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Darknet Unified Pipeline Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="utils/config.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    pipeline = DarknetPipeline(config_path=args.config)
    pipeline.run()
