#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
darknet.py
----------
Unified YOLOv2 / YOLOv4 Darknet pipeline.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from src.yolo_cropper.core.cropper import YOLOCropper
from src.yolo_cropper.models.darknet.evaluate import DarknetEvaluator
from src.yolo_cropper.models.darknet.predict import DarknetPredictor
from src.yolo_cropper.models.darknet.setup.cfg_manager import CfgManager
from src.yolo_cropper.models.darknet.setup.data_prepare import DarknetDataPreparer
from src.yolo_cropper.models.darknet.setup.make_manager import MakeManager
from src.yolo_cropper.models.darknet.train import DarknetTrainer
from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging


class DarknetPipeline:
    """Unified workflow for Darknet YOLOv2/YOLOv4."""

    def __init__(self, config_path: str = "utils/config.yaml"):
        setup_logging("logs/yolo_cropper")
        self.logger = get_logger("yolo_cropper.DarknetPipeline")

        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)

        self.global_main_cfg = self.cfg.get("main", {})

        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = yolo_cropper_cfg.get("main", {})
        self.darknet_cfg = yolo_cropper_cfg.get("darknet", {})
        self.dataset_cfg = yolo_cropper_cfg.get("dataset", {})

        # Demo mode
        self.demo_mode = self.global_main_cfg.get("demo", "off") == "on"

        # Paths
        self.model_name = self.main_cfg.get("model_name", "yolov4").lower()
        self.darknet_dir = Path(self.darknet_cfg.get("darknet_dir", "third_party/darknet")).resolve()
        self.saved_model_dir = Path(self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")).resolve()
        self.base_dataset_dir = Path(self.dataset_cfg.get("train_data_dir", "data/yolo_cropper")).resolve()
        self.train_dataset_dir = Path(f"{self.dataset_cfg.get('train_data_dir', 'data/yolo_cropper')}/{self.model_name}").resolve()
        self.input_dir = Path(self.main_cfg.get("input_dir", "data/original")).resolve()

        self.saved_weight_path = self.saved_model_dir / f"{self.model_name}.weights"

        self.logger.info(f"Initialized DarknetPipeline ({self.model_name.upper()})")
        self.logger.info(f" - Demo Mode   : {self.demo_mode}")
        self.logger.info(f" - Config path : {self.config_path}")
        self.logger.info(f" - Input dir   : {self.input_dir}")
        self.logger.info(f" - Saved model : {self.saved_weight_path}")

    # ------------------------------------------------------
    # STEP FUNCTIONS
    # ------------------------------------------------------
    def step_cfg_manager(self):
        self.logger.info("[STEP 1] Generating Darknet cfg...")
        cfg_mgr = CfgManager(config=self.cfg)
        return cfg_mgr.generate()

    def step_make_manager(self):
        self.logger.info("[STEP 2] Building Darknet...")
        maker = MakeManager(config=self.cfg)
        maker.configure()
        maker.rebuild()
        maker.verify_darknet()

    def step_data_prepare(self):
        self.logger.info("[STEP 3] Preparing dataset for Darknet...")
        preparer = DarknetDataPreparer(config=self.cfg)
        preparer.prepare()

    def step_train(self):
        self.logger.info("[STEP 4] Training Darknet model...")
        if self.saved_weight_path.exists():
            self.logger.info(f"[SKIP] Found pretrained weights → {self.saved_weight_path}")
            return
        trainer = DarknetTrainer(config=self.cfg)
        if trainer.verify_files():
            trainer.train()

    def step_evaluate(self):
        self.logger.info("[STEP 5] Evaluating Darknet model...")
        evaluator = DarknetEvaluator(config=self.cfg)
        return evaluator.run()

    def step_predict(self):
        self.logger.info("[STEP 6] Running Darknet prediction...")
        predictor = DarknetPredictor(config=self.cfg)
        return predictor.run()

    def step_cropping(self):
        self.logger.info("[STEP 7] Running ROI cropping...")
        cropper = YOLOCropper(config=self.cfg)
        cropper.crop_from_json()

    # ------------------------------------------------------
    # Unified Runner
    # ------------------------------------------------------
    def run(self):
        self.logger.info(f"Starting Darknet Pipeline ({self.model_name.upper()})")

        # Always required
        self.step_cfg_manager()
        self.step_make_manager()

        if not self.demo_mode:
            self.step_data_prepare()
            self.step_train()
            self.step_evaluate()
        else:
            self.logger.info("[DEMO MODE] Skipping build/train/evaluate.")

        result_json, txt = self.step_predict()
        self.step_cropping()

        self.logger.info("\nDarknet pipeline completed successfully!")
        return result_json
