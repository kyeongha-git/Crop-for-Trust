#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
darknet.py
----------
Unified YOLOv2 / YOLOv4 Darknet pipeline.
"""

import sys
from pathlib import Path
import shutil

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
        self.darknet_cfg = yolo_cropper_cfg.get("darknet", {})
        self.dataset_cfg = yolo_cropper_cfg.get("dataset", {})

        # Paths
        self.model_name = self.main_cfg.get("model_name", "yolov4").lower()
        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()
        self.train_dataset_dir = Path(
            f"{self.dataset_cfg.get('train_data_dir', 'data/yolo_cropper')}/{self.model_name}"
            ).resolve()
        self.input_dir = Path(self.main_cfg.get("input_dir", "data/original")).resolve()
        self.detect_output_dir = Path(
            self.dataset_cfg.get("detect_output_dir", "runs/detect")
        ).resolve()
        self.darknet_dir = Path(self.darknet_cfg.get("darknet_dir", "third_party/darknet")).resolve()
        
        # Derived paths
        self.weight_path = self.saved_model_dir / f"{self.model_name}.weights"

        self.logger.info(f"Initialized DarknetPipeline ({self.model_name.upper()})")
        self.logger.info(f" - Demo Mode   : {self.demo_mode}")
        self.logger.info(f" - Config path : {self.config_path}")
        self.logger.info(f" - Input dir   : {self.input_dir}")
        self.logger.info(f" - Saved model : {self.weight_path}")

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
    # Step 1. Cfg Manager
    # --------------------------------------------------------
    def step_cfg_manager(self):
        self.logger.info("[STEP 1] Generating Darknet cfg...")
        cfg_mgr = CfgManager(config=self.cfg)
        cfg_mgr.run()
    
    # --------------------------------------------------------
    # Step 2. Makefile Manager
    # --------------------------------------------------------
    def step_make_manager(self):
        self.logger.info("[STEP 2] Building Darknet...")
        maker = MakeManager(config=self.cfg)
        maker.run()

    # --------------------------------------------------------
    # Step 3. Data Prepare
    # --------------------------------------------------------
    def step_data_prepare(self):
        if self.demo_mode:
            self.logger.info("[STEP 3] Demo mode → Skipping Data Preparing")
            return
        self.logger.info("[STEP 3] Preparing dataset for Darknet...")
        preparer = DarknetDataPreparer(config=self.cfg)
        preparer.run()

    # --------------------------------------------------------
    # Step 4. Train
    # --------------------------------------------------------
    def step_train(self):
        if self.demo_mode:
            self.logger.info("[STEP 4] Demo mode → Skipping training")
            return        
        
        self.logger.info(f"Starting Darknet training ({self.model_name.upper()})")

        trainer = DarknetTrainer(config=self.cfg)
        if trainer.verify_files():
            trainer.run()

    # --------------------------------------------------------
    # Step 5. Evaluate
    # --------------------------------------------------------
    def step_evaluate(self):
        if self.demo_mode:
            self.logger.info("[STEP 5] Demo mode → Skipping evaluation")
            return
        
        self.logger.info("[STEP 5] Evaluating Darknet model...")
        evaluator = DarknetEvaluator(config=self.cfg)
        evaluator.run()
    
    # --------------------------------------------------------
    # Step 6. Predict
    # --------------------------------------------------------
    def step_predict(self):
        self.logger.info("[STEP 6] Running Darknet prediction...")
        predictor = DarknetPredictor(config=self.cfg)
        return predictor.run()
    
    # --------------------------------------------------------
    # Step 7. Cropper
    # --------------------------------------------------------
    def step_cropper(self):
        self.logger.info("[STEP 7] Running ROI cropping...")
        cropper = YOLOCropper(config=self.cfg)
        cropper.run()

    # ------------------------------------------------------
    # Entrypoint
    # ------------------------------------------------------
    def run(self, save_image):
        self.logger.info(f"Running Darknet Pipeline ({self.model_name.upper()})")
        self.cleanup_previous_runs()
        self.step_cfg_manager()
        self.step_make_manager()
        self.step_data_prepare()
        self.step_train()
        self.step_evaluate()
        self.step_predict()
        if save_image:
            self.step_cropper()
        else:
            self.logger.info("[YOLO Crop: False] → Cropper Skip.")

        self.logger.info("\nDarknet pipeline completed successfully!")
