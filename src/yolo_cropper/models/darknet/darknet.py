#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Darknet Pipeline.

Orchestrates the end-to-end workflow for YOLOv2 and YOLOv4:
Configuration, Compilation, Data Preparation, Training, Evaluation, Prediction, and Cropping.
"""

import sys
from pathlib import Path
import shutil
from typing import Optional, Tuple

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
    """
    Manages the execution flow for the Darknet-based (YOLOv2/v4) detection and cropping pipeline.
    """

    def __init__(self, config_path: str = "utils/config.yaml") -> None:
        setup_logging("logs/yolo_cropper")
        self.logger = get_logger("yolo_cropper.DarknetPipeline")

        # Load Configuration
        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)

        self.global_main_cfg = self.cfg.get("main", {})
        self.demo_mode = self.global_main_cfg.get("demo", False)

        # Component configurations
        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = yolo_cropper_cfg.get("main", {})
        self.darknet_cfg = yolo_cropper_cfg.get("darknet", {})
        self.dataset_cfg = yolo_cropper_cfg.get("dataset", {})

        # Path setup
        self.model_name = self.main_cfg.get("model_name", "yolov4").lower()
        self.saved_model_dir = (
            Path(self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper"))
            .resolve()
        )
        self.input_dir = Path(
            self.main_cfg.get("input_dir", "data/original")
        ).resolve()
        self.detect_output_dir = (
            Path(self.dataset_cfg.get("detect_output_dir", "runs/detect"))
            / self.model_name
        )
        self.darknet_dir = Path(
            self.darknet_cfg.get("darknet_dir", "third_party/darknet")
        ).resolve()
        
        self.weight_path = self.saved_model_dir / f"{self.model_name}.weights"

        self.logger.info(f"Initialized DarknetPipeline (Model: {self.model_name.upper()})")

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

    def step_cfg_manager(self) -> None:
        """
        Generates the Darknet configuration (.cfg) file based on YAML settings.
        """
        self.logger.info("[Step 1] Generating Darknet configuration")
        cfg_mgr = CfgManager(config=self.cfg)
        cfg_mgr.run()

    def step_make_manager(self) -> None:
        """
        Compiles the Darknet binary using the Makefile.
        """
        self.logger.info("[Step 2] Building Darknet binary")
        maker = MakeManager(config=self.cfg)
        maker.run()

    def step_data_prepare(self) -> None:
        """
        Prepares the dataset for Darknet ingestion.
        """
        if self.demo_mode:
            self.logger.info("Demo mode enabled: Skipping data preparation.")
            return

        self.logger.info("[Step 3] Preparing dataset")
        preparer = DarknetDataPreparer(config=self.cfg)
        preparer.run()

    def step_train(self) -> None:
        """
        Executes the training process if not in demo mode.
        """
        if self.demo_mode:
            self.logger.info("Demo mode enabled: Skipping training.")
            return
        
        self.logger.info(f"[Step 4] Starting Training ({self.model_name.upper()})")
        trainer = DarknetTrainer(config=self.cfg)
        if trainer.verify_files():
            trainer.run()

    def step_evaluate(self) -> None:
        """
        Executes the evaluation module to assess model performance.
        """
        if self.demo_mode:
            self.logger.info("Demo mode enabled: Skipping evaluation.")
            return
        
        self.logger.info("[Step 5] Starting Evaluation")
        evaluator = DarknetEvaluator(config=self.cfg)
        evaluator.run()

    def step_predict(self) -> Tuple[str, str]:
        """
        Runs inference on the target images using Darknet.

        Returns:
            Tuple[str, str]: Paths to result JSON and prediction list.
        """
        self.logger.info("[Step 6] Running Prediction")
        predictor = DarknetPredictor(config=self.cfg)
        return predictor.run()

    def step_cropper(self) -> None:
        """
        Crops images based on the detection coordinates.
        """
        self.logger.info("[Step 7] Cropping images")
        cropper = YOLOCropper(config=self.cfg)
        cropper.run()

    def run(self, save_image: bool) -> None:
        """
        Executes the full pipeline sequence.

        Args:
            save_image (bool): If True, proceeds to crop images after detection.
        """
        self.logger.info("===== Starting Darknet Pipeline =====")
        
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
            self.logger.info("Skipping crop step (save_image=False)")

        self.logger.info("Darknet pipeline completed successfully.")