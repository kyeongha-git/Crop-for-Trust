#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolov8_trainer.py
-----------------
This module defines a configuration-driven training pipeline for YOLOv8 models.

It provides a fully automated workflow for training YOLOv8 using Ultralytics’ API,
saving both intermediate checkpoints and the final best-performing model.

Steps:
1️⃣ Load configuration and initialize YOLOv8 model
2️⃣ Train model using provided parameters
3️⃣ Save best and last checkpoints
4️⃣ Export final best model to `saved_model/yolo_cropper/{model_name}.pt`
"""

import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class YOLOv8Trainer:
    """
    Handles YOLOv8 model training in a configuration-driven pipeline.

    This class automates dataset loading, model initialization,
    training execution, checkpoint management, and saving of the
    best-performing model. It is designed for reproducible, clean,
    and easily configurable training workflows.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the YOLOv8 trainer from a configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration object containing YOLOv8,
                training, and dataset parameters.
        """
        self.logger = get_logger("yolo_cropper.YOLOv8Trainer")
        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.yolov8_cfg = self.yolo_cropper_cfg.get("yolov8", {})
        self.train_cfg = self.yolo_cropper_cfg.get("train", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        # Model and data parameters
        self.model_name = self.main_cfg.get("model_name", "yolov8s")
        self.data_yaml = Path(
            self.yolov8_cfg.get("data_yaml", "data/yolo_cropper/yolov8/data.yaml")
        ).resolve()
        self.epochs = self.train_cfg.get("epochs", 200)
        self.imgsz = self.train_cfg.get("imgsz", 416)
        self.batch = self.train_cfg.get("batch", 16)

        # Directory setup
        self.runs_dir = Path(
            self.dataset_cfg.get("train_output_dir", "runs/train")
        ).resolve()
        self.checkpoint_dir = (
            Path(self.dataset_cfg.get("checkpoint_dir", "checkpoints/yolo_cropper"))
            / self.model_name
        )
        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saved_model_dir.mkdir(parents=True, exist_ok=True)

        self.final_model_path = self.saved_model_dir / f"{self.model_name}.pt"

        self.logger.info(f"Initialized YOLOv8Trainer ({self.model_name})")
        self.logger.debug(f" - Data YAML : {self.data_yaml}")
        self.logger.debug(f" - Runs dir  : {self.runs_dir}")
        self.logger.debug(f" - Checkpoints: {self.checkpoint_dir}")
        self.logger.debug(f" - Saved model: {self.final_model_path}")

    def run(self):
        """
        Train the YOLOv8 model using Ultralytics’ API.

        This method automatically handles:
            - Model loading or initialization
            - Training execution based on configuration parameters
            - Saving best and last model checkpoints
            - Copying the best-performing model to the `saved_model` directory

        Returns:
            ultralytics.engine.results.Results | Path:
                The training results object returned by Ultralytics,
                or the saved model path if training was skipped.

        """
        if self.final_model_path.exists():
            self.logger.info(f"[SKIP] Found existing model → {self.final_model_path}")
            return self.final_model_path

        self.logger.info(f"Starting YOLOv8 training for {self.epochs} epochs")
        self.logger.info(f"   Model : {self.model_name}")
        self.logger.info(f"   Batch : {self.batch}, ImgSize : {self.imgsz}")
        self.logger.info(f"   Data  : {self.data_yaml}")

        # Load pretrained weights or create a new model
        weight_file = f"{self.model_name}.pt"
        model = YOLO(weight_file)

        results = model.train(
            data=str(self.data_yaml),
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            project=str(self.runs_dir),
            name=self.model_name,
        )

        # Save checkpoints
        best_model_src = results.save_dir / "weights" / "best.pt"
        last_model_src = results.save_dir / "weights" / "last.pt"

        if best_model_src.exists():
            shutil.copy2(
                best_model_src, self.checkpoint_dir / f"{self.model_name}_best.pt"
            )
        if last_model_src.exists():
            shutil.copy2(
                last_model_src, self.checkpoint_dir / f"{self.model_name}_last.pt"
            )

        # Save best model
        if best_model_src.exists():
            shutil.copy2(best_model_src, self.final_model_path)
            self.logger.info(f"Best model saved → {self.final_model_path}")

        self.logger.info(f"Checkpoints saved → {self.checkpoint_dir}")
        self.logger.info(f"Training logs saved → {results.save_dir}")
        return results
