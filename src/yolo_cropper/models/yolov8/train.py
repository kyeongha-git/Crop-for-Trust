#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8 Training Module.

Wraps the Ultralytics API to execute configuration-driven training
and manages checkpoint artifacts.
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
    Manages the training lifecycle for YOLOv8 models, including configuration parsing
    and checkpoint management.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = get_logger("yolo_cropper.YOLOv8Trainer")
        self.cfg = config
        
        # specific configs
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

        self.logger.info(f"Initialized Trainer (Model: {self.model_name})")

    def run(self) -> Any:
        """
        Executes the training loop using Ultralytics API.

        Returns:
            Any: The training results object from Ultralytics, or None if skipped.
        """
        if self.final_model_path.exists():
            self.logger.info(f"Model exists, skipping training: {self.final_model_path}")
            return None

        self.logger.info(
            f"Starting Training (Epochs: {self.epochs}, Batch: {self.batch}, ImgSz: {self.imgsz})"
        )

        # Load pretrained weights or create new model
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

        # Save final best model
        if best_model_src.exists():
            shutil.copy2(best_model_src, self.final_model_path)
            self.logger.info(f"Best model exported to: {self.final_model_path}")

        return results