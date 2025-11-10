#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8 Trainer (Config-driven)
------------------------------
- config.yaml 기반 학습 파이프라인
- best.pt → saved_model/yolo_cropper/{model_name}.pt
- 전체 weight → checkpoints/yolo_cropper/{model_name}/ 저장
"""

import shutil
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import setup_logging, get_logger


class YOLOv8Trainer:
    """YOLOv8 Training Class (config-driven)"""

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.YOLOv8Trainer")
        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg .get("main", {})
        self.yolov8_cfg = self.yolo_cropper_cfg .get("yolov8", {})
        self.train_cfg = self.yolo_cropper_cfg .get("train", {})
        self.dataset_cfg = self.yolo_cropper_cfg .get("dataset", {})

        self.model_name = self.main_cfg.get("model_name", "yolov8s")
        self.data_yaml = Path(self.yolov8_cfg.get("data_yaml", "data/yolo_cropper/yolov8/data.yaml")).resolve()
        self.epochs = self.train_cfg.get("epochs", 200)
        self.imgsz = self.train_cfg.get("imgsz", 416)
        self.batch = self.train_cfg.get("batch", 16)

        # Directory setup
        self.runs_dir = Path(self.dataset_cfg.get("train_output_dir", "runs/train")).resolve()
        self.checkpoint_dir = Path(self.dataset_cfg.get("checkpoint_dir", "checkpoints/yolo_cropper")) / self.model_name
        self.saved_model_dir = Path(self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")).resolve()

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saved_model_dir.mkdir(parents=True, exist_ok=True)

        self.final_model_path = self.saved_model_dir / f"{self.model_name}.pt"

        self.logger.info(f"Initialized YOLOv8Trainer ({self.model_name})")
        self.logger.debug(f" - Data YAML : {self.data_yaml}")
        self.logger.debug(f" - Runs dir  : {self.runs_dir}")
        self.logger.debug(f" - Checkpoints: {self.checkpoint_dir}")
        self.logger.debug(f" - Saved model: {self.final_model_path}")

    # --------------------------------------------------------
    def run(self):
        """Train YOLOv8 model"""
        if self.final_model_path.exists():
            self.logger.info(f"[SKIP] Found existing model → {self.final_model_path}")
            return self.final_model_path

        self.logger.info(f"🚀 Starting YOLOv8 training for {self.epochs} epochs")
        self.logger.info(f"   Model : {self.model_name}")
        self.logger.info(f"   Batch : {self.batch}, ImgSize : {self.imgsz}")
        self.logger.info(f"   Data  : {self.data_yaml}")

        # Load pretrained model or create new one
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

        # --- Save Checkpoints ---
        best_model_src = results.save_dir / "weights" / "best.pt"
        last_model_src = results.save_dir / "weights" / "last.pt"

        if best_model_src.exists():
            shutil.copy2(best_model_src, self.checkpoint_dir / f"{self.model_name}_best.pt")
        if last_model_src.exists():
            shutil.copy2(last_model_src, self.checkpoint_dir / f"{self.model_name}_last.pt")

        # --- Save Best Model ---
        if best_model_src.exists():
            shutil.copy2(best_model_src, self.final_model_path)
            self.logger.info(f"[✓] Best model saved → {self.final_model_path}")

        self.logger.info(f"[✓] Checkpoints saved → {self.checkpoint_dir}")
        self.logger.info(f"[✓] Training logs saved → {results.save_dir}")
        return results