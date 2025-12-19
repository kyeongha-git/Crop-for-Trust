#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv5 Training Module.

Wraps the YOLOv5 repository's training script, handling path resolution
and checkpoint management via a configuration-driven approach.
"""

import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class YOLOv5Trainer:
    """
    Manages configuration-driven training for YOLOv5.
    Resolves dataset paths and executes the training subprocess.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = get_logger("yolo_cropper.YOLOv5Trainer")

        self.cfg = config
        self.main_cfg = self.cfg.get("main", {})
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.yolov5_cfg = self.yolo_cropper_cfg.get("yolov5", {})
        self.train_cfg = self.yolo_cropper_cfg.get("train", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})
        self.model_name = self.main_cfg.get("model_name", "yolov5")

        self.yolov5_dir = Path(
            self.yolov5_cfg.get("yolov5_dir", "third_party/yolov5")
        ).resolve()
        self.data_yaml_path = Path(
            self.yolov5_cfg.get("data_yaml", "data/yolo_cropper/yolov5/data.yaml")
        ).resolve()
        self.checkpoints_dir = Path(
            f"{self.dataset_cfg.get('checkpoints_dir', 'checkpoints/yolo_cropper')}/{self.model_name}"
        ).resolve()
        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.saved_model_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = self.train_cfg.get("epochs", 400)
        self.batch_size = self.train_cfg.get("batch_size", 16)
        self.imgsz = self.train_cfg.get("imgsz", 416)
        self.device = str(self.train_cfg.get("device", "cpu")).strip()
        self.name_prefix = self.model_name

        self.logs_dir = self.yolov5_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.final_model_path = self.saved_model_dir / f"{self.model_name}.pt"

        self.logger.info("Initialized Trainer")
        self.data_yaml = self._resolve_data_yaml(self.data_yaml_path)

    def _resolve_data_yaml(self, data_yaml_path: Path) -> Path:
        """
        Converts relative paths in data.yaml to absolute paths for YOLOv5 compatibility.

        Returns:
            Path: Path to the temporary resolved YAML file.
        """
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        base = data_yaml_path.parent.resolve()
        for key in ("train", "val", "test"):
            if key in data and isinstance(data[key], str):
                abs_path = (base / data[key]).resolve()
                if not abs_path.exists() and (abs_path / "images").exists():
                    abs_path = abs_path / "images"
                data[key] = str(abs_path)

        tmp_dir = Path(tempfile.mkdtemp(prefix="yolov5_datayaml_"))
        resolved_yaml = tmp_dir / "data_resolved.yaml"
        with open(resolved_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

        self.logger.debug(f"Resolved data.yaml created at {resolved_yaml}")
        return resolved_yaml

    def _save_best_weight(self) -> None:
        """
        Copies the best model checkpoint to the saved model directory.
        """
        best_weight_src = next(self.checkpoints_dir.rglob("best.pt"), None)
        target_path = self.saved_model_dir / "yolov5.pt"

        if best_weight_src and best_weight_src.exists():
            shutil.copy2(best_weight_src, target_path)
            self.logger.info(f"Best model exported to {target_path}")
        else:
            self.logger.warning("No best.pt found in checkpoints.")

    def run(self) -> Optional[Path]:
        """
        Executes the YOLOv5 training subprocess.

        Returns:
            Optional[Path]: Path to the training log file, or None if skipped.
        """
        if self.final_model_path.exists():
            self.logger.info(f"Model exists, skipping training: {self.final_model_path}")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.logs_dir / f"train_{timestamp}.log"
        exp_name = f"{self.name_prefix}_{timestamp}"

        cmd = [
            "python",
            "train.py",
            "--data", str(self.data_yaml),
            "--epochs", str(self.epochs),
            "--batch-size", str(self.batch_size),
            "--imgsz", str(self.imgsz),
            "--device", str(self.device),
            "--project", str(self.checkpoints_dir),
            "--name", exp_name,
        ]

        self.logger.info(
            f"Starting Training (Epochs: {self.epochs}, Batch: {self.batch_size}, Device: {self.device})"
        )

        with open(log_path, "w", encoding="utf-8") as log_f:
            process = subprocess.run(
                cmd, cwd=self.yolov5_dir, stdout=log_f, stderr=subprocess.STDOUT
            )

        if process.returncode != 0:
            self.logger.error(f"Training failed. Check log: {log_path}")
            raise RuntimeError("YOLOv5 training subprocess failed.")

        self.logger.info(f"Training completed. Logs: {log_path}")
        self._save_best_weight()
        return log_path