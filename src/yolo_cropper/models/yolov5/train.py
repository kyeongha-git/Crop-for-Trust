#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py
--------
This module manages the YOLOv5 training process using a configuration-driven approach.

It provides automated handling of dataset path resolution (data.yaml), structured logging,
and checkpoint saving. The class encapsulates the entire training process without requiring
manual CLI calls to the YOLOv5 repository.
"""

import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class YOLOv5Trainer:
    """
    Handles YOLOv5 training via a configuration-driven execution pipeline.

    This class automates training by resolving dataset paths, executing
    YOLOv5's training script, managing logs, and saving the best model
    checkpoint for later use.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the YOLOv5 trainer using a provided configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing main,
                YOLOv5, training, and dataset parameters.

        """
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
        self.device = str(self.train_cfg.get("device", 0)).strip()
        self.name_prefix = self.model_name

        self.logs_dir = self.yolov5_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("YOLOv5Trainer initialized (config-driven)")
        self.logger.debug(f"Repo Dir : {self.yolov5_dir}")
        self.logger.debug(f"Data YAML: {self.data_yaml_path}")

        # Resolve data.yaml paths to absolute before training
        self.data_yaml = self._resolve_data_yaml(self.data_yaml_path)

    def _resolve_data_yaml(self, data_yaml_path: Path) -> Path:
        """
        Convert relative paths in data.yaml to absolute paths.

        This ensures YOLOv5 correctly locates training, validation,
        and test image directories regardless of the working directory.

        Returns:
            Path: Path to the temporary resolved data.yaml file with absolute paths.

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
                self.logger.info(f"Resolved {key}: {abs_path}")

        tmp_dir = Path(tempfile.mkdtemp(prefix="yolov5_datayaml_"))
        resolved_yaml = tmp_dir / "data_resolved.yaml"
        with open(resolved_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

        self.logger.info(f"Temporary data.yaml created → {resolved_yaml}")
        return resolved_yaml

    def run(self):
        """
        Run the YOLOv5 training process.

        This method executes the official YOLOv5 `train.py` script with
        parameters from the configuration file. Logs are saved to a timestamped
        file, and the best model checkpoint is automatically stored after training.

        Returns:
            Path: Path to the generated training log file.

        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.logs_dir / f"train_{timestamp}.log"
        exp_name = f"{self.name_prefix}_{timestamp}"

        cmd = [
            "python",
            "train.py",
            "--data",
            str(self.data_yaml),
            "--epochs",
            str(self.epochs),
            "--batch-size",
            str(self.batch_size),
            "--imgsz",
            str(self.imgsz),
            "--device",
            str(self.device),
            "--project",
            str(self.checkpoints_dir),
            "--name",
            exp_name,
        ]

        self.logger.info("Starting YOLOv5 training")
        for key, val in {
            "Epochs": self.epochs,
            "Batch": self.batch_size,
            "Image Size": self.imgsz,
            "Device": self.device,
        }.items():
            self.logger.info(f"   - {key:<12}: {val}")

        with open(log_path, "w", encoding="utf-8") as log_f:
            process = subprocess.run(
                cmd, cwd=self.yolov5_dir, stdout=log_f, stderr=subprocess.STDOUT
            )

        if process.returncode != 0:
            self.logger.error(
                f"Training failed (code: {process.returncode}). See log: {log_path}"
            )
            raise RuntimeError(f"YOLOv5 training failed — check log: {log_path}")

        self.logger.info(f"Training complete → {log_path}")
        self._save_best_weight()
        return log_path

    def _save_best_weight(self):
        """
        Copy the best YOLOv5 model checkpoint to the saved model directory.

        The best checkpoint is located under the checkpoints directory and saved
        as `best.pt`. This method copies it to `saved_model/yolo_cropper/yolov5.pt`
        for downstream tasks or inference.

        """
        best_weight_src = next(self.checkpoints_dir.rglob("best.pt"), None)
        target_path = self.saved_model_dir / "yolov5.pt"

        if best_weight_src and best_weight_src.exists():
            shutil.copy2(best_weight_src, target_path)
            self.logger.info(f"Copied best weight → {target_path}")
        else:
            self.logger.warning("No best.pt found in checkpoints directory.")
