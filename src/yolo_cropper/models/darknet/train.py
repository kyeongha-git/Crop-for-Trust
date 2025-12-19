#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Darknet Training Module.

Manages the training lifecycle for YOLOv2 and YOLOv4 models using the
Darknet framework. Handles file verification, process execution, and
checkpoint management.
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class DarknetTrainer:
    """
    Manages Darknet-based training execution and artifact retrieval.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = get_logger("yolo_cropper.DarknetTrainer")

        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.darknet_cfg = self.yolo_cropper_cfg.get("darknet", {})
        self.dataset_cfg = self.darknet_cfg.get("dataset", {})

        self.darknet_dir = Path(
            self.darknet_cfg.get("darknet_dir", "third_party/darknet")
        ).resolve()
        self.model_name = self.main_cfg.get("model_name", "yolov2")

        # Darknet internal paths
        self.binary = self.darknet_dir / "darknet"
        self.data_file = "data/obj.data"
        self.cfg_file = f"cfg/{self.model_name}-obj.cfg"

        # Pretrained weights selection
        if self.model_name == "yolov2":
            self.weights_file = self.darknet_dir / "yolov2.weights"
        elif self.model_name == "yolov4":
            self.weights_file = self.darknet_dir / "yolov4.conv.137"
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

        self.logs_dir = self.darknet_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized Trainer (Model: {self.model_name.upper()})")

    def verify_files(self) -> bool:
        """
        Verifies existence of required Darknet binaries and configuration files.

        Returns:
            bool: True if all required files exist, False otherwise.
        """
        required = [
            self.binary,
            self.darknet_dir / self.data_file,
            self.darknet_dir / self.cfg_file,
            self.weights_file,
        ]
        missing = [str(f) for f in required if not f.exists()]

        if missing:
            self.logger.error(f"Missing required files: {missing}")
            return False

        return True

    def run(self, weights_init: Optional[str] = None) -> str:
        """
        Executes the Darknet training subprocess.

        Args:
            weights_init (Optional[str]): Path to custom initial weights.

        Returns:
            str: Path to the saved best model weights.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.logs_dir / f"train_{self.model_name}_{timestamp}.log"

        init_weight = weights_init or self.weights_file.name
        clear_flag = "-clear" if self.model_name == "yolov2" else ""

        cmd = [
            "bash",
            "-lc",
            (
                f"./darknet detector train {self.data_file} {self.cfg_file} {init_weight} "
                f"{clear_flag} -dont_show -map | tee {log_path}"
            ),
        ]

        self.logger.info(f"Starting Training ({self.model_name.upper()})")

        process = subprocess.run(cmd, cwd=self.darknet_dir, shell=False)

        if process.returncode != 0:
            self.logger.error(f"Training failed. Check log: {log_path}")
            raise RuntimeError(f"Darknet training failed (Code: {process.returncode})")

        self.logger.info(f"Training complete. Log: {log_path}")

        # Artifact retrieval
        ckpt_dir = (
            Path(self.dataset_cfg.get("checkpoints_dir", "checkpoints/yolo_cropper"))
            / self.model_name
        )
        saved_dir = (
            Path(self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper"))
            / self.model_name
        )

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        saved_dir.mkdir(parents=True, exist_ok=True)

        best_weight = next(ckpt_dir.glob("*_best.weights"), None)
        target_weight = saved_dir / f"{self.model_name}.weights"

        if best_weight:
            shutil.copy2(best_weight, target_weight)
            self.logger.info(f"Best weights exported to {target_weight}")
        else:
            self.logger.warning("No best weights found in checkpoints.")

        return str(target_weight)