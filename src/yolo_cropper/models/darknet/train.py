#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py
--------
This module handles the Darknet training process for YOLOv2 and YOLOv4 models.

It is fully configuration-driven and uses structured logging instead of print
statements. The class verifies required files, runs the Darknet training command,
and automatically saves the best model weights after training.
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class DarknetTrainer:
    """
    Manages YOLOv2/YOLOv4 training in Darknet.

    This class automates the training process, verifying all required files,
    executing the Darknet command, logging outputs, and copying the best
    checkpoint to the designated saved model directory.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Darknet trainer with configuration data.

        Args:
            config (Dict[str, Any]): Full configuration dictionary passed from the main controller.

        Raises:
            ValueError: If the specified model_name is unsupported.
        """
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

        # Paths inside Darknet
        self.binary = self.darknet_dir / "darknet"
        self.data_file = "data/obj.data"
        self.cfg_file = f"cfg/{self.model_name}-obj.cfg"

        # Select pretrained weights
        if self.model_name == "yolov2":
            self.weights_file = self.darknet_dir / "yolov2.weights"
        elif self.model_name == "yolov4":
            self.weights_file = self.darknet_dir / "yolov4.conv.137"
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

        # Log directory setup
        self.logs_dir = self.darknet_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized DarknetTrainer for {self.model_name.upper()}")
        self.logger.debug(f"  - Darknet dir : {self.darknet_dir}")
        self.logger.debug(f"  - Weight file : {self.weights_file}")
        self.logger.debug(f"  - Config file : {self.cfg_file}")

    def verify_files(self) -> bool:
        """
        Check that all required files exist before training begins.

        Returns:
            bool: True if all files exist, False otherwise.
        """
        required = [
            self.binary,
            self.darknet_dir / self.data_file,
            self.darknet_dir / self.cfg_file,
            self.weights_file,
        ]
        missing = [str(f) for f in required if not f.exists()]

        if missing:
            self.logger.error("Missing required files:")
            for m in missing:
                self.logger.error(f"   - {m}")
            return False

        self.logger.info("All required files are present for training.")
        return True

    def run(self, weights_init: str = None):
        """
        Start the Darknet training process.

        The method runs Darknet’s `detector train` command using the provided
        configuration and pretrained weights. It logs the full output and
        copies the best-performing checkpoint to the saved model directory.

        Args:
            weights_init (str, optional): Custom initial weights file path. Defaults to the model’s standard pretrained weights.

        Returns:
            str: Path to the final saved weights file.

        Raises:
            RuntimeError: If the Darknet training process fails.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.logs_dir / f"train_{self.model_name}_{timestamp}.log"

        init_weight = weights_init or os.path.basename(self.weights_file)
        clear_flag = "-clear" if self.model_name == "yolov2" else ""

        cmd = [
            "bash",
            "-lc",
            (
                f"./darknet detector train {self.data_file} {self.cfg_file} {init_weight} "
                f"{clear_flag} -dont_show -map | tee {log_path}"
            ),
        ]

        self.logger.debug(f"[CMD] {' '.join(cmd[2:])}")

        process = subprocess.run(cmd, cwd=self.darknet_dir, shell=False)

        if process.returncode != 0:
            raise RuntimeError(
                f"Training failed (code: {process.returncode}). See log: {log_path}"
            )

        self.logger.info(f"Training complete! Log saved → {log_path}")

        # Copy best checkpoint
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
            self.logger.info(f"Copied best weight → {target_weight}")
        else:
            self.logger.warning(
                "No '_best.weights' file found in checkpoints directory."
            )

        return str(target_weight)