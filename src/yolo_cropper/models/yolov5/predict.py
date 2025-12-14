#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict.py
----------
This module manages YOLOv5 inference using a configuration-driven setup.

It automatically iterates through all subfolders under the input root directory,
executes YOLOv5 detection for each one, and organizes the results in structured
output folders. Existing results are removed before each run to prevent duplicates.
"""

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import os

# Resolve Project Root Directory
ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger
from utils.weights import download_file, verify_sha256


class YOLOv5Predictor:
    """
    Runs YOLOv5 detection across multiple folders based on configuration input.

    The predictor automatically detects all subdirectories within the given
    input root, executes YOLOv5 detection for each, and manages log files
    and output directories in a clean, reproducible manner.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the YOLOv5 predictor from a configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration object passed from the main
                controller, containing YOLOv5, dataset, and runtime parameters.
        """
        self.logger = get_logger("yolo_cropper.YOLOv5Predictor")

        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.yolov5_cfg = self.yolo_cropper_cfg.get("yolov5", {})
        self.train_cfg = self.yolo_cropper_cfg.get("train", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})
        
        self.project_root = ROOT_DIR
        
        self.model_name = self.main_cfg.get("model_name", "yolov5")

        self.yolov5_dir = Path(
            self.yolov5_cfg.get("yolov5_dir", "third_party/yolov5")
        ).resolve()
        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()
        self.input_root = Path(
            self.main_cfg.get("input_dir", "data/original")
        ).resolve()
        self.detect_root = Path(
            self.dataset_cfg.get("detect_output_dir", "runs/detect")
        ).resolve()

        self.saved_model_path = self.saved_model_dir / f"{self.model_name}.pt"
        self.logs_dir = self.yolov5_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.detect_root.mkdir(parents=True, exist_ok=True)

        self.device = str(self.train_cfg.get("device", "cpu"))
        self.save_crop = bool(self.train_cfg.get("save_crop", False))
        self.save_txt = bool(self.train_cfg.get("save_txt", True))
        self.save_conf = bool(self.train_cfg.get("save_conf", True))
        self.name_prefix = self.model_name

        self.logger.info(f"YOLOv5Predictor initialized ({self.model_name.upper()})")
        self.logger.debug(f"Repo Dir   : {self.yolov5_dir}")
        self.logger.debug(f"Weights    : {self.saved_model_path}")
        self.logger.debug(f"Source Dir : {self.input_root}")
        self.logger.debug(f"Output Dir : {self.detect_root}")

    def _prepare_weights(self):
        """
        Ensure YOLOv5 weight exists in saved_model directory.
        If missing, download from config-defined pretrained weights.
        """
        weight_file = self.saved_model_path

        if weight_file.exists():
            self.logger.info(f"[WEIGHT] Using existing weight: {weight_file}")
            return

        self.logger.info("[WEIGHT] Local YOLOv5 weight missing → downloading...")

        weights_cfg = self.cfg.get("weights", {})
        sha_cfg = self.cfg.get("sha256", {})

        if self.model_name not in weights_cfg:
            raise FileNotFoundError(
                f"No pretrained weight URL found for model '{self.model_name}', "
                f"and local weight does not exist: {weight_file}"
            )

        url = weights_cfg[self.model_name]
        sha = sha_cfg.get(self.model_name)

        weight_file.parent.mkdir(parents=True, exist_ok=True)
        download_file(url, weight_file)

        if sha:
            if verify_sha256(weight_file, sha):
                self.logger.info("[WEIGHT] SHA256 verified.")
            else:
                raise RuntimeError(
                    f"SHA256 mismatch for downloaded YOLOv5 weight: {weight_file}"
                )
        else:
            self.logger.warning("[WEIGHT] No SHA256 provided — skipping integrity check.")

    def _run_inference(self, folder_path: Path):
        """
        Execute YOLOv5 detection for a single input folder.

        This method builds and runs a YOLOv5 command targeting a specific
        subdirectory, logging output and deleting any existing results
        before re-running to ensure clean outputs.
        """
        if not folder_path.exists():
            self.logger.warning(f"[!] Source folder not found: {folder_path}")
            return

        exp_name = f"{self.name_prefix}_{folder_path.name}"
        exp_dir = self.detect_root / exp_name

        # Remove old results (avoid exp/exp2 duplication)
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
            self.logger.warning(f"Existing result folder deleted → {exp_dir}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.logs_dir / f"detect_{folder_path.name}_{timestamp}.log"

        cmd = [
            "python",
            "detect.py",
            "--weights",
            str(self.saved_model_path),
            "--source",
            str(folder_path),
            "--project",
            str(self.detect_root),
            "--name",
            exp_name,
            "--device",
            "cpu",
        ]

        if self.save_crop:
            cmd.append("--save-crop")
        if self.save_txt:
            cmd.append("--save-txt")
        if self.save_conf:
            cmd.append("--save-conf")

        env = os.environ.copy()
        yolo_path = str(self.yolov5_dir)
        app_path = str(self.project_root)

        env["PYTHONPATH"] = f"{yolo_path}:{env.get('PYTHONPATH', '')}"

        self.logger.info(f"Running YOLOv5 detection → {folder_path.name}")
        self.logger.debug(f"Command: {' '.join(cmd)}")

        with open(log_path, "w", encoding="utf-8") as log_f:
            process = subprocess.run(
                cmd, 
                cwd=self.yolov5_dir,
                stdout=log_f, 
                stderr=subprocess.STDOUT, 
                env=env 
            )

        if process.returncode != 0:
            self.logger.error(
                f"Detection failed ({folder_path.name}), code={process.returncode}. See log: {log_path}"
            )
        else:
            self.logger.info(f"Detection complete → {exp_dir}")

    def run(self):
        """
        Perform YOLOv5 inference for every subfolder under the input root directory.

        This method scans all subdirectories inside the configured input path
        (e.g., `data/original`), then runs `_run_inference()` sequentially
        for each. Each folder’s results are stored in separate output directories.
        """
        self._prepare_weights()

        subfolders = [p for p in self.input_root.iterdir() if p.is_dir()]
        if not subfolders:
            raise FileNotFoundError(f"No subfolders found under {self.input_root}")

        self.logger.info(
            f"Found {len(subfolders)} folders → {[p.name for p in subfolders]}"
        )
        for folder in subfolders:
            self._run_inference(folder)

        self.logger.info("All detections complete.")