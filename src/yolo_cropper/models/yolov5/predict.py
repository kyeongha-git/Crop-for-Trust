#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv5 Inference Module.

Iterates through input directories, executes YOLOv5 detection via subprocess,
and organizes results in structured output folders.
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
from utils.model_hub import download_fine_tuned_weights


class YOLOv5Predictor:
    """
    Manages batch inference for YOLOv5 across multiple subdirectories.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = get_logger("yolo_cropper.YOLOv5Predictor")
        self.cfg = config
        
        self.global_main_cfg = self.cfg.get("main", {})
        self.demo_mode = self.global_main_cfg.get("demo", False)
        
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
        self.detect_output_dir = Path(
            self.dataset_cfg.get("detect_output_dir", "runs/detect")
        ).resolve()

        self.saved_model_path = self.saved_model_dir / f"{self.model_name}.pt"
        self.logs_dir = self.yolov5_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.detect_output_dir.mkdir(parents=True, exist_ok=True)

        self.device = str(self.train_cfg.get("device", "cpu"))
        self.save_crop = bool(self.train_cfg.get("save_crop", False))
        self.save_txt = bool(self.train_cfg.get("save_txt", True))
        self.save_conf = bool(self.train_cfg.get("save_conf", True))
        self.name_prefix = self.model_name

        self.logger.info(f"Initialized Predictor (Model: {self.model_name.upper()})")

    def _run_inference(self, folder_path: Path) -> None:
        """
        Executes YOLOv5 detection for a single input folder using subprocess.
        """
        if not folder_path.exists():
            self.logger.warning(f"Source folder not found: {folder_path}")
            return

        exp_name = f"{self.name_prefix}_{folder_path.name}"
        exp_dir = self.detect_output_dir / exp_name

        # Clean up previous results to prevent duplication
        if exp_dir.exists():
            shutil.rmtree(exp_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.logs_dir / f"detect_{folder_path.name}_{timestamp}.log"

        cmd = [
            "python",
            "detect.py",
            "--weights", str(self.saved_model_path),
            "--source", str(folder_path),
            "--project", str(self.detect_output_dir),
            "--name", exp_name,
            "--device", self.device,
        ]

        if self.save_crop:
            cmd.append("--save-crop")
        if self.save_txt:
            cmd.append("--save-txt")
        if self.save_conf:
            cmd.append("--save-conf")

        env = os.environ.copy()
        yolo_path = str(self.yolov5_dir)
        env["PYTHONPATH"] = f"{yolo_path}:{env.get('PYTHONPATH', '')}"

        self.logger.info(f"Running detection on: {folder_path.name}")

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
                f"Detection failed for {folder_path.name} (Code: {process.returncode}). Check log: {log_path}"
            )
        else:
            self.logger.info(f"Results saved to: {exp_dir}")

    def run(self) -> None:
        """
        Orchestrates the inference process for all subdirectories in the input root.
        """
        if self.demo_mode:
            self.logger.info("Demo mode: Downloading fine-tuned weights")
            download_fine_tuned_weights(
                cfg=self.cfg,
                model_name=self.model_name,
                saved_model_path=self.saved_model_path,
                logger=self.logger,
            )

        subfolders = [p for p in self.input_root.iterdir() if p.is_dir()]
        if not subfolders:
            raise FileNotFoundError(f"No subfolders found in {self.input_root}")

        self.logger.info(f"Processing {len(subfolders)} folders")

        for folder in subfolders:
            self._run_inference(folder)