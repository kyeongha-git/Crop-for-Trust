#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict.py
----------
YOLOv5Predictor (Config-driven)
- Automatically runs detection for each subfolder under the specified source root.
- Config-driven: no YAML loading inside; receives config dict from controller.
- Deletes existing result folders before re-running detection.
"""

import subprocess
from pathlib import Path
from datetime import datetime
import shutil
from typing import Dict, Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class YOLOv5Predictor:
    """Handles YOLOv5 detection using config-driven settings."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.YOLOv5Predictor")

        # --------------------------------------------------------
        # Parse config
        # --------------------------------------------------------
        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.yolov5_cfg = self.yolo_cropper_cfg.get("yolov5", {})
        self.train_Cfg = self.yolo_cropper_cfg.get("train", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})
        self.model_name = self.main_cfg.get("model_name", "yolov5")

        # --------------------------------------------------------
        # Directories
        # --------------------------------------------------------
        self.yolov5_dir = Path(self.yolov5_cfg.get("yolov5_dir", "third_party/yolov5")).resolve()
        self.saved_model_dir = Path(self.dataset_cfg.get('saved_model_dir', 'saved_model/yolo_cropper')).resolve()
        self.input_root = Path(self.main_cfg.get("input_dir", "data/original")).resolve()
        self.detect_root = Path(self.dataset_cfg.get("detect_output_dir", "runs/detect")).resolve()

        self.saved_model_path = self.saved_model_dir / f"{self.model_name}.pt"
        self.logs_dir = self.yolov5_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.detect_root.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------
        # Inference options
        # --------------------------------------------------------
        self.device = str(self.train_Cfg.get("device", "0"))
        self.save_crop = bool(self.train_Cfg.get("save_crop", False))
        self.save_txt = bool(self.train_Cfg.get("save_txt", True))
        self.save_conf = bool(self.train_Cfg.get("save_conf", True))
        self.name_prefix = self.model_name

        # --------------------------------------------------------
        self.logger.info(f"YOLOv5Predictor initialized ({self.model_name.upper()})")
        self.logger.debug(f"Repo Dir   : {self.yolov5_dir}")
        self.logger.debug(f"Weights    : {self.saved_model_path}")
        self.logger.debug(f"Source Dir : {self.input_root}")
        self.logger.debug(f"Output Dir : {self.detect_root}")

    # ==========================================================
    # 🔹 Single-folder inference
    # ==========================================================
    def _run_inference(self, folder_path: Path):
        """Run YOLOv5 detection for a single folder."""
        if not folder_path.exists():
            self.logger.warning(f"[!] Source folder not found: {folder_path}")
            return

        exp_name = f"{self.name_prefix}_{folder_path.name}"
        exp_dir = self.detect_root / exp_name

        # 🔹 기존 결과 폴더 삭제 (exp, exp2 방지)
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
            self.logger.warning(f"[!] Existing result folder deleted → {exp_dir}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.logs_dir / f"detect_{folder_path.name}_{timestamp}.log"

        cmd = [
            "python", "detect.py",
            "--weights", str(self.saved_model_path),
            "--source", str(folder_path),
            "--project", str(self.detect_root),
            "--name", exp_name,
            "--device", self.device,
        ]

        if self.save_crop:
            cmd.append("--save-crop")
        if self.save_txt:
            cmd.append("--save-txt")
        if self.save_conf:
            cmd.append("--save-conf")

        self.logger.info(f"🚀 Running YOLOv5 detection → {folder_path.name}")
        self.logger.debug(f"Command: {' '.join(cmd)}")

        with open(log_path, "w", encoding="utf-8") as log_f:
            process = subprocess.run(cmd, cwd=self.yolov5_dir, stdout=log_f, stderr=subprocess.STDOUT)

        if process.returncode != 0:
            self.logger.error(f"[!] Detection failed ({folder_path.name}), code={process.returncode}. See log: {log_path}")
        else:
            self.logger.info(f"[✓] Detection complete → {exp_dir}")

    # ==========================================================
    # 🔹 Multi-folder inference
    # ==========================================================
    def run(self):
        """Run YOLOv5 detection for each subfolder under input_root."""
        subfolders = [p for p in self.input_root.iterdir() if p.is_dir()]
        if not subfolders:
            raise FileNotFoundError(f"No subfolders found under {self.input_root}")

        self.logger.info(f"🔍 Found {len(subfolders)} folders → {[p.name for p in subfolders]}")
        for folder in subfolders:
            self._run_inference(folder)

        self.logger.info("[✓] All detections complete.")
