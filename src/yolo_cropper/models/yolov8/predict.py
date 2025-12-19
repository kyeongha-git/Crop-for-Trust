#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8 Inference Module.

Executes object detection using the Ultralytics API based on
file paths defined in `predict.txt`.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger
from utils.model_hub import download_fine_tuned_weights


class YOLOv8Predictor:
    """
    Manages the inference workflow for YOLOv8, including model loading
    and result serialization.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = get_logger("yolo_cropper.YOLOv8Predictor")
        self.cfg = config

        # Configuration shortcuts
        self.global_main_cfg = self.cfg.get("main", {})
        self.demo_mode = self.global_main_cfg.get("demo", False)

        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.yolov8_cfg = self.yolo_cropper_cfg.get("yolov8", {})
        self.train_cfg = self.yolo_cropper_cfg.get("train", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        self.model_name = self.main_cfg.get("model_name", "yolov8s")

        # Paths
        saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()

        self.weights_path = (saved_model_dir / f"{self.model_name}.pt").resolve()
        self.detect_root = Path(
            self.dataset_cfg.get("detect_output_dir", "runs/detect")
        ).resolve()

        self.results_root = Path(
            self.dataset_cfg.get("results_dir", "outputs/json_results")
        ).resolve()

        self.predict_txt = (self.results_root / self.model_name / "predict.txt").resolve()

        # Inference parameters
        self.imgsz = self.train_cfg.get("imgsz", 416)
        self.save_crop = bool(self.train_cfg.get("save_crop", False))
        self.save_txt = bool(self.train_cfg.get("save_txt", True))
        self.save_conf = bool(self.train_cfg.get("save_conf", True))
        self.quiet = bool(self.yolov8_cfg.get("quiet", True))

        self.logger.info(f"Initialized Predictor (Model: {self.model_name.upper()})")

    def run(self) -> Tuple[str, str]:
        """
        Executes YOLOv8 detection.

        Returns:
            Tuple[str, str]: A tuple containing the results directory path and the prediction list file path.
        """
        if self.demo_mode:
            self.logger.info("Demo mode: Downloading fine-tuned weights")
            download_fine_tuned_weights(
                cfg=self.cfg,
                model_name=self.model_name,
                saved_model_path=self.weights_path,
                logger=self.logger,
            )

        if not self.predict_txt.exists():
            raise FileNotFoundError(f"Prediction list not found: {self.predict_txt}")

        self.logger.info(f"Loading model: {self.weights_path}")
        model = YOLO(str(self.weights_path))

        self.logger.info(f"Starting Inference ({self.model_name.upper()})")

        # Read source paths
        source = [
            ln.strip()
            for ln in self.predict_txt.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ],

        results = model.predict(
            source=str(self.predict_txt),
            imgsz=self.imgsz,
            save=True,
            save_crop=self.save_crop,
            save_txt=self.save_txt,
            save_conf=self.save_conf,
            project=str(self.detect_root),
            name=self.model_name,
            exist_ok=True,
            verbose=not self.quiet,
        )

        result_dir = Path(results[0].save_dir).resolve()
        self.logger.info(f"Inference completed. Results: {result_dir}")

        return str(result_dir), str(self.predict_txt)