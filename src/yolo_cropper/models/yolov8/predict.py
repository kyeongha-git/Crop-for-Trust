#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predictor.py
------------
YOLOv8 inference module (config-driven).

Runs YOLOv8 detection on image paths listed in `predict.txt`
and saves results to a structured output directory.
"""

import sys
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger
from utils.model_hub import download_fine_tuned_weights


class YOLOv8Predictor:
    """
    Handles YOLOv8 inference using the Ultralytics YOLO API.
    """

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.YOLOv8Predictor")
        self.cfg = config

        # --------------------------------------------------------
        # Config shortcuts
        # --------------------------------------------------------
        self.global_main_cfg = self.cfg.get("main", {})
        self.demo_mode = self.global_main_cfg.get("demo", False)

        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.yolov8_cfg = self.yolo_cropper_cfg.get("yolov8", {})
        self.train_cfg = self.yolo_cropper_cfg.get("train", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        self.model_name = self.main_cfg.get("model_name", "yolov8s")

        # --------------------------------------------------------
        # Paths
        # --------------------------------------------------------
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

        self.output_dir = (self.results_root / self.model_name).resolve()
        self.predict_txt = (self.results_root / "predict.txt").resolve()

        # --------------------------------------------------------
        # Inference options
        # --------------------------------------------------------
        self.imgsz = self.train_cfg.get("imgsz", 416)
        self.save_crop = bool(self.train_cfg.get("save_crop", False))
        self.save_txt = bool(self.train_cfg.get("save_txt", True))
        self.save_conf = bool(self.train_cfg.get("save_conf", True))
        self.quiet = bool(self.yolov8_cfg.get("quiet", True))

        self.logger.info(f"Initialized YOLOv8Predictor ({self.model_name.upper()})")
        self.logger.debug(f" - Weights : {self.weights_path}")
        self.logger.debug(f" - Predict : {self.predict_txt}")
        self.logger.debug(f" - Output  : {self.output_dir}")

    def run(self):
        """
        Run YOLOv8 detection using image paths listed in `predict.txt`.
        """
        if self.demo_mode:
            self.logger.info("Demo mode → Download fine-tuned YOLO weights")
            download_fine_tuned_weights(
                cfg=self.cfg,
                model_name=self.model_name,
                saved_model_path=self.weights_path,
                logger=self.logger,
            )

        if not self.predict_txt.exists():
            raise FileNotFoundError(f"predict.txt not found: {self.predict_txt}")

        self.logger.info(f"Loading YOLOv8 model from: {self.weights_path}")
        model = YOLO(str(self.weights_path))

        self.logger.info(f"Starting YOLOv8 detection ({self.model_name.upper()})")

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
        self.logger.info(f"Detection complete → {result_dir}")

        return str(result_dir), str(self.predict_txt)
