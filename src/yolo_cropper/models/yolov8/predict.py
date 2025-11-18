#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predictor.py
------------
This module performs inference using a YOLOv8 model in a config-driven pipeline.

It loads a trained YOLOv8 model specified in `config.yaml`, runs detection on
image paths listed in `predict.txt`, and saves prediction outputs (labels,
JSONs, and optional crops) to a structured output directory.

"""

import sys
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class YOLOv8Predictor:
    """
    Handles YOLOv8 inference using the Ultralytics YOLO library.

    This class runs object detection on a given image list (from `predict.txt`)
    and manages result storage, logging, and reproducible configuration-based
    execution. It supports optional cropping, label saving, and confidence output.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the YOLOv8 predictor from a configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration object containing model,
                dataset, and inference parameters.
        """
        self.logger = get_logger("yolo_cropper.YOLOv8Predictor")
        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.yolov8_cfg = self.yolo_cropper_cfg.get("yolov8", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.tain_cfg = self.yolo_cropper_cfg.get("train", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})
        self.model_name = self.main_cfg.get("model_name", "yolov8s")

        # --------------------------------------------------------
        # Directories
        # --------------------------------------------------------
        saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()
        self.weights_path = (saved_model_dir / f"{self.model_name}.pt").resolve()
        self.detect_root = Path(
            self.dataset_cfg.get("detect_output_dir", "runs/detect")
        ).resolve()
        self.output_root = Path(
            self.dataset_cfg.get("results_dir", "outputs/json_results")
        )
        self.output_dir = (self.output_root / self.model_name).resolve()

        # Inference options
        self.imgsz = self.tain_cfg.get("imgsz", 416)
        self.save_crop = self.tain_cfg.get("save_crop", False)
        self.save_txt = self.tain_cfg.get("save_txt", True)
        self.save_conf = self.tain_cfg.get("save_conf", True)
        self.quiet = self.yolov8_cfg.get("quiet", True)
        self.predict_txt = (self.output_root / "predict.txt").resolve()

        self.logger.info(f"Initialized YOLOv8Predictor ({self.model_name.upper()})")
        self.logger.debug(f" - Weights : {self.weights_path}")
        self.logger.debug(f" - Predict : {self.predict_txt}")
        self.logger.debug(f" - Output  : {self.output_dir}")

    def run(self):
        """
        Run YOLOv8 detection on all images listed in `predict.txt`.

        This method performs:
            1. Model loading from the specified weights path.
            2. Batch prediction over image paths defined in `predict.txt`.
            3. Output saving (labels, crops, and metadata) to the detection directory.

        Returns:
            tuple[str, str]: A tuple containing:
                - Path to the directory with detection results.
                - Path to the input `predict.txt` file used for inference.

        """
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weight not found: {self.weights_path}")
        if not self.predict_txt.exists():
            raise FileNotFoundError(f"predict.txt not found: {self.predict_txt}")

        # Load model
        model = YOLO(self.weights_path)
        self.logger.info(f"Starting YOLOv8 detection ({self.model_name.upper()})")

        # Execute prediction
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
