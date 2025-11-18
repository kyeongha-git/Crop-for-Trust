#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluator.py
------------
This module evaluates a YOLOv8 model in a config-driven pipeline.

It loads a trained YOLOv8 model, performs validation on the dataset
specified in `config.yaml`, computes key metrics (Precision, Recall,
mAP@0.5, mAP@0.5:0.95), and saves them to a CSV file.

"""

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


def safe_mean(value):
    """
    Safely compute the mean of a scalar or array-like value.

    Returns:
        float: Mean value (converted safely to float).
    """
    if hasattr(value, "mean"):
        return float(np.mean(value))
    return float(value)


class YOLOv8Evaluator:
    """
    Handles YOLOv8 model evaluation and metric computation.

    This class runs the validation pipeline for a trained YOLOv8 model
    using Ultralytics' API, extracts precision, recall, and mean Average
    Precision (mAP) scores, and stores them in a CSV log for tracking
    experiment results.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the YOLOv8 evaluator using a configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration object containing paths and
                evaluation parameters defined in `config.yaml`.
        """
        self.logger = get_logger("yolo_cropper.YOLOv8Evaluator")
        self.cfg = config

        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.yolov8_cfg = self.yolo_cropper_cfg.get("yolov8", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.train_cfg = self.yolo_cropper_cfg.get("train", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        # Paths and parameters
        self.model_name = self.main_cfg.get("model_name", "yolov8s")
        self.data_yaml = Path(
            self.yolov8_cfg.get("data_yaml", "data/yolo_cropper/yolov8/data.yaml")
        ).resolve()
        self.imgsz = self.train_cfg.get("imgsz", 416)
        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()
        self.weights_path = (self.saved_model_dir / f"{self.model_name}.pt").resolve()

        # Output directory for metrics
        self.metrics_dir = Path(
            self.dataset_cfg.get("metrics_dir", "metrics/yolo_cropper")
        ).resolve()
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.metrics_dir / f"{self.model_name}_metrics.csv"

        self.logger.info(f"Initialized YOLOv8Evaluator ({self.model_name.upper()})")
        self.logger.debug(f" - Data YAML  : {self.data_yaml}")
        self.logger.debug(f" - Weights    : {self.weights_path}")
        self.logger.debug(f" - Metrics CSV: {self.csv_path}")

    def run(self):
        """
        Evaluate the YOLOv8 model on the configured dataset.

        This method performs:
            1. Model loading from the saved weights.
            2. Evaluation on the dataset specified in `data.yaml`.
            3. Computation of Precision, Recall, mAP@0.5, and mAP@0.5:0.95.
            4. Logging and saving results to a metrics CSV file.

        Returns:
            dict: A dictionary containing evaluation metrics with the following keys:
                - `"precision"` (float): Detection precision.
                - `"recall"` (float): Detection recall.
                - `"mAP@0.5"` (float): Mean Average Precision at IoU threshold 0.5.
                - `"mAP@0.5:0.95"` (float): Mean Average Precision averaged across IoU thresholds.
                - `"model"` (str): Model name used during evaluation.

        """
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found → {self.weights_path}")
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found → {self.data_yaml}")

        self.logger.info(
            f"Evaluating {self.model_name} on dataset: {self.data_yaml}"
        )

        # Run YOLOv8 validation
        model = YOLO(self.weights_path)
        metrics = model.val(data=str(self.data_yaml), imgsz=self.imgsz, verbose=False)

        # Extract key metrics
        result_dict = {
            "model": self.model_name,
            "precision": safe_mean(metrics.box.p),
            "recall": safe_mean(metrics.box.r),
            "mAP@0.5": safe_mean(metrics.box.map50),
            "mAP@0.5:0.95": safe_mean(metrics.box.map),
        }

        # Save metrics to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[*result_dict.keys(), "timestamp"])
            if write_header:
                writer.writeheader()
            writer.writerow({**result_dict, "timestamp": timestamp})

        self.logger.info(f"Metrics saved → {self.csv_path}")
        self.logger.info(
            f"Precision: {result_dict['precision']:.4f} | Recall: {result_dict['recall']:.4f} | "
            f"mAP@0.5: {result_dict['mAP@0.5']:.4f} | mAP@0.5:0.95: {result_dict['mAP@0.5:0.95']:.4f}"
        )

        return result_dict
