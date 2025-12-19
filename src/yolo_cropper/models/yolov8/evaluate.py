#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8 Evaluation Module.

Computes performance metrics (Precision, Recall, mAP) for the YOLOv8 model
and records the results to a CSV file.
"""

import csv
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger, setup_logging
from utils.load_config import load_yaml_config
from utils.config_manager import ConfigManager


def safe_mean(value: Any) -> float:
    """Computes the mean of a value safely, handling numpy arrays."""
    if hasattr(value, "mean"):
        return float(np.mean(value))
    return float(value)


class YOLOv8Evaluator:
    """
    Manages the evaluation of YOLOv8 models against a specified dataset.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = get_logger("yolo_cropper.YOLOv8Evaluator")
        self.cfg = config

        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.yolov8_cfg = self.yolo_cropper_cfg.get("yolov8", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.train_cfg = self.yolo_cropper_cfg.get("train", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        self.model_name = self.main_cfg.get("model_name", "yolov8s")
        self.data_yaml = Path(
            self.yolov8_cfg.get("data_yaml", "data/yolo_cropper/yolov8/data.yaml")
        ).resolve()
        self.imgsz = self.train_cfg.get("imgsz", 416)

        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()
        self.weights_path = self.saved_model_dir / f"{self.model_name}.pt"

        self.metrics_dir = Path(
            self.dataset_cfg.get("metrics_dir", "metrics/yolo_cropper")
        ).resolve()
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.metrics_dir / f"{self.model_name}_metrics.csv"

        self.logger.info(f"Initialized Evaluator (Model: {self.model_name.upper()})")

    def run(self) -> Dict[str, Any]:
        """
        Executes the evaluation process.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics.
        """
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {self.data_yaml}")

        self.logger.info(f"Starting evaluation on {self.data_yaml}")

        model = YOLO(self.weights_path)
        metrics = model.val(data=str(self.data_yaml), imgsz=self.imgsz, verbose=False)

        # Calculate macro-average for class-wise metrics
        result_dict = {
            "model": self.model_name,
            "precision": safe_mean(metrics.box.p),
            "recall": safe_mean(metrics.box.r),
            "mAP@0.5": safe_mean(metrics.box.map50),
        }

        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        write_header = not self.csv_path.exists()

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=[*result_dict.keys(), "timestamp"]
            )
            if write_header:
                writer.writeheader()
            writer.writerow({**result_dict, "timestamp": timestamp})

        self.logger.info(f"Metrics saved to {self.csv_path}")
        return result_dict


def main() -> None:
    """
    CLI entrypoint for standalone evaluation.
    """
    parser = argparse.ArgumentParser(description="Standalone YOLOv8 Evaluator")
    parser.add_argument(
        "--config",
        type=str,
        default="utils/config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    setup_logging("logs/yolo_cropper")
    logger = get_logger("yolo_cropper.yolov8_eval")

    try:
        # Sync and reload config
        cfg_manager = ConfigManager(args.config)
        cfg_manager.update_paths()
        cfg_manager.save()

        cfg = load_yaml_config(args.config)

        evaluator = YOLOv8Evaluator(cfg)
        metrics = evaluator.run()

        logger.info(f"Evaluation completed: {metrics}")

    except Exception:
        logger.exception("Evaluation failed")
        raise


if __name__ == "__main__":
    main()