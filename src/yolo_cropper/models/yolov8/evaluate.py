#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluator.py
------------
This module evaluates a YOLOv8 model in a config-driven pipeline.
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

from utils.logging import get_logger, setup_logging
from utils.load_config import load_yaml_config
from utils.config_manager import ConfigManager


def safe_mean(value):
    if hasattr(value, "mean"):
        return float(np.mean(value))
    return float(value)


class YOLOv8Evaluator:
    """
    Handles YOLOv8 model evaluation and metric computation.
    """

    def __init__(self, config: Dict[str, Any]):
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

        self.logger.info(f"Initialized YOLOv8Evaluator ({self.model_name.upper()})")

    # --------------------------------------------------
    def run(self):
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found → {self.weights_path}")
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found → {self.data_yaml}")

        self.logger.info(
            f"Evaluating {self.model_name.upper()} on dataset: {self.data_yaml}"
        )

        model = YOLO(self.weights_path)
        metrics = model.val(data=str(self.data_yaml), imgsz=self.imgsz, verbose=False)

        result_dict = {
            "model": self.model_name,
            "precision": safe_mean(metrics.box.p),
            "recall": safe_mean(metrics.box.r),
            "mAP@0.5": safe_mean(metrics.box.map50),
            "mAP@0.5:0.95": safe_mean(metrics.box.map),
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

        self.logger.info(f"Metrics saved → {self.csv_path}")
        return result_dict


# ======================================================
# Standalone Entrypoint
# ======================================================
def main():
    """
    Standalone entrypoint for YOLOv8Evaluator.

    Example:
        python src/yolo_cropper/models/yolov8/evaluator.py --config utils/config.yaml
    """
    import argparse

    parser = argparse.ArgumentParser(description="Standalone YOLOv8 Evaluator")
    parser.add_argument(
        "--config",
        type=str,
        default="utils/config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    setup_logging("logs/yolo_cropper")
    logger = get_logger("yolo_cropper.yolov8_eval")

    logger.info("Starting standalone YOLOv8 evaluation")
    logger.info(f"Using config: {args.config}")

    try:
        # 1. Sync config (overwrite)
        cfg_manager = ConfigManager(args.config)
        cfg_manager.update_paths()
        cfg_manager.save()

        # 2. Reload updated config
        cfg = load_yaml_config(args.config)

        # 3. Run evaluation
        evaluator = YOLOv8Evaluator(cfg)
        metrics = evaluator.run()

        logger.info(f"YOLOv8 evaluation finished successfully → {metrics}")

    except Exception:
        logger.exception("YOLOv8 evaluation failed")
        raise


if __name__ == "__main__":
    main()
