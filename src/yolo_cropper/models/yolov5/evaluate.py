#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv5 Evaluation Module.

Wraps the YOLOv5 `val.py` script to evaluate model performance,
parse metrics, and record results to CSV.
"""

import csv
import subprocess
import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from src.yolo_cropper.metrics.metrics import get_metrics_parser
from utils.logging import get_logger, setup_logging
from utils.load_config import load_yaml_config
from utils.config_manager import ConfigManager


class YOLOv5Evaluator:
    """
    Manages the execution of YOLOv5 validation and metric aggregation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = get_logger("yolo_cropper.YOLOv5Evaluator")

        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.yolov5_cfg = self.yolo_cropper_cfg.get("yolov5", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        self.yolov5_dir = Path(
            self.yolov5_cfg.get("yolov5_dir", "third_party/yolov5")
        ).resolve()

        self.data_yaml_path = Path(
            self.yolov5_cfg.get("data_yaml", "data/yolo_cropper/yolov5/data.yaml")
        ).resolve()

        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()

        self.metrics_dir = Path(
            self.dataset_cfg.get("metrics_dir", "metrics/yolo_cropper")
        ).resolve()

        self.log_dir = self.yolov5_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = self.main_cfg.get("model_name", "yolov5").lower()
        self.weights_path = self.saved_model_dir / f"{self.model_name}.pt"

        self.logger.info(f"Initialized Evaluator (Model: {self.model_name.upper()})")
        self.data_yaml = self._resolve_data_yaml(self.data_yaml_path)

    def _resolve_data_yaml(self, data_yaml_path: Path) -> Path:
        """
        Converts relative paths in data.yaml to absolute paths for compatibility.

        Returns:
            Path: Path to the temporary resolved YAML file.
        """
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        base = data_yaml_path.parent.resolve()
        for key in ("train", "val", "test"):
            if key in data and isinstance(data[key], str):
                abs_path = (base / data[key]).resolve()
                if not abs_path.exists() and (abs_path / "images").exists():
                    abs_path = abs_path / "images"
                data[key] = str(abs_path)

        tmp_dir = Path(tempfile.mkdtemp(prefix="yolov5_datayaml_"))
        resolved_yaml = tmp_dir / "data_resolved.yaml"

        with open(resolved_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

        return resolved_yaml

    def run(self) -> Dict[str, Any]:
        """
        Executes the validation subprocess and returns parsed metrics.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"val_{timestamp}.log"

        cmd = [
            "python",
            "val.py",
            "--data", str(self.data_yaml),
            "--weights", str(self.weights_path),
            "--task", "val",
            "--save-json",
        ]

        self.logger.info(f"Starting validation for {self.model_name}")

        with open(log_path, "w", encoding="utf-8") as log_f:
            process = subprocess.run(
                cmd, cwd=self.yolov5_dir, stdout=log_f, stderr=subprocess.STDOUT
            )

        if process.returncode != 0:
            raise RuntimeError(f"YOLOv5 evaluation failed. Check log: {log_path}")

        parser = get_metrics_parser(self.model_name)
        metrics = parser(str(log_path))

        self._save_metrics_to_csv(metrics)
        return metrics

    def _save_metrics_to_csv(self, metrics: Dict[str, Any]) -> None:
        """
        Appends the calculated metrics to a CSV file.
        """
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.metrics_dir / f"{self.model_name}_metrics.csv"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        row = {
            "model": self.model_name,
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "mAP@0.5": metrics.get("mAP@0.5"),
            "timestamp": timestamp,
        }

        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self.logger.info(f"Metrics saved to {csv_path}")


def main() -> None:
    """
    CLI entrypoint for standalone evaluation.
    """
    parser = argparse.ArgumentParser(description="Standalone YOLOv5 Evaluator")
    parser.add_argument(
        "--config",
        type=str,
        default="utils/config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    setup_logging("logs/yolo_cropper")
    logger = get_logger("yolo_cropper.yolov5_eval")

    try:
        cfg_manager = ConfigManager(args.config)
        cfg_manager.update_paths()
        cfg_manager.save()

        cfg = load_yaml_config(args.config)

        evaluator = YOLOv5Evaluator(cfg)
        metrics = evaluator.run()

        logger.info(f"Evaluation completed: {metrics}")

    except Exception:
        logger.exception("Evaluation failed")
        raise


if __name__ == "__main__":
    main()