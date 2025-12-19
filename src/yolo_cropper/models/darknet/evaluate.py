#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Darknet Evaluation Module.

Executes the Darknet `detector map` command to evaluate trained YOLOv2 or YOLOv4 models,
parses the output logs for performance metrics, and archives results.
"""

import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import argparse

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from src.yolo_cropper.metrics.metrics import get_metrics_parser
from utils.logging import get_logger, setup_logging
from utils.load_config import load_yaml_config
from utils.config_manager import ConfigManager


class DarknetEvaluator:
    """
    Manages the evaluation of YOLOv2 and YOLOv4 models via Darknet.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = get_logger("yolo_cropper.DarknetEvaluator")

        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.darknet_cfg = self.yolo_cropper_cfg.get("darknet", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        self.darknet_dir = Path(
            self.darknet_cfg.get("darknet_dir", "third_party/darknet")
        ).resolve()
        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        )
        self.metrics_dir = Path(
            self.dataset_cfg.get("metrics_dir", "metrics/yolo_cropper")
        )
        self.log_dir = self.darknet_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = self.main_cfg.get("model_name", "yolov2").lower()

        self.logger.info(f"Initialized Evaluator (Model: {self.model_name.upper()})")

    def run(self) -> Dict[str, Any]:
        """
        Executes the evaluation subprocess and parses metrics.

        Returns:
            Dict[str, Any]: Dictionary containing Precision, Recall, and mAP metrics.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"eval_{self.model_name}_{timestamp}.log"
        log_path = self.log_dir / log_filename
        relative_log_path = f"logs/{log_filename}"

        obj_data = "data/obj.data"
        cfg_path = f"cfg/{self.model_name}-obj.cfg"
        weights_path = Path(
            f"{self.saved_model_dir}/{self.model_name}.weights"
        ).resolve()

        # Construct Darknet map command
        command = (
            f"./darknet detector map {obj_data} {cfg_path} {weights_path} "
            f"-dont_show -iou_thresh 0.5 -points 101 | tee {relative_log_path}"
        )

        self.logger.info(f"Starting Evaluation ({self.model_name.upper()})")

        process = subprocess.run(
            ["bash", "-lc", command],
            cwd=self.darknet_dir,
            shell=False,
        )

        # Darknet exit codes: 0 (success), 1 (error, sometimes non-fatal)
        if process.returncode not in (0, 1):
            raise RuntimeError(f"Evaluation failed (Code: {process.returncode})")
        elif process.returncode == 1:
            self.logger.warning("Darknet exited with code 1 (Non-fatal)")

        self.logger.info(f"Evaluation complete. Log: {log_path}")

        parser = get_metrics_parser(self.model_name)
        metrics = parser(str(log_path))

        self._save_metrics_to_csv(metrics)
        return metrics

    def _save_metrics_to_csv(self, metrics: Dict[str, Any]) -> None:
        """
        Appends the calculated metrics to a CSV file.
        """
        save_dir = self.metrics_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        csv_path = save_dir / f"{self.model_name}_metrics.csv"
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
    parser = argparse.ArgumentParser(description="Standalone Darknet Evaluator")
    parser.add_argument(
        "--config",
        type=str,
        default="utils/config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    setup_logging("logs/yolo_cropper")
    logger = get_logger("yolo_cropper.darknet_eval")

    try:
        cfg_manager = ConfigManager(args.config)
        cfg_manager.update_paths()
        cfg_manager.save()

        cfg = load_yaml_config(args.config)

        evaluator = DarknetEvaluator(cfg)
        metrics = evaluator.run()

        logger.info(f"Evaluation completed: {metrics}")

    except Exception:
        logger.exception("Evaluation failed")
        raise


if __name__ == "__main__":
    main()