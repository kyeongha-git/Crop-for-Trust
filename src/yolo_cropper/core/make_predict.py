#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_predict_list.py
--------------------
This module automatically generates a text file (`predict.txt`)
that lists all image paths in the dataset.

It scans the dataset folders (e.g., `repair` and `replace`)
and records every image path in a single file, which is later
used by YOLO detection scripts to know which images to process.

In short, it creates a complete list of dataset images ready
for YOLO inference or evaluation.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class YOLOPredictListGenerator:
    """
    Generates a `predict.txt` file that lists all image paths under the dataset root.

    The file serves as an input reference for YOLO detection pipelines,
    ensuring that every image in the dataset can be automatically processed.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize generator with paths and configuration."""
        self.logger = get_logger("yolo_cropper.YOLOvPredictListGenerator")

        # Load configuration
        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.yolov5_cfg = self.yolo_cropper_cfg.get("yolov5", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        # Resolve directories
        self.input_root = Path(
            self.main_cfg.get("input_dir", "data/yolo_cropper/original")
        ).resolve()
        self.output_dir = Path(
            self.dataset_cfg.get("results_dir", "outputs/json_results")
        ).resolve()
        self.output_path = self.output_dir / "predict.txt"

        # Validate directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.input_root.exists():
            raise FileNotFoundError(f"Source root not found: {self.input_root}")

        self.logger.info("YOLOPredictListGenerator initialized")
        self.logger.debug(f"Source root : {self.input_root}")
        self.logger.debug(f"Output path : {self.output_path}")

    # ==========================================================
    # Collect image paths
    # ==========================================================
    def _collect_images(self) -> List[str]:
        """
        Collect all image paths under the dataset root.

        Searches within both `repair` and `replace` folders
        and returns a sorted list of image file paths.
        """
        exts = [".jpg", ".jpeg", ".png"]
        all_images = []

        for cls in ["repair", "replace"]:
            class_dir = self.input_root / cls
            if not class_dir.exists():
                self.logger.warning(f"[!] Missing class folder: {class_dir}")
                continue

            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in exts:
                    all_images.append(str(img_path.resolve()))

        if not all_images:
            raise FileNotFoundError(f"No images found under {self.input_root}")

        all_images.sort()
        return all_images

    # ==========================================================
    # Write predict.txt
    # ==========================================================
    def _write_output(self, image_paths: List[str]):
        """Write collected image paths to `predict.txt`."""
        self.output_path.write_text("\n".join(image_paths), encoding="utf-8")
        self.logger.info(f"Generated predict.txt → {self.output_path}")
        self.logger.info(f"   - Dataset root : {self.input_root}")
        self.logger.info(f"   - Total images : {len(image_paths)}")

    # ==========================================================
    # Run full process
    # ==========================================================
    def run(self):
        """Generate the full image list and save to file."""
        images = self._collect_images()
        self._write_output(images)
