#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_predict.py

Generates a list of image paths for YOLO inference.

This module scans the input dataset directory for images belonging to the
configured categories and aggregates their absolute paths into a single
text file (predict.txt) required by the detection model.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class YOLOPredictListGenerator:
    """
    Generates a model-specific `predict.txt` listing all dataset images.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the generator with configuration settings.

        Args:
            config (Dict[str, Any]): The loaded configuration dictionary.
        """
        self.logger = get_logger("yolo_cropper.YOLOPredictListGenerator")

        # Load configuration sections
        self.cfg = config
        global_main_cfg = self.cfg.get("main", {})
        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        main_cfg = yolo_cropper_cfg.get("main", {})
        dataset_cfg = yolo_cropper_cfg.get("dataset", {})

        # Validation
        self.categories = global_main_cfg.get("categories", [])
        if not self.categories:
            raise ValueError("main.categories must be defined in config.yaml")

        self.model_name = main_cfg.get("model_name")
        if not self.model_name:
            raise ValueError("yolo_cropper.main.model_name must be defined")

        # Path setup
        self.input_root = Path(
            main_cfg.get("input_dir", "data/original")
        ).resolve()

        self.results_root = Path(
            dataset_cfg.get("results_dir", "outputs/json_results")
        ).resolve()

        # Define output path: results_dir / model_name / predict.txt
        self.output_dir = self.results_root / self.model_name
        self.output_path = self.output_dir / "predict.txt"

        # Ensure input exists
        if not self.input_root.exists():
            raise FileNotFoundError(f"Input root not found: {self.input_root}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("YOLOPredictListGenerator initialized")
        self.logger.debug(f"Input root : {self.input_root}")
        self.logger.debug(f"Output txt : {self.output_path}")

    # --------------------------------------------------
    # Core Logic
    # --------------------------------------------------

    def _collect_images(self) -> List[str]:
        """
        Scans class directories for valid image files.

        Returns:
            List[str]: A list of absolute file paths.
        """
        exts = (".jpg", ".jpeg", ".png")
        image_paths: List[str] = []

        for cls in self.categories:
            class_dir = self.input_root / cls
            if not class_dir.exists():
                self.logger.warning(f"Missing class dir: {class_dir}")
                continue

            for p in class_dir.rglob("*"):
                if p.suffix.lower() in exts:
                    image_paths.append(str(p.resolve()))

        if not image_paths:
            raise FileNotFoundError(
                f"No images found under {self.input_root} (categories={self.categories})"
            )

        image_paths.sort()
        return image_paths

    def _write_output(self, image_paths: List[str]) -> None:
        """
        Writes the collected image paths to the output file.

        Args:
            image_paths (List[str]): List of absolute image paths.
        """
        self.output_path.write_text(
            "\n".join(image_paths) + "\n",
            encoding="utf-8",
        )

        self.logger.info(f"predict.txt generated → {self.output_path}")
        self.logger.info(f" - Total images : {len(image_paths)}")

    def run(self) -> None:
        """
        Executes the list generation process.
        """
        images = self._collect_images()
        self._write_output(images)