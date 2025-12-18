#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_predict_list.py
--------------------
Generate predict.txt listing all image paths in the dataset.

- Fully multi-class aware
- Model-aware output directory
- Shared by YOLOv5 / YOLOv8 / Darknet
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

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.YOLOvPredictListGenerator")

        # ---- Load config ----
        self.cfg = config
        global_main_cfg = self.cfg.get("main", {})
        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        main_cfg = yolo_cropper_cfg.get("main", {})
        dataset_cfg = yolo_cropper_cfg.get("dataset", {})

        # ---- Required settings ----
        self.categories = global_main_cfg.get("categories", [])
        if not self.categories:
            raise ValueError("main.categories must be defined in config.yaml")

        self.model_name = main_cfg.get("model_name")
        if not self.model_name:
            raise ValueError("yolo_cropper.main.model_name must be defined")

        # ---- Paths ----
        self.input_root = Path(
            main_cfg.get("input_dir", "data/original")
        ).resolve()

        self.results_root = Path(
            dataset_cfg.get("results_dir", "outputs/json_results")
        ).resolve()

        # 👉 핵심: model_name 하위에 predict.txt 생성
        self.output_dir = self.results_root / self.model_name
        self.output_path = self.output_dir / "predict.txt"

        # ---- Validate & prepare ----
        if not self.input_root.exists():
            raise FileNotFoundError(f"Input root not found: {self.input_root}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("YOLOPredictListGenerator initialized")
        self.logger.debug(f"Input root : {self.input_root}")
        self.logger.debug(f"Output txt : {self.output_path}")

    # --------------------------------------------------
    def _collect_images(self) -> List[str]:
        exts = (".jpg", ".jpeg", ".png")
        image_paths: List[str] = []

        for cls in self.categories:
            class_dir = self.input_root / cls
            if not class_dir.exists():
                self.logger.warning(f"[SKIP] Missing class dir: {class_dir}")
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

    # --------------------------------------------------
    def _write_output(self, image_paths: List[str]):
        self.output_path.write_text(
            "\n".join(image_paths) + "\n",
            encoding="utf-8",
        )

        self.logger.info(f"predict.txt generated → {self.output_path}")
        self.logger.info(f"  - Total images : {len(image_paths)}")

    # --------------------------------------------------
    def run(self):
        images = self._collect_images()
        self._write_output(images)
