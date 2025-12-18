#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cropper.py
-----------
Extracts and saves object regions detected by YOLO models
based on bounding box information stored in `result.json`.

Design Principles:
- Class semantics are defined exclusively in `config.yaml (main.categories)`
- Crops are organized by YOLO-predicted class labels
- No folder- or path-based class inference
- Fully supports multi-class detection scenarios
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Set

import cv2

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger

class YOLOCropper:
    """
    Crop detected regions from images using YOLO detection results.
    All outputs are stored under output_dir grouped by GT class.
    """

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.Cropper")

        # ---- Load configuration ----
        self.cfg = config
        global_main_cfg = self.cfg.get("main", {})
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.cropper_cfg = self.yolo_cropper_cfg.get("cropper", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        self.categories = global_main_cfg.get("categories", [])
        if not self.categories:
            raise ValueError("main.categories must be defined in config.yaml")

        self.min_size = int(self.cropper_cfg.get("min_size", 8))
        self.pad = int(self.cropper_cfg.get("pad", 0))
        self.model_name = self.main_cfg.get("model_name", "yolov5")

        # ---- Paths ----
        results_root = Path(
            self.dataset_cfg.get("results_dir", "outputs/json_results")
        )

        self.json_path = results_root / self.model_name / "result.json"
        self.predict_list = results_root / self.model_name / "predict.txt"

        self.output_dir = Path(
            self.main_cfg.get("output_dir", "data/generation_crop")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for cls in self.categories:
            (self.output_dir / cls).mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    def _infer_gt_class(self, img_path: Path) -> str | None:
        """
        Infer GT class from image path using main.categories.
        """
        for cls in self.categories:
            if cls in img_path.parts:
                return cls
        return None

    # --------------------------------------------------
    def run(self):
        if not self.json_path.exists():
            raise FileNotFoundError(f"result.json not found → {self.json_path}")
        if not self.predict_list.exists():
            raise FileNotFoundError(f"predict.txt not found → {self.predict_list}")

        # ---- Load predict list (input universe) ----
        pred_imgs: Set[Path] = {
            Path(ln.strip())
            for ln in self.predict_list.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }

        # ---- Load YOLO results ----
        with open(self.json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        processed: Set[Path] = set()
        images_with_detection: Set[Path] = set()
        images_without_detection: Set[Path] = set()

        total_crops = 0

        # Process images appearing in result.json
        for item in results:
            img_path = Path(item.get("filename", ""))
            if not img_path.exists():
                continue

            gt_class = self._infer_gt_class(img_path)
            if gt_class is None:
                self.logger.warning(f"[SKIP] Cannot infer GT class: {img_path}")
                continue

            processed.add(img_path)

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            H, W = img.shape[:2]
            base = img_path.stem
            suffix = img_path.suffix  # includes leading dot
            dets = item.get("objects", [])

            crops_in_image = 0
            out_dir = self.output_dir / gt_class

            # ---- Crop loop ----
            for idx, det in enumerate(dets, 1):
                rc = det.get("relative_coordinates")
                if not rc:
                    continue

                cx, cy = rc["center_x"] * W, rc["center_y"] * H
                bw, bh = rc["width"] * W, rc["height"] * H

                x1 = int(cx - bw / 2) - self.pad
                y1 = int(cy - bh / 2) - self.pad
                x2 = int(cx + bw / 2) + self.pad
                y2 = int(cy + bh / 2) + self.pad

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                if (x2 - x1 < self.min_size) or (y2 - y1 < self.min_size):
                    continue

                crop = img[y1:y2, x1:x2]
                out_name = f"{base}_{idx}{suffix}"

                cv2.imwrite(str(out_dir / out_name), crop)

                crops_in_image += 1
                total_crops += 1

            # ---- Image-level decision ----
            if crops_in_image > 0:
                images_with_detection.add(img_path)

                # ensure original image does not coexist
                orig_out = out_dir / img_path.name
                if orig_out.exists():
                    orig_out.unlink()
            else:
                images_without_detection.add(img_path)
                shutil.copy2(img_path, out_dir / img_path.name)

        # Images never appearing in result.json → no detection
        for img_path in pred_imgs - processed:
            if not img_path.exists():
                continue

            gt_class = self._infer_gt_class(img_path)
            if gt_class is None:
                continue

            images_without_detection.add(img_path)
            shutil.copy2(
                img_path,
                self.output_dir / gt_class / img_path.name,
            )

        # Logging
        self.logger.info(f"Cropping complete → {self.output_dir}")
        self.logger.info(f" - Saved Crops        : {total_crops}")
        self.logger.info(
            f" - No-detection imgs : {len(images_without_detection)}"
        )