#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cropper.py
-----------
This module extracts and saves object regions detected by YOLO models
based on the bounding box information stored in `result.json`.

It reads the YOLO detection results, crops the corresponding regions
from the original images, and organizes them into class-specific folders
(e.g., `repair`, `replace`). Images without detections are copied as-is.

In short, this script turns YOLO detection outputs into a clean,
cropped dataset ready for training or analysis.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import cv2

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class YOLOCropper:
    """
    Crops detected regions from original images using YOLO detection results.

    The cropper reads `result.json` and `predict.txt`, determines the
    bounding boxes for each detection, and saves cropped regions (or
    original images if no detection) into structured output directories.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize cropper with configuration and paths."""
        self.logger = get_logger("yolo_cropper.Cropper")

        # Load configuration
        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.cropper_cfg = self.yolo_cropper_cfg.get("cropper", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        input_dir = Path(self.main_cfg.get("input_dir", "data/original"))
        self.dataset_name = input_dir.name
        self.model_name = self.main_cfg.get("model_name", "yolov5")
        self.min_size = int(self.cropper_cfg.get("min_size", 8))
        self.pad = int(self.cropper_cfg.get("pad", 0))

        # Resolve paths
        self.json_path = Path(
            f"{self.dataset_cfg.get('results_dir', 'outputs/json_results')}/{self.model_name}/result.json"
        ).resolve()
        self.predict_list = Path(
            f"{self.dataset_cfg.get('results_dir', 'outputs/json_results')}/predict.txt"
        ).resolve()
        self.output_dir = Path(self.main_cfg.get("output_dir", "data/generation_crop"))

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized Cropper ({self.model_name.upper()})")
        self.logger.debug(f" - Dataset : {self.dataset_name}")
        self.logger.debug(f" - JSON    : {self.json_path}")
        self.logger.debug(f" - Predict : {self.predict_list}")
        self.logger.debug(f" - Output  : {self.output_dir}")

    # --------------------------------------------------------
    def crop_from_json(self):
        """
        Crop detected regions from images according to YOLO `result.json`.

        For each image listed in `predict.txt`:
        - Crops regions based on bounding boxes in `result.json`
        - Saves each cropped patch under its class directory
        - Copies original images when no detection exists
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"result.json not found → {self.json_path}")
        if not self.predict_list.exists():
            raise FileNotFoundError(f"predict.txt not found → {self.predict_list}")

        pred_imgs = [
            ln.strip()
            for ln in self.predict_list.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        pred_set = set(pred_imgs)

        with open(self.json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        processed = set()
        saved_crops, saved_originals = 0, 0

        for item in results:
            img_path = item.get("filename") or item.get("file", "")
            if not img_path or not os.path.exists(img_path):
                continue

            processed.add(img_path)
            img = cv2.imread(img_path)
            if img is None:
                continue

            H, W = img.shape[:2]
            base = os.path.splitext(os.path.basename(img_path))[0]

            parts = os.path.normpath(img_path).split(os.sep)
            class_name = next(
                (c for c in ["repair", "replace"] if c in parts), "unknown"
            )

            out_dir = self.output_dir / class_name
            out_dir.mkdir(exist_ok=True, parents=True)

            dets = item.get("objects", []) or item.get("detections", [])
            crops_here = 0

            for i, det in enumerate(dets, 1):
                rc = det.get("relative_coordinates")
                if not rc:
                    continue

                cx, cy = rc["center_x"] * W, rc["center_y"] * H
                bw, bh = rc["width"] * W, rc["height"] * H
                x1, y1 = int(cx - bw / 2) - self.pad, int(cy - bh / 2) - self.pad
                x2, y2 = int(cx + bw / 2) + self.pad, int(cy + bh / 2) + self.pad
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                if (x2 - x1 < self.min_size) or (y2 - y1 < self.min_size):
                    continue

                crop = img[y1:y2, x1:x2]
                outname = f"{base}_{i}.jpg"
                cv2.imwrite(str(out_dir / outname), crop)
                crops_here += 1
                saved_crops += 1

            # If no detections, copy original image
            if crops_here == 0:
                shutil.copy2(img_path, out_dir / os.path.basename(img_path))
                saved_originals += 1

        # Handle images missing in JSON
        missing = pred_set - processed
        for img_path in missing:
            if not os.path.exists(img_path):
                continue
            parts = os.path.normpath(img_path).split(os.sep)
            class_name = next(
                (c for c in ["repair", "replace"] if c in parts), "unknown"
            )
            out_dir = self.output_dir / class_name
            out_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy2(img_path, out_dir / os.path.basename(img_path))
            saved_originals += 1

        self.logger.info(
            f"Cropping complete ({self.model_name.upper()}) → {self.output_dir}"
        )
        self.logger.info(f"   - Saved Crops   : {saved_crops}")
        self.logger.info(f"   - Saved Originals (No Detection) : {saved_originals}")
