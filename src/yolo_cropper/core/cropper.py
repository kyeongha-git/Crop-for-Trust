#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cropper.py

Extracts and saves object regions detected by YOLO models.

This module parses detection results from `result.json`, converts relative
coordinates to absolute pixel values, and saves the cropped regions.
If no objects are detected, the original image is preserved to maintain dataset integrity.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set

import cv2

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class YOLOCropper:
    """
    Controller for cropping detected objects from images.

    Organizes outputs by Ground Truth (GT) class folders defined in the configuration.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the cropper with configuration settings.

        Args:
            config (Dict[str, Any]): The loaded configuration dictionary.
        """
        self.logger = get_logger("yolo_cropper.Cropper")

        # Configuration setup
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

        # Path setup
        results_root = Path(
            self.dataset_cfg.get("results_dir", "outputs/json_results")
        )

        self.json_path = results_root / self.model_name / "result.json"
        self.predict_list = results_root / self.model_name / "predict.txt"

        self.output_dir = Path(
            self.main_cfg.get("output_dir", "data/generation_crop")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create output directories for each class
        for cls in self.categories:
            (self.output_dir / cls).mkdir(parents=True, exist_ok=True)

    def _infer_gt_class(self, img_path: Path) -> Optional[str]:
        """
        Infers the Ground Truth class from the image file path.

        Args:
            img_path (Path): Path to the image file.

        Returns:
            Optional[str]: Class name if found, else None.
        """
        for cls in self.categories:
            if cls in img_path.parts:
                return cls
        return None

    def run(self) -> None:
        """
        Executes the cropping pipeline.

        Iterates through detection results, calculates bounding box coordinates,
        crops the objects, and saves them. Handles cases with no detections by
        copying the original image.
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"result.json not found → {self.json_path}")
        if not self.predict_list.exists():
            raise FileNotFoundError(f"predict.txt not found → {self.predict_list}")

        # Load the universe of input images (from prediction list)
        pred_imgs: Set[Path] = {
            Path(ln.strip())
            for ln in self.predict_list.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }

        # Load detection results
        with open(self.json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        processed: Set[Path] = set()
        images_with_detection: Set[Path] = set()
        images_without_detection: Set[Path] = set()

        total_crops = 0

        # Process detected images
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

            # Crop detected objects
            for idx, det in enumerate(dets, 1):
                rc = det.get("relative_coordinates")
                if not rc:
                    continue

                # Convert relative coordinates (center_x, center_y, w, h) to absolute (x1, y1, x2, y2)
                cx, cy = rc["center_x"] * W, rc["center_y"] * H
                bw, bh = rc["width"] * W, rc["height"] * H

                x1 = int(cx - bw / 2) - self.pad
                y1 = int(cy - bh / 2) - self.pad
                x2 = int(cx + bw / 2) + self.pad
                y2 = int(cy + bh / 2) + self.pad

                # Clip to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                if (x2 - x1 < self.min_size) or (y2 - y1 < self.min_size):
                    continue

                crop = img[y1:y2, x1:x2]
                out_name = f"{base}_{idx}{suffix}"

                cv2.imwrite(str(out_dir / out_name), crop)

                crops_in_image += 1
                total_crops += 1

            # Handling Empty Detections
            if crops_in_image > 0:
                images_with_detection.add(img_path)
                
                # Cleanup: Ensure original image copy is removed if crops exist
                orig_out = out_dir / img_path.name
                if orig_out.exists():
                    orig_out.unlink()
            else:
                images_without_detection.add(img_path)
                shutil.copy2(img_path, out_dir / img_path.name)

        # Process images with zero detections (missing from result.json)
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

        self.logger.info(f"Cropping complete → {self.output_dir}")
        self.logger.info(f" - Saved Crops       : {total_crops}")
        self.logger.info(
            f" - No-detection imgs : {len(images_without_detection)}"
        )