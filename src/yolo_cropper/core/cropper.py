#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cropper.py
-----------
Unified Cropper (Config-driven)
- Supports YOLOv2 / YOLOv4 / YOLOv5 / YOLOv8 result.json outputs.
- Reads configuration from upper controller or CLI (for debug).

Usage (debug):
    python src/yolo_cropper/core/cropper.py --config utils/config.yaml
"""

import os
import json
import shutil
import cv2
from pathlib import Path
from typing import Dict, Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger, setup_logging
from utils.load_config import load_yaml_config


class YOLOCropper:
    """Unified cropper that handles result.json from multiple YOLO versions."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.Cropper")

        # --------------------------------------------------------
        # Config 구조 해석
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # Paths
        # --------------------------------------------------------
        self.json_path = Path(
            f"{self.dataset_cfg.get('results_dir', 'outputs/json_results')}/{self.model_name}/result.json"
        ).resolve()
        self.predict_list = Path(f"{self.dataset_cfg.get('results_dir', 'outputs/json_results')}/predict.txt"
        ).resolve()
        self.output_dir = Path(self.main_cfg.get('output_dir', "data/generation_crop"))

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized Cropper ({self.model_name.upper()})")
        self.logger.debug(f" - Dataset : {self.dataset_name}")
        self.logger.debug(f" - JSON    : {self.json_path}")
        self.logger.debug(f" - Predict : {self.predict_list}")
        self.logger.debug(f" - Output  : {self.output_dir}")

    # --------------------------------------------------------
    def crop_from_json(self):
        """Perform cropping from result.json"""
        if not self.json_path.exists():
            raise FileNotFoundError(f"❌ result.json not found → {self.json_path}")
        if not self.predict_list.exists():
            raise FileNotFoundError(f"❌ predict.txt not found → {self.predict_list}")

        pred_imgs = [ln.strip() for ln in self.predict_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
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
            class_name = next((c for c in ["repair", "replace"] if c in parts), "unknown")

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

            # 🔹 No detection → copy original
            if crops_here == 0:
                shutil.copy2(img_path, out_dir / os.path.basename(img_path))
                saved_originals += 1

        # 🔹 Images missing in JSON
        missing = pred_set - processed
        for img_path in missing:
            if not os.path.exists(img_path):
                continue
            parts = os.path.normpath(img_path).split(os.sep)
            class_name = next((c for c in ["repair", "replace"] if c in parts), "unknown")
            out_dir = self.output_dir / class_name
            out_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy2(img_path, out_dir / os.path.basename(img_path))
            saved_originals += 1

        self.logger.info(f"[✓] Cropping complete ({self.model_name.upper()}) → {self.output_dir}")
        self.logger.info(f"   - Saved Crops   : {saved_crops}")
        self.logger.info(f"   - Saved Originals (No Detection) : {saved_originals}")


# --------------------------------------------------------
# ✅ CLI Debug Entry
# --------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified Cropper (Config-driven)")
    parser.add_argument("--config", type=str, default="utils/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    setup_logging("logs/yolo_cropper")
    cfg = load_yaml_config(args.config)

    cropper = YOLOCropper(config=cfg)
    cropper.crop_from_json()
