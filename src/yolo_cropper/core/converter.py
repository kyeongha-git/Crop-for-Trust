#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
converter.py
------------
This module collects object detection results produced by YOLO models
and converts them into a single, unified JSON file.

It reads detection outputs (from folders such as `repair` and `replace`),
matches each detection with its original image, and stores all bounding box
information in a structured format for later evaluation and analysis.

Simply put, this script merges all YOLO detection results into one
readable and reusable dataset summary.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger

# ============================================================
# Folder → Class Mapping
# ============================================================
FOLDER_TO_CLASS = {
    "repair": {"id": 0, "name": "repair"},
    "replace": {"id": 1, "name": "replace"},
}


def infer_class_from_folder(path: Path) -> Dict[str, Any]:
    """Infer object class (repair / replace) from folder name."""
    parts = [p.lower() for p in path.parts]
    for key, value in FOLDER_TO_CLASS.items():
        if any(key in part for part in parts):
            return value
    return {"id": -1, "name": "unknown"}


# ============================================================
# Main Converter Class
# ============================================================
class YOLOConverter:
    """
    Collects and organizes YOLO detection results into a unified JSON dataset.

    Each detection file (.txt) is parsed to extract bounding boxes,
    matched with its original image path, and stored in a structured format.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize converter with paths and configurations."""
        self.logger = get_logger("yolo_cropper.YOLOConverter")

        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        # Directory paths
        self.model_name = self.main_cfg.get("model_name", "yolov5").lower()
        self.detect_root = Path(
            self.dataset_cfg.get("detect_dir", "runs/detect")
        ).resolve()
        self.output_json = Path(
            f"{self.dataset_cfg.get('results_dir', 'outputs/json_results')}/{self.model_name}/result.json"
        ).resolve()
        self.data_root = Path(self.main_cfg.get("input_dir", "data/original")).resolve()

        self.logger.info(f"Initialized YOLOConverter ({self.model_name.upper()})")
        self.logger.debug(f" - Detect Root : {self.detect_root}")
        self.logger.debug(f" - Output JSON : {self.output_json}")
        self.logger.debug(f" - Data Root   : {self.data_root}")

    # --------------------------------------------------------
    def _parse_detect_folder(
        self, detect_dir: Path, frame_start: int = 1
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Parse all detection text files in a given folder.

        Each YOLO .txt file contains bounding boxes for one image.
        This function reads them, associates with the image, and converts to JSON-friendly format.
        """
        label_dir = detect_dir / "labels"
        if not label_dir.exists():
            self.logger.warning(f"[!] Skipping {detect_dir} (no labels/ folder found)")
            return [], frame_start

        folder_class = infer_class_from_folder(detect_dir)
        class_id, class_name = folder_class["id"], folder_class["name"]

        results = []
        frame_id = frame_start

        for label_file in sorted(label_dir.glob("*.txt")):
            with open(label_file, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            base = label_file.stem
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                cand = detect_dir / f"{base}{ext}"
                if cand.exists():
                    img_path = cand.resolve()
                    break

            # Restore original image path from dataset
            orig_path = self.data_root / class_name / f"{base}.png"
            if not orig_path.exists():
                for ext in [".jpg", ".jpeg"]:
                    cand = self.data_root / class_name / f"{base}{ext}"
                    if cand.exists():
                        orig_path = cand
                        break

            objects = []
            for ln in lines:
                parts = ln.split()
                if len(parts) not in (5, 6):
                    continue
                x, y, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if len(parts) == 6 else None

                objects.append(
                    {
                        "class_id": class_id,
                        "name": class_name,
                        "relative_coordinates": {
                            "center_x": x,
                            "center_y": y,
                            "width": w,
                            "height": h,
                        },
                        "confidence": conf,
                    }
                )

            results.append(
                {
                    "frame_id": frame_id,
                    "filename": str(orig_path.resolve()),
                    "objects": objects,
                }
            )
            frame_id += 1

        self.logger.info(f"[+] Parsed {detect_dir.name} → {len(results)} frames")
        return results, frame_id

    # --------------------------------------------------------
    def run(self):
        """
        Collect detection results from all YOLO output folders
        and save them as a single structured JSON file.
        """
        detect_dirs = sorted(
            [
                p
                for p in self.detect_root.iterdir()
                if p.is_dir() and any(k in p.name for k in ("repair", "replace"))
            ]
        )

        if not detect_dirs:
            raise FileNotFoundError(
                f"No detection folders found in {self.detect_root}"
            )

        all_results = []
        frame_id = 1

        for detect_dir in detect_dirs:
            results, frame_id = self._parse_detect_folder(
                detect_dir, frame_start=frame_id
            )
            all_results.extend(results)

        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        self.logger.info(f"All detection results saved → {self.output_json}")
        self.logger.info(f"   - Total frames: {len(all_results)}")
