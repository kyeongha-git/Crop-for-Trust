#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
converter.py
------------
Collects YOLO detection results and converts them into a unified JSON file.

Design Principles:
- Class semantics are defined exclusively in `config.yaml (main.categories)`
- YOLO-predicted class_id is trusted as-is
- No folder-name-based class inference (fully class-agnostic)
- Supports YOLOv5 / YOLOv8 unified parsing
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class YOLOConverter:
    """
    Collect and organize YOLO detection results into a unified JSON dataset.
    """

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.YOLOConverter")

        self.cfg = config
        global_main_cfg = self.cfg.get("main", {})
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        self.categories: List[str] = global_main_cfg.get("categories", [])
        if not self.categories:
            raise ValueError(
                "main.categories must be defined in config.yaml for YOLOConverter"
            )
        self.num_classes = len(self.categories)

        self.model_name = (
            self.yolo_cropper_cfg.get("main", {})
            .get("model_name", "yolov5")
            .lower()
        )

        self.detect_root = Path(
            self.dataset_cfg.get("detect_dir", "runs/detect")
        ).resolve()

        self.output_json = Path(
            self.dataset_cfg.get("results_dir", "outputs/json_results")
        ) / self.model_name / "result.json"

        self.data_root = Path(
            self.main_cfg.get("input_dir", "data/original")
        ).resolve()

        self.logger.info(f"Initialized YOLOConverter ({self.model_name.upper()})")
        self.logger.debug(f" - Detect Root : {self.detect_root}")
        self.logger.debug(f" - Output JSON : {self.output_json}")
        self.logger.debug(f" - Data Root   : {self.data_root}")


    def _index_original_images(self) -> Dict[str, Path]:
        """Index original images by stem for fast lookup."""
        image_map = {}
        for p in self.data_root.rglob("*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                image_map[p.stem] = p
        return image_map


    def _parse_detect_folder(
        self, detect_dir: Path, frame_start: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Parse YOLO label files inside a detection folder."""
        label_dir = detect_dir / "labels"
        if not label_dir.exists():
            self.logger.warning(f"[SKIP] {detect_dir} (no labels/)")
            return [], frame_start

        results = []
        frame_id = frame_start

        image_map = self._index_original_images()

        for label_file in sorted(label_dir.glob("*.txt")):
            with open(label_file, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            base = label_file.stem
            orig_path = image_map.get(base)

            if orig_path is None or not orig_path.exists():
                self.logger.warning(
                    f"[WARN] Original image not found for label: {base}"
                )
                continue

            objects = []
            for ln in lines:
                parts = ln.split()
                if len(parts) not in (5, 6):
                    continue

                cls_id = int(parts[0])
                if cls_id < 0 or cls_id >= self.num_classes:
                    self.logger.warning(
                        f"[WARN] Invalid class_id {cls_id} in {label_file.name}"
                    )
                    continue

                x, y, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if len(parts) == 6 else None

                objects.append(
                    {
                        "class_id": cls_id,
                        "name": self.categories[cls_id],
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

        self.logger.info(
            f"[+] Parsed {detect_dir.name} → {len(results)} frames"
        )
        return results, frame_id

    # --------------------------------------------------------
    def run(self):
        """
        Collect YOLO detection outputs from detect directories.
        Any folder containing `labels/` is treated as a valid detection result.
        """

        if not self.detect_root.exists():
            raise FileNotFoundError(f"Detect root not found: {self.detect_root}")

        detect_dirs = [
            d
            for d in sorted(self.detect_root.iterdir())
            if d.is_dir() and (d / "labels").exists()
        ]

        if not detect_dirs:
            raise FileNotFoundError(
                f"No detection folders with labels/ found in {self.detect_root}"
            )

        self.logger.info(
            f"[YOLOConverter] Found {len(detect_dirs)} detection folders:"
        )
        for d in detect_dirs:
            self.logger.info(f" - {d.name}")

        all_results = []
        frame_id = 1

        for detect_dir in detect_dirs:
            results, frame_id = self._parse_detect_folder(
                detect_dir, frame_start=frame_id
            )
            all_results.extend(results)

        # ---- Save JSON ----
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        self.logger.info(f"[DONE] Detection results saved → {self.output_json}")
        self.logger.info(f" - Total frames: {len(all_results)}")
