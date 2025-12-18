#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate.py
-------------------
Provides an integrated evaluation module for comparing original and generated images.

Fully configuration-driven:
- Receives only `config`
- Reads all paths, metrics, and YOLO settings internally
- Exposes a single public API: run()
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import json
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.annotation_cleaner.metrics.metrics import (
    edge_iou,
    l1_distance,
    ssim_score,
)
from utils.logging import get_logger, setup_logging


class Evaluator:
    """
    Evaluates image quality between original and generated images using various metrics.
    """

    METRIC_MAP = {
        "l1": l1_distance,
        "ssim": ssim_score,
        "edge_iou": edge_iou,
    }

    def __init__(self, config: Dict):
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("annotation_cleaner.Evaluator")

        self.cfg = config
        global_main_cfg = self.cfg.get("main", {})
        cleaner_cfg = self.cfg.get("annotation_cleaner", {})
        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.eval_cfg = cleaner_cfg.get("evaluate", {})
        self.main_cfg = cleaner_cfg.get("main", {})
        self.debug = self.eval_cfg.get("debug", True)

        self.orig_dir = Path(self.eval_cfg.get("orig_dir")).resolve()
        self.gen_dir = Path(self.eval_cfg.get("gen_dir")).resolve()
        self.metric_dir = Path(self.eval_cfg.get("metric_dir", "metrics/annotation_cleaner")).resolve()
        self.debug_dir = self.metric_dir / "debug_crops"
        self.yolo_model = global_main_cfg.get("yolo_model", "yolov5")
        self.metadata_root = (
            Path(
                yolo_cropper_cfg.get("dataset", {})
                .get("results_dir", "outputs/json_results")
            )
            / self.yolo_model
            / "result.json"
        ).resolve()
        self.metrics: List[str] = self.eval_cfg.get(
            "metrics", ["ssim", "l1", "edge_iou"]
        )
        self.categories: List[str] = global_main_cfg.get(
            "categories", ["repair", "replace"]
        )
        test_mode = "test" in self.gen_dir.name

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        self.logger.info("Initialized Evaluator")
        self.logger.info(f" - Original dir : {self.orig_dir}")
        self.logger.info(f" - Generated dir: {self.gen_dir}")
        self.logger.info(f" - Metric dir   : {self.metric_dir}")
        self.logger.info(f" - Metrics      : {self.metrics}")
        self.logger.info(f" - Test Mode    : {test_mode}")

    def _rel_to_abs_bbox(self, bbox, img_w, img_h):
        cx, cy = bbox["center_x"], bbox["center_y"]
        w, h = bbox["width"], bbox["height"]

        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)

        return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

    def _compute_metrics(self, orig_img, gen_img) -> Dict[str, float]:
        results = {}
        for metric_name in self.metrics:
            func = self.METRIC_MAP.get(metric_name)
            if not func:
                self.logger.warning(f"Unsupported metric: {metric_name}")
                continue
            try:
                val = func(orig_img, gen_img)
                key = metric_name.upper() if metric_name != "edge_iou" else "Edge_IoU"
                results[key] = float(val)
            except Exception as e:
                self.logger.error(f"{metric_name} failed: {e}")
        return results

    # Evaluate Global Consistency
    def _evaluate_full_images(self, save_path: Path) -> Optional[Dict[str, float]]:
        self.logger.info("[1/2] Full Image Evaluation")
        results = []

        for split in self.categories:
            o_dir = self.orig_dir / split
            g_dir = self.gen_dir / split

            if not o_dir.exists() or not g_dir.exists():
                self.logger.warning(f"Missing folder: {split}")
                continue

            o_files = {f.stem: f for f in o_dir.glob("*.[jp][pn]g")}
            g_files = {f.stem: f for f in g_dir.glob("*.[jp][pn]g")}
            common = set(o_files) & set(g_files)

            for name in tqdm(common, desc=split):
                o_img = cv2.imread(str(o_files[name]))
                g_img = cv2.imread(str(g_files[name]))
                if o_img is None or g_img is None:
                    continue
                if o_img.shape != g_img.shape:
                    g_img = cv2.resize(g_img, (o_img.shape[1], o_img.shape[0]))

                results.append(
                    {"split": split, "file": name, **self._compute_metrics(o_img, g_img)}
                )

        if not results:
            return None

        df = pd.DataFrame(results)
        avg = df.drop(columns=["split", "file"]).mean().to_dict()

        avg_row = {**{k: "" for k in df.columns}, **avg}
        avg_row["split"] = "AVG"
        avg_row["file"] = "AVG"

        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
        self.metric_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

        return avg

    # Evaluate Local Representation    
    def _evaluate_crop_images(self, save_path: Path) -> Optional[Dict[str, float]]:
        self.logger.info("[2/2] Metadata-based Crop Evaluation")

        metadata_path = self.metadata_root
        if not metadata_path.exists():
            self.logger.error(f"Metadata not found: {metadata_path}")
            return None

        with open(metadata_path, "r") as f:
            records = json.load(f)

        results = []

        # Visualization directory
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        for record_idx, record in enumerate(tqdm(records, desc="Metadata evaluation")):
            record_path = Path(record["filename"])
            split = record_path.parent.name
            fname = record_path.name
            stem = record_path.stem
            ext = record_path.suffix.lower()

            orig_path = self.orig_dir / split / fname
            gen_path  = self.gen_dir  / split / fname

            if not orig_path.exists() or not gen_path.exists():
                continue

            o_img = cv2.imread(str(orig_path))
            g_img = cv2.imread(str(gen_path))
            if o_img is None or g_img is None:
                continue

            h, w = o_img.shape[:2]

            for idx, obj in enumerate(record.get("objects", [])):
                bbox = obj.get("relative_coordinates", None)
                if bbox is None:
                    continue

                x1, y1, x2, y2 = self._rel_to_abs_bbox(bbox, w, h)

                c1 = o_img[y1:y2, x1:x2]
                c2 = g_img[y1:y2, x1:x2]

                if c1.size == 0 or c2.size == 0:
                    continue

                if c1.shape != c2.shape:
                    c2 = cv2.resize(c2, (c1.shape[1], c1.shape[0]))
                
                diff = cv2.absdiff(c1, c2)

                if record_idx < 5:
                    cv2.imwrite(
                        str(self.debug_dir / f"{stem}_{idx}_orig{ext}"), c1
                    )
                    cv2.imwrite(
                        str(self.debug_dir / f"{stem}_{idx}_gen{ext}"), c2
                    )
                    cv2.imwrite(
                        str(self.debug_dir / f"{stem}_{idx}_diff{ext}"), diff
                    )                    

                results.append(
                    {
                        "split": split,
                        "file": stem,
                        "crop_idx": idx,
                        **self._compute_metrics(c1, c2),
                    }
                )

        if not results:
            self.logger.warning("No valid metadata-based crop results.")
            return None

        
        # Aggregate & save results
        df = pd.DataFrame(results)
        avg = df.drop(columns=["split", "file", "crop_idx"]).mean().to_dict()

        avg_row = {**{k: "" for k in df.columns}, **avg}
        avg_row.update({"split": "AVG", "file": "AVG", "crop_idx": "AVG"})

        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
        self.metric_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

        return avg


    def run(self) -> Dict[str, Optional[Dict[str, float]]]:
        full_path = self.metric_dir / "metrics_full_image.csv"
        crop_path = self.metric_dir / "metrics_yolo_crop.csv"

        avg_full = self._evaluate_full_images(full_path)
        avg_crop = self._evaluate_crop_images(crop_path)

        self.logger.info("Evaluation complete")
        self.logger.info(f" - Global Evaluation: {avg_full}")
        self.logger.info(f" - Local Evaluation : {avg_crop}")
