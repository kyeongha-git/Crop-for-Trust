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
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.annotation_cleaner.metrics.metrics import (
    edge_iou,
    l1_distance,
    ssim_score,
)
from utils.logging import get_logger, setup_logging
from utils.model_hub import download_fine_tuned_weights


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
        self.eval_cfg = cleaner_cfg.get("evaluate", {})
        self.main_cfg = cleaner_cfg.get("main", {})

        self.orig_dir = Path(self.eval_cfg.get("orig_dir")).resolve()
        self.gen_dir = Path(self.eval_cfg.get("gen_dir")).resolve()
        self.metric_dir = Path(self.eval_cfg.get("metric_dir", "metrics/annotation_cleaner")).resolve()

        self.metrics: List[str] = self.eval_cfg.get(
            "metrics", ["ssim", "l1", "edge_iou"]
        )
        self.categories: List[str] = self.main_cfg.get(
            "categories", ["repair", "replace"]
        )

        self.yolo_model: str = self.eval_cfg.get(
            "yolo_model", "saved_model/yolo_cropper/yolov5.pt"
        )

        self.yolo_model_name = global_main_cfg.get("yolo_model", "yolov5")
        self.imgsz: int = int(self.eval_cfg.get("imgsz", 416))
        self.conf_thres: float = float(self.eval_cfg.get("conf_thres", 0.25))

        self.saved_model_path = Path(self.yolo_model).resolve()
        test_mode = "test" in self.gen_dir.name

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        self.logger.info("Initialized Evaluator")
        self.logger.info(f" - YOLO model   : {self.yolo_model_name}")
        self.logger.info(f" - Original dir : {self.orig_dir}")
        self.logger.info(f" - Generated dir: {self.gen_dir}")
        self.logger.info(f" - Metric dir   : {self.metric_dir}")
        self.logger.info(f" - Metrics      : {self.metrics}")
        self.logger.info(f" - Test Mode    : {test_mode}")

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

    # ============================================================
    # Full Image Evaluation
    # ============================================================
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

    # ============================================================
    # YOLO Crop Evaluation
    # ============================================================
    def _evaluate_with_yolo_crop(self, save_path: Path) -> Optional[Dict[str, float]]:
        self.logger.info("[2/2] YOLO Crop Evaluation")

        download_fine_tuned_weights(
            cfg=self.cfg,
            model_name=self.yolo_model_name,
            saved_model_path=self.saved_model_path,
            logger=self.logger,
        )

        yolo = YOLO(str(self.saved_model_path))
        results = []

        with tempfile.TemporaryDirectory(prefix="eval_yolo_") as temp_root:
            temp_root = Path(temp_root)

            image_list = [
                img
                for c in self.categories
                for img in (self.gen_dir / c).glob("*.[jp][pn]g")
            ]

            for img_path in tqdm(image_list, desc="YOLO inference"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                preds = yolo.predict(
                    source=str(img_path),
                    imgsz=self.imgsz,
                    conf=self.conf_thres,
                    save=False,
                    verbose=False,
                )
                if not preds or not preds[0].boxes.xyxy.numel():
                    continue

                split = img_path.parent.name
                base = img_path.stem
                orig_path = self.orig_dir / split / f"{base}.jpg"
                if not orig_path.exists():
                    continue

                o_img = cv2.imread(str(orig_path))
                if o_img is None:
                    continue

                for idx, box in enumerate(preds[0].boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    c1 = o_img[y1:y2, x1:x2]
                    c2 = img[y1:y2, x1:x2]
                    if c1.size == 0 or c2.size == 0:
                        continue
                    if c1.shape != c2.shape:
                        c2 = cv2.resize(c2, (c1.shape[1], c1.shape[0]))

                    results.append(
                        {
                            "split": split,
                            "file": base,
                            "crop_idx": idx,
                            **self._compute_metrics(c1, c2),
                        }
                    )

        if not results:
            return None

        df = pd.DataFrame(results)
        avg = df.drop(columns=["split", "file", "crop_idx"]).mean().to_dict()

        avg_row = {**{k: "" for k in df.columns}, **avg}
        avg_row["split"] = "AVG"
        avg_row["file"] = "AVG"
        avg_row["crop_idx"] = "AVG"

        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
        self.metric_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

        return avg

    def run(self) -> Dict[str, Optional[Dict[str, float]]]:
        full_path = self.metric_dir / "metrics_full_image.csv"
        crop_path = self.metric_dir / "metrics_yolo_crop.csv"

        avg_full = self._evaluate_full_images(full_path)
        avg_crop = self._evaluate_with_yolo_crop(crop_path)

        self.logger.info("Evaluation complete")
        self.logger.info(f" - Full Image: {avg_full}")
        self.logger.info(f" - YOLO Crop : {avg_crop}")

        return {"full": avg_full, "crop": avg_crop}
