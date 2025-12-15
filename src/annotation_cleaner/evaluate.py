#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate.py
-------------------
Provides an integrated evaluation module for comparing original and generated images.
Supports both full-image evaluation and YOLO-based region (crop) evaluation.
Calculates quantitative metrics such as L1 distance, SSIM, and Edge IoU.
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

from src.annotation_cleaner.metrics.metrics import (edge_iou, l1_distance,
                                                    ssim_score)
from utils.logging import get_logger, setup_logging
from utils.model_hub import download_fine_tuned_weights


# ============================================================
# Evaluator Class
# ============================================================
class Evaluator:
    """
    Evaluates image quality between original and generated images using various metrics.

    Supports:
    - Full image comparison
    - Region-based comparison via YOLO detection

    Args:
        orig_dir (str): Directory containing original images.
        gen_dir (str): Directory containing generated images.
        metric_dir (str): Directory to save metric results.
        metrics (List[str]): List of metric names to compute (e.g., ["ssim", "l1", "edge_iou"]).
        yolo_model (str): Path to YOLO model weights.
        imgsz (int): YOLO input image size.
        categories (List[str], optional): Image category subfolders.
        conf_thres (float, optional): YOLO confidence threshold.
    """

    # ------------------------------------------------------------
    # Metric Function Mapping
    # ------------------------------------------------------------
    METRIC_MAP = {
        "l1": l1_distance,
        "ssim": ssim_score,
        "edge_iou": edge_iou,
    }

    def __init__(
        self,
        orig_dir: str,
        gen_dir: str,
        metric_dir: str,
        metrics: List[str],
        yolo_model: str,
        imgsz: int,
        categories: Optional[List[str]] = None,
        conf_thres: float = 0.25,
    ):
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("Evaluator")

        # --- Configuration ---
        self.orig_dir = Path(orig_dir)
        self.gen_dir = Path(gen_dir)
        self.metric_dir = Path(metric_dir)
        self.metrics = metrics
        self.categories = categories or ["repair", "replace"]

        # --- YOLO Settings ---
        self.yolo_model = yolo_model
        self.imgsz = imgsz
        self.conf_thres = conf_thres

        # --- Log Setup ---
        self.logger.info(f"Original images: {self.orig_dir}")
        self.logger.info(f"Generated images: {self.gen_dir}")
        self.logger.info(f"Metric output: {self.metric_dir}")
        self.logger.info(f"YOLO model: {self.yolo_model}")
        self.logger.info(f"Active metrics: {', '.join(self.metrics)}")

    # ============================================================
    # Metric Computation
    # ============================================================
    def _compute_metrics(self, orig_img, gen_img) -> Dict[str, float]:
        """
        Dynamically computes all requested metrics using functions from metrics.py.

        Returns:
            Dict[str, float]: Computed metric results for a single image pair.
        """
        results = {}
        for metric_name in self.metrics:
            func = self.METRIC_MAP.get(metric_name)
            if not func:
                self.logger.warning(f"Unsupported metric: {metric_name} — skipped")
                continue

            try:
                val = func(orig_img, gen_img)
                key = metric_name.upper() if metric_name != "edge_iou" else "Edge_IoU"
                results[key] = float(val)
            except Exception as e:
                self.logger.error(f"Metric computation failed ({metric_name}): {e}")
        return results

    # ============================================================
    # Full Image Evaluation
    # ============================================================
    def evaluate_full_images(self, save_path: Path) -> Optional[Dict[str, float]]:
        """
        Compares original and generated images on a full-image basis.

        Iterates through each category, matches images by filename,
        computes metrics, and aggregates average scores across all samples.
        """
        self.logger.info("[1/2] Starting Full Image Evaluation...")
        results = []

        for split in self.categories:
            orig_split = self.orig_dir / split
            gen_split = self.gen_dir / split

            if not orig_split.exists() or not gen_split.exists():
                self.logger.warning(f"Missing folder: {split} — skipped")
                continue

            o_files = {f.stem: f for f in orig_split.glob("*.[jp][pn]g")}
            g_files = {f.stem: f for f in gen_split.glob("*.[jp][pn]g")}
            common = set(o_files.keys()) & set(g_files.keys())

            for name in tqdm(common, desc=f"{split}"):
                o_img = cv2.imread(str(o_files[name]))
                g_img = cv2.imread(str(g_files[name]))
                if o_img is None or g_img is None:
                    continue
                if o_img.shape != g_img.shape:
                    g_img = cv2.resize(g_img, (o_img.shape[1], o_img.shape[0]))

                metric_vals = self._compute_metrics(o_img, g_img)
                results.append({"split": split, "file": name, **metric_vals})

        if not results:
            self.logger.warning("No images found for evaluation.")
            return None

        df = pd.DataFrame(results)
        avg = df.drop(columns=["split", "file"]).mean().to_dict()

        avg_row = {**{k: "" for k in df.columns}, **avg}
        avg_row["split"] = "AVG"
        avg_row["file"] = "AVG"
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        self.metric_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        self.logger.info(f"Full Image results saved → {save_path}")
        return avg

    # ============================================================
    # YOLO-Based Crop Evaluation
    # ============================================================
    def evaluate_with_yolo_crop(self, save_path: Path) -> Optional[Dict[str, float]]:
        """
        Performs region-based evaluation using YOLO-detected bounding boxes.

        Each generated image is passed through YOLO to extract regions of interest (ROIs).
        Corresponding regions from original images are compared using the same metrics.
        """
        self.logger.info("[2/2] Starting YOLO Crop Evaluation...")
        yolo = YOLO(self.yolo_model)
        results = []

        # Temporary working directory for intermediate crops
        with tempfile.TemporaryDirectory(prefix="eval_yolo_") as temp_root:
            temp_root = Path(temp_root)
            crop_dir = temp_root / "crops"
            bbox_dir = temp_root / "bboxes"
            crop_dir.mkdir(parents=True, exist_ok=True)
            bbox_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Temporary folder created: {temp_root}")

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

                split_name = img_path.parent.name
                base_name = img_path.stem
                bbox_txt = bbox_dir / split_name / f"{base_name}.txt"
                crop_split_dir = crop_dir / split_name
                bbox_txt.parent.mkdir(parents=True, exist_ok=True)
                crop_split_dir.mkdir(parents=True, exist_ok=True)

                # Record bounding box coordinates for traceability
                with open(bbox_txt, "w") as f:
                    for idx, box in enumerate(preds[0].boxes.xyxy):
                        x1, y1, x2, y2 = map(int, box)
                        f.write(f"abs {x1} {y1} {x2} {y2}\n")

                        crop = img[y1:y2, x1:x2]
                        if crop.size > 0:
                            cv2.imwrite(
                                str(crop_split_dir / f"{base_name}_crop{idx}.jpg"), crop
                            )

                orig_path = self.orig_dir / split_name / f"{base_name}.jpg"
                if not orig_path.exists():
                    continue

                o_img = cv2.imread(str(orig_path))
                if o_img is None:
                    continue

                # Evaluate metrics for each detected region
                for idx, box in enumerate(preds[0].boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    c1, c2 = o_img[y1:y2, x1:x2], img[y1:y2, x1:x2]
                    if c1.size == 0 or c2.size == 0:
                        continue
                    if c1.shape != c2.shape:
                        c2 = cv2.resize(c2, (c1.shape[1], c1.shape[0]))

                    metric_vals = self._compute_metrics(c1, c2)
                    results.append(
                        {
                            "split": split_name,
                            "file": base_name,
                            "crop_idx": idx,
                            **metric_vals,
                        }
                    )

            self.logger.info(f"Temporary YOLO crop data cleaned up: {temp_root}")

        self.metric_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate results
        if not results:
            self.logger.warning("No YOLO crop evaluation results.")
            return None

        df = pd.DataFrame(results)
        avg = df.drop(columns=["split", "file", "crop_idx"]).mean().to_dict()

        avg_row = {**{k: "" for k in df.columns}, **avg}
        avg_row["split"] = "AVG"
        avg_row["file"] = "AVG"
        avg_row["crop_idx"] = "AVG"
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        df.to_csv(save_path, index=False)
        self.logger.info(f"YOLO Crop results saved → {save_path}")
        return avg

    def run(self) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Runs both full-image and YOLO crop-based evaluations, saving all results.

        Returns:
            Dict[str, Optional[Dict[str, float]]]: Average metric results for each evaluation mode.
        """


        full_path = self.metric_dir / "metrics_full_image.csv"
        crop_path = self.metric_dir / "metrics_yolo_crop.csv"

        download_fine_tuned_weights(
            cfg=self.cfg,
            model_name=self.yolo_model,
            saved_model_path=self.saved_model_path,
            logger=self.logger,
        )

        avg_full = self.evaluate_full_images(full_path)
        avg_crop = self.evaluate_with_yolo_crop(crop_path)

        self.logger.info("\n=== Final Average Results ===")
        self.logger.info(f"Full Image: {avg_full}")
        self.logger.info(f"YOLO Crop:  {avg_crop}")

        return {"full": avg_full, "crop": avg_crop}
