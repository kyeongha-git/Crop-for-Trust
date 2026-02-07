#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8 Inference Module.

Executes object detection using the Ultralytics API based on
file paths defined in `predict.txt`.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, List

from ultralytics import YOLO

from utils.logging import get_logger
from utils.model_hub import download_fine_tuned_weights


class YOLOv8Predictor:
    """
    Manages the inference workflow for YOLOv8, including model loading
    and result serialization.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = get_logger("yolo_cropper.YOLOv8Predictor")
        self.cfg = config

        # Configuration shortcuts
        self.global_main_cfg = self.cfg.get("main", {})
        self.demo_mode = self.global_main_cfg.get("demo", False)

        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.yolov8_cfg = self.yolo_cropper_cfg.get("yolov8", {})
        self.train_cfg = self.yolo_cropper_cfg.get("train", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        self.model_name = self.main_cfg.get("model_name", "yolov8s")

        # Paths
        saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()

        self.weights_path = (saved_model_dir / f"{self.model_name}.pt").resolve()
        self.detect_root = Path(
            self.dataset_cfg.get("detect_output_dir", "runs/detect")
        ).resolve()

        self.results_root = Path(
            self.dataset_cfg.get("results_dir", "outputs/json_results")
        ).resolve()

        self.predict_txt = (self.results_root / self.model_name / "predict.txt").resolve()

        # Inference parameters
        self.imgsz = self.train_cfg.get("imgsz", 416)
        self.save_crop = bool(self.train_cfg.get("save_crop", False))
        self.save_txt = bool(self.train_cfg.get("save_txt", True))
        self.save_conf = bool(self.train_cfg.get("save_conf", True))
        self.quiet = bool(self.yolov8_cfg.get("quiet", True))

        self.logger.info(f"Initialized Predictor (Model: {self.model_name.upper()})")

    def run(self) -> Tuple[str, str]:
        """
        Executes YOLOv8 detection.

        Returns:
            Tuple[str, str]: (last_result_dir, predict_txt_path)
        """
        if self.demo_mode:
            self.logger.info("Demo mode: Downloading fine-tuned weights")
            download_fine_tuned_weights(
                cfg=self.cfg,
                model_name=self.model_name,
                saved_model_path=self.weights_path,
                logger=self.logger,
            )

        if not self.predict_txt.exists():
            raise FileNotFoundError(f"Prediction list not found: {self.predict_txt}")

        # Read absolute image paths from predict.txt
        img_paths = [
            ln.strip()
            for ln in self.predict_txt.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if not img_paths:
            raise ValueError(f"predict.txt is empty: {self.predict_txt}")

        self.logger.info(f"Loading model: {self.weights_path}")
        model = YOLO(str(self.weights_path))

        subset_groups: Dict[str, List[str]] = {}
        for p in img_paths:
            parent = Path(p).parent.name.lower()
            subset_groups.setdefault(parent, []).append(p)

        known_subsets = {"repair", "replace"}
        use_split = set(subset_groups.keys()).issuperset(known_subsets)

        last_result_dir: Path = None  # type: ignore

        if use_split:
            self.logger.info(
                f"Detected subset folders in predict list: {sorted(subset_groups.keys())}. "
                f"Running split inference for {sorted(known_subsets)}."
            )

            for subset in ["repair", "replace"]:
                subset_sources = subset_groups.get(subset, [])
                if not subset_sources:
                    self.logger.warning(f"[SKIP] No images for subset '{subset}'")
                    continue

                run_name = f"{self.model_name}_{subset}"
                self.logger.info(
                    f"Starting Inference ({self.model_name.upper()}) on subset '{subset}' "
                    f"({len(subset_sources)} images) → runs/detect/{run_name}"
                )

                results = model.predict(
                    source=subset_sources,              
                    imgsz=self.imgsz,
                    save=True,
                    save_crop=self.save_crop,
                    save_txt=self.save_txt,
                    save_conf=self.save_conf,
                    project=str(self.detect_root),      # runs/detect
                    name=run_name,                      # yolov8s_repair / yolov8s_replace
                    exist_ok=True,
                    verbose=not self.quiet,
                )

                # results[0].save_dir points to this run's directory
                last_result_dir = Path(results[0].save_dir).resolve()
                self.logger.info(f"Subset '{subset}' results: {last_result_dir}")

        else:
            self.logger.warning(
                "Could not infer repair/replace subset folders from predict list. "
                "Falling back to a single inference run (may be ambiguous if stems collide)."
            )

            self.logger.info(f"Starting Inference ({self.model_name.upper()}) (single run)")
            results = model.predict(
                source=img_paths,
                imgsz=self.imgsz,
                save=True,
                save_crop=self.save_crop,
                save_txt=self.save_txt,
                save_conf=self.save_conf,
                project=str(self.detect_root),
                name=self.model_name,
                exist_ok=True,
                verbose=not self.quiet,
            )
            last_result_dir = Path(results[0].save_dir).resolve()
            self.logger.info(f"Inference completed. Results: {last_result_dir}")

        if last_result_dir is None:
            raise RuntimeError("Inference produced no output directory (no subsets ran).")

        return str(last_result_dir), str(self.predict_txt)