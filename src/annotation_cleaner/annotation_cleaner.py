#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
annotation_cleaner.py
-------------------
Manages the full AnnotationCleaner pipeline, including image padding, annotation
removal, restoration, and result merging. Each step is modularized and configured
via sections in the YAML configuration file.
"""

import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.annotation_cleaner.core.clean_annotation import \
    CleanAnnotation
from src.annotation_cleaner.core.image_padding import ImagePadder
from src.annotation_cleaner.core.restore_crop import \
    RestoreCropper
from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging


class AnnotationCleaner:
    """
    Controls and executes the complete annotation cleaning pipeline.

    The pipeline performs the following stages:
    1. Pads original images to a fixed size.
    2. Removes hand-drawn annotations using the Gemini API.
    3. Restores images back to their original dimensions.
    4. Merges restored results with the original dataset.
    5. (Optional) Evaluates image quality metrics.

    Configuration for each stage is managed via YAML file sections.
    """

    def __init__(self, config_path="./utils/config.yaml"):
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("AnnotationCleaner")

        # ------------------------------
        # Load Configuration
        # ------------------------------
        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)

        # Separate sections by task
        cleaner_cfg = self.cfg.get("annotation_cleaner", {})
        self.main_cfg = cleaner_cfg.get("main", {})
        self.img_padd_cfg = cleaner_cfg.get("image_padding", {})
        self.annot_clean_cfg = cleaner_cfg.get("annotation_clean", {})
        self.restore_crop_cfg = cleaner_cfg.get("restore_crop", {})
        self.evaluate_cfg = cleaner_cfg.get("evaluate", {})

        # Common attributes
        self.categories = self.main_cfg.get("categories", ["repair", "replace"])
        self.metadata_name = self.main_cfg.get("metadata_name", "padding_info.json")

        self.input_dir = Path(self.main_cfg.get("input_dir", "./data/original"))
        self.output_dir = Path(self.main_cfg.get("output_dir", "./data/generation"))

        self.logger.info("[INIT] AnnotationCleaner initialized.")
        self.logger.info(f"Config file: {self.config_path}")
        self.logger.info(f"Input folder: {self.input_dir}")
        self.logger.info(f"Output folder: {self.output_dir}")

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------
    def cleanup_temp_dirs(self):
        """
        Removes temporary directories created during padding and cleaning.

        If 'keep_metadata' is enabled in the config, the padding directory
        is preserved for future reference.
        """
        main_cfg = self.restore_crop_cfg
        temp_dirs = [
            Path(self.img_padd_cfg.get("output_dir", "")),
            Path(self.annot_clean_cfg.get("output_dir", "")),
        ]

        if main_cfg.get("keep_metadata", False):
            self.logger.info("keep_metadata=True → preserving padding folder")
            temp_dirs.pop(0)

        for d in temp_dirs:
            if not d.exists():
                continue
            try:
                shutil.rmtree(d)
                self.logger.info(f"Deleted: {d}")
            except Exception as e:
                self.logger.error(f"Failed to delete: {d} ({e})")

    # --------------------------------------------------------
    # Replace & Export
    # --------------------------------------------------------
    def replace_and_export(self):
        """
        Merges the restored images with the original dataset.

        Original images are copied first, and restored versions overwrite
        any matching files. This ensures final outputs are clean and updated.
        """
        input_dir = Path(self.main_cfg["input_dir"])
        restored_dir = Path(self.restore_crop_cfg["output_dir"])
        output_dir = Path(self.main_cfg["output_dir"])

        output_dir.mkdir(parents=True, exist_ok=True)
        valid_exts = (".jpg", ".jpeg", ".png")

        for category in self.categories:
            orig_cat = input_dir / category
            restored_cat = restored_dir / category
            out_cat = output_dir / category
            out_cat.mkdir(parents=True, exist_ok=True)

            for file in orig_cat.glob("*"):
                if file.suffix.lower() in valid_exts:
                    shutil.copy2(file, out_cat / file.name)

            if restored_cat.exists():
                for rest_file in restored_cat.glob("*"):
                    dst = out_cat / rest_file.name
                    if dst.exists():
                        shutil.copy2(rest_file, dst)

        self.logger.info(f"Merging complete → {output_dir}")

    # --------------------------------------------------------
    # Main Pipeline
    # --------------------------------------------------------
    def run(self, test_mode: bool = False):
        """
        Executes the entire annotation cleaning pipeline step by step.

        Args:
            test_mode (bool): If True, limits cleaning to a small sample for testing.
        """
        self.logger.info("===== Starting Annotation Cleaner Pipeline =====")

        # 1️⃣ Image Padding
        self.logger.info("[1/4] IMAGE PADDING")
        ImagePadder(
            input_dir=self.img_padd_cfg["input_dir"],
            output_dir=self.img_padd_cfg["output_dir"],
            categories=self.categories,
            target_size=self.img_padd_cfg.get("target_size", 1024),
            metadata_name=self.metadata_name,
        ).run()

        # 2️⃣ Annotation Cleaning
        self.logger.info("[2/4] ANNOTATION CLEANING")

        # Enable test mode if requested
        if test_mode:
            self.logger.info("⚙️ Test mode enabled (processing 3 images only).")
            self.annot_clean_cfg["test_mode"] = True
            self.annot_clean_cfg["test_limit"] = 3

        test_mode_flag = self.annot_clean_cfg.get("test_mode", False)
        test_limit = (
            self.annot_clean_cfg.get("test_limit", 3) if test_mode_flag else None
        )

        CleanAnnotation(
            input_dir=self.annot_clean_cfg["input_dir"],
            output_dir=self.annot_clean_cfg["output_dir"],
            model=self.annot_clean_cfg["model"],
            prompt=self.annot_clean_cfg["prompt"],
            categories=self.categories,
            test_mode=test_mode_flag,
            test_limit=test_limit,
        ).run()

        # 3️⃣ Restore Crop
        self.logger.info("[3/4] RESTORE CROP")
        RestoreCropper(
            input_dir=self.restore_crop_cfg["input_dir"],
            output_dir=self.restore_crop_cfg["output_dir"],
            meta_dir=self.restore_crop_cfg["metadata_root"],
            categories=self.categories,
            metadata_name=self.metadata_name,
        ).run()

        # 4️⃣ Merge & Cleanup
        self.logger.info("[4/4] MERGE RESULTS AND CLEANUP")
        self.replace_and_export()
        self.cleanup_temp_dirs()

        # 5️⃣ Evaluation
        # self.logger.info("[5/5] EVALUATION")
        # Evaluator(
        #     orig_dir=self.evaluate_cfg["orig_dir"],
        #     gen_dir=self.evaluate_cfg["gen_dir"],
        #     metric_dir=self.evaluate_cfg["metric_dir"],
        #     metrics=self.evaluate_cfg.get("metrics", ["ssim", "l1", "edge_iou"]),
        #     yolo_model=self.evaluate_cfg.get("yolo_model", "./saved_model/yolo_cropper/yolov8s.pt"),
        #     imgsz=self.evaluate_cfg.get("imgsz", 416),
        #     categories=self.categories,
        # ).run()

        self.logger.info("🎉 Annotation Cleaner pipeline completed successfully!")


# ------------------------------------------------------------
# CLI Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotation Cleaner Pipeline")
    parser.add_argument("--config", default="./utils/config.yaml")
    args = parser.parse_args()

    cleaner = AnnotationCleaner(config_path=args.config)
    cleaner.run()
