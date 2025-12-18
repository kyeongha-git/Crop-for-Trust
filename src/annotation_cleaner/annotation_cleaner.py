#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
annotation_cleaner.py
-------------------
Configuration-driven Annotation Cleaner pipeline.

Pipeline Steps:
1. Image Padding
2. Annotation Cleaning (Gemini-based)
3. Restore Cropped Images
4. Merge Results & Cleanup
5. Evaluation
"""

import shutil
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.annotation_cleaner.core.image_padding import ImagePadder
from src.annotation_cleaner.core.clean_annotation import CleanAnnotation
from src.annotation_cleaner.core.restore_crop import RestoreCropper
from src.annotation_cleaner.evaluate import Evaluator

from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging


class AnnotationCleaner:
    """
    Full Annotation Cleaner pipeline controller.

    Each stage is:
    - Config-driven
    - Executed via `.run()`
    - Isolated for readability and reproducibility
    """

    def __init__(self, config_path: str = "./utils/config.yaml"):
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("annotation_cleaner.Pipeline")

        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)

        global_main_cfg = self.cfg.get("main", {})
        cleaner_cfg = self.cfg.get("annotation_cleaner", {})
        self.main_cfg = cleaner_cfg.get("main", {})
        self.img_padd_cfg = cleaner_cfg.get("image_padding", {})
        self.annot_clean_cfg = cleaner_cfg.get("annotation_clean", {})
        self.restore_crop_cfg = cleaner_cfg.get("restore_crop", {})

        self.categories: List[str] = global_main_cfg.get(
            "categories", ["repair", "replace"]
        )

        self.logger.info("Initialized Annotation Cleaner Pipeline")
        self.logger.info(f"Config file : {self.config_path}")

    # --------------------------------------------------------
    # Cleanup Temporary Directories
    # --------------------------------------------------------
    def cleanup_temp_dirs(self):
        """
        Remove intermediate folders unless keep_metadata is enabled.
        """
        keep_metadata = self.restore_crop_cfg.get("keep_metadata", False)

        candidates = [
            self.img_padd_cfg.get("output_dir"),
            self.annot_clean_cfg.get("output_dir"),
        ]

        for path_str in candidates:
            if not path_str:
                continue
            path = Path(path_str)
            if not path.exists():
                continue
            if keep_metadata and path == Path(self.img_padd_cfg.get("output_dir")):
                self.logger.info("Preserving padding metadata directory")
                continue
            try:
                shutil.rmtree(path)
                self.logger.info(f"Removed temp directory: {path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove {path}: {e}")

    # --------------------------------------------------------
    # Evaluation (Global & Crop Evaluation)
    # --------------------------------------------------------
    def step_evaluate(self):
        Evaluator(config=self.cfg).run()

    # --------------------------------------------------------
    # Step 1. Image Padding
    # --------------------------------------------------------
    def step_padding(self):
        self.logger.info("[STEP 1] Starting Image Padding...")
        ImagePadder(config=self.cfg).run()

    # --------------------------------------------------------
    # Step 2. Annotation Cleaning
    # --------------------------------------------------------
    def step_annotation_clean(self):
        self.logger.info("[STEP 2] Starting Annotation Cleaning...")
        CleanAnnotation(config=self.cfg).run()

    # --------------------------------------------------------
    # Step 3. Restore Cropped Images
    # --------------------------------------------------------
    def step_restore_crop(self):
        self.logger.info("[STEP 3] Starting Restore Cropped Images...")
        RestoreCropper(config=self.cfg).run()

    # --------------------------------------------------------
    # Step 4. Merge & Cleanup
    # --------------------------------------------------------
    def step_merge_and_cleanup(self):
        self.logger.info("[STEP 4] Starting Merge Results & Cleanup...")

        input_dir = Path(self.main_cfg.get("input_dir"))
        output_dir = Path(self.main_cfg.get("output_dir"))
        restored_dir = Path(self.restore_crop_cfg.get("output_dir"))

        output_dir.mkdir(parents=True, exist_ok=True)

        for category in self.categories:
            out_cat = output_dir / category
            out_cat.mkdir(parents=True, exist_ok=True)

            for src in (input_dir / category).glob("*"):
                shutil.copy2(src, out_cat / src.name)

            for gen in (restored_dir / category).glob("*"):
                shutil.copy2(gen, out_cat / gen.name)

        self.cleanup_temp_dirs()

    # --------------------------------------------------------
    # Entrypoint
    # --------------------------------------------------------
    def run(self):
        """
        Run full annotation cleaner pipeline.

        Args:
            test_mode (bool): Enables limited processing for debugging.
        """
        self.logger.info("===== Starting Annotation Cleaner Pipeline =====")

        self.step_padding()
        self.step_annotation_clean()
        self.step_restore_crop()
        self.step_merge_and_cleanup()

        self.logger.info("Annotation Cleaner pipeline completed successfully!")


# ------------------------------------------------------------
# CLI Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotation Cleaner Pipeline")
    parser.add_argument("--config", default="./utils/config.yaml")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    cleaner = AnnotationCleaner(config_path=args.config)
    cleaner.run(test_mode=args.test)
