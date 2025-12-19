#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Orchestrates the annotation cleaning pipeline.

This module manages the sequential execution of image padding, generative
annotation removal, restoration, and result merging.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

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
    Controller for the Annotation Cleaner pipeline.

    Orchestrates the following stages:
    1. Image Padding: Pads images to square dimensions.
    2. Annotation Cleaning: Removes visual annotations using a generative model.
    3. Restoration: Restores images to their original dimensions.
    4. Merging: Combines processed images with the original dataset.
    """

    def __init__(self, config_path: str = "./utils/config.yaml") -> None:
        """
        Initializes the pipeline controller.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("annotation_cleaner.Pipeline")

        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)

        # Configuration setup
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
        self.logger.info(f"Config file: {self.config_path}")

    # --------------------------------------------------------
    # Pipeline Steps
    # --------------------------------------------------------

    def cleanup_temp_dirs(self) -> None:
        """
        Removes intermediate directories to save disk space.
        Skips cleanup if 'keep_metadata' is enabled in configuration.
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
            
            # Preserve metadata if configured
            if keep_metadata and path == Path(self.img_padd_cfg.get("output_dir")):
                self.logger.info("Preserving padding metadata directory")
                continue
            
            try:
                shutil.rmtree(path)
                self.logger.info(f"Removed temp directory: {path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove {path}: {e}")

    def step_evaluate(self) -> None:
        """
        Runs the evaluation module to compare generated results against ground truth.
        """
        Evaluator(config=self.cfg).run()

    def step_padding(self) -> None:
        """
        Step 1: Pads images to a fixed square size.
        """
        self.logger.info("[STEP 1] Starting Image Padding...")
        ImagePadder(config=self.cfg).run()

    def step_annotation_clean(self) -> None:
        """
        Step 2: Removes annotations using the generative model.
        """
        self.logger.info("[STEP 2] Starting Annotation Cleaning...")
        CleanAnnotation(config=self.cfg).run()

    def step_restore_crop(self) -> None:
        """
        Step 3: Restores images to original dimensions using padding metadata.
        """
        self.logger.info("[STEP 3] Starting Restore Cropped Images...")
        RestoreCropper(config=self.cfg).run()

    def step_merge_and_cleanup(self) -> None:
        """
        Step 4: Merges final outputs into the destination directory and cleans up temp files.
        """
        self.logger.info("[STEP 4] Starting Merge Results & Cleanup...")

        input_dir = Path(self.main_cfg.get("input_dir"))
        output_dir = Path(self.main_cfg.get("output_dir"))
        restored_dir = Path(self.restore_crop_cfg.get("output_dir"))

        output_dir.mkdir(parents=True, exist_ok=True)

        for category in self.categories:
            out_cat = output_dir / category
            out_cat.mkdir(parents=True, exist_ok=True)

            # Copy original input files first
            for src in (input_dir / category).glob("*"):
                shutil.copy2(src, out_cat / src.name)

            # Overwrite with restored (cleaned) files
            for gen in (restored_dir / category).glob("*"):
                shutil.copy2(gen, out_cat / gen.name)

        self.cleanup_temp_dirs()

    def run(self, test_mode: bool = False) -> None:
        """
        Executes the full annotation cleaning pipeline.

        Args:
            test_mode (bool): If True, enables limited processing for debugging/testing.
        """
        self.logger.info("===== Starting Annotation Cleaner Pipeline =====")

        self.step_padding()
        self.step_annotation_clean()
        self.step_restore_crop()
        self.step_merge_and_cleanup()

        self.logger.info("Annotation Cleaner pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotation Cleaner Pipeline")
    parser.add_argument("--config", default="./utils/config.yaml")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    cleaner = AnnotationCleaner(config_path=args.config)
    cleaner.run(test_mode=args.test)