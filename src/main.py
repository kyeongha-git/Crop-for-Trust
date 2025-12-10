#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py
--------
Unified AI Pipeline Entrypoint (Config-driven)

This script serves as the top-level entrypoint for the entire AI pipeline,
integrating multiple components such as:

1️⃣ AnnotationCleaner – removes annotation artifacts
2️⃣ YOLOCropper – runs YOLO-based object detection and cropping
3️⃣ DataAugmentor – performs dataset augmentation
4️⃣ Classifier – classifies cropped or restored images

Users can selectively enable or disable each module via command-line
arguments or by editing the configuration file.

Example Usage:
    # Run full pipeline using default config.yaml
    $ python src/main.py

    # Run with AnnotationCleaner disabled
    $ python src/main.py --annot_clean off

    # Run with AnnotationCleaner on, YOLOCropper off
    $ python src/main.py --annot_clean on --yolo_crop off

    # Run in test mode (process 3 sample images)
    $ python src/main.py --test on
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.annotation_cleaner.annotation_cleaner import AnnotationCleaner
from src.yolo_cropper.yolo_cropper import YOLOCropperController
from src.data_augmentor.data_augmentor import DataAugmentor
from src.classifier.classifier import Classifier            
from utils.config_manager import ConfigManager
from utils.logging import get_logger, setup_logging


def main():
    """
    Main entrypoint for the unified AI pipeline.

    Execution Flow:
        1. Parse CLI arguments
        2. Load and update configuration
        3. Initialize logging
        4. Run each enabled pipeline stage

    CLI Overrides:
        --annot_clean [on|off] : Enable/disable annotation cleaning
        --yolo_crop [on|off]   : Enable/disable YOLO cropper
        --yolo_model <str>     : Override YOLO model type (e.g., yolov8s)
        --test [on|off]        : Enable test mode (3 sample images)

    """
    # --------------------------------------------------------
    # 1️⃣ CLI Arguments
    # --------------------------------------------------------
    env_config = os.environ.get("CONFIG", None)

    parser = argparse.ArgumentParser(description="Full AI Pipeline Controller")

    parser.add_argument(
        "--config",
        type=str,
        default=env_config if env_config else "utils/config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument("--annot_clean", type=str, choices=["on", "off"], default=None)
    parser.add_argument("--yolo_crop", type=str, choices=["on", "off"], default=None)
    parser.add_argument("--yolo_model", type=str, default=None)
    parser.add_argument(
        "--test",
        type=str,
        choices=["on", "off"],
        default="off",
        help="AnnotationCleaner test mode (3 images only)",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # 2️⃣ ConfigManager: Load + Apply Overrides + Update Paths
    # --------------------------------------------------------
    cfg_mgr = ConfigManager(args.config)
    updated_cfg = cfg_mgr.update_paths(
        annot_clean=args.annot_clean,
        yolo_crop=args.yolo_crop,
        yolo_model=args.yolo_model,
        test_mode=args.test,
    )
    cfg_mgr.save()  # Persist updated version for reproducibility

    main_cfg = updated_cfg.get("main", {})
    annot_clean = main_cfg.get("annot_clean", "on")
    yolo_crop = main_cfg.get("yolo_crop", "on")
    yolo_model = main_cfg.get("yolo_model", "yolov8s")
    classify_model = main_cfg.get("classify_model", "vgg16")

    # --------------------------------------------------------
    # 3️⃣ Logging Setup
    # --------------------------------------------------------
    setup_logging("logs/main")
    logger = get_logger("main")

    logger.info("Unified AI Pipeline Starting")
    logger.info(f"annot_clean    : {annot_clean}")
    logger.info(f"yolo_crop      : {yolo_crop}")
    logger.info(f"yolo_model     : {yolo_model}")
    logger.info(f"classify_model : {classify_model}")

    # --------------------------------------------------------
    # 4️⃣ AnnotationCleaner
    # --------------------------------------------------------
    if annot_clean == "on":
        try:
            print("\n[1] Running AnnotationCleaner...")
            cleaner = AnnotationCleaner(config_path=args.config)
            cleaner.run(test_mode=(args.test == "on"))
        except Exception as e:
            logger.error(f"[AnnotationCleaner] Failed: {e}")
            traceback.print_exc()
    else:
        print("[1] AnnotationCleaner skipped")

    # --------------------------------------------------------
    # 5️⃣ YOLOCropper
    # --------------------------------------------------------
    if yolo_crop == "on":
        try:
            print(f"\n[2] Running YOLOCropper ({yolo_model})...")
            yolo_cropper = YOLOCropperController(config_path=args.config)
            yolo_cropper.run()
        except Exception as e:
            logger.error(f"[YOLOCropper] Failed: {e}")
            traceback.print_exc()
    else:
        print("[2] YOLOCropper skipped")

    # --------------------------------------------------------
    # 6️⃣ DataAugmentor (Optional)
    # --------------------------------------------------------
    try:
        print("\n[3] Running DataAugmentor...")
        augmentor = DataAugmentor(config_path=args.config)
        augmentor.run()
    except Exception as e:
        logger.error(f"[DataAugmentor] Failed: {e}")
        traceback.print_exc()

    # --------------------------------------------------------
    # 7️⃣ Classifier (Optional)
    # --------------------------------------------------------
    try:
        print(f"\n[4] Running Classifier ({classify_model})...")
        classifier = Classifier(config_path=args.config)
        classifier.run()
    except Exception as e:
        logger.error(f"[Classifier] Failed: {e}")
        traceback.print_exc()

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------
    print("\n All pipeline stages completed!")
    logger.info("All pipeline stages completed successfully.")


if __name__ == "__main__":
    main()
