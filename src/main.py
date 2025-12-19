#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crop for Trust Pipeline Entrypoint.

Orchestrates the complete visual debiasing workflow:
1. Annotation Cleaning
2. YOLO-based Region Cropping
3. Data Augmentation
4. Classification
"""

import argparse
import sys
import traceback
import warnings
from pathlib import Path

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.annotation_cleaner.annotation_cleaner import AnnotationCleaner
from src.classifier.classifier import Classifier
from src.data_augmentor.data_augmentor import DataAugmentor
from src.yolo_cropper.yolo_cropper import YOLOCropperController
from utils.config_manager import ConfigManager
from utils.logging import get_logger, setup_logging


def main() -> None:
    """
    Parses CLI arguments and executes the pipeline stages sequentially.
    """
    # --------------------------------------------------------
    # CLI Configuration
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Crop for Trust: Reliability-Aware Visual Debiasing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="utils/config.yaml",
        help="Path to configuration file",
    )

    # Annotation Cleaner flags
    parser.add_argument(
        "--annot_clean",
        action="store_true",
        help="Enable annotation cleaning",
    )
    parser.add_argument(
        "--no_annot_clean",
        action="store_false",
        dest="annot_clean",
        help="Disable annotation cleaning",
    )
    parser.set_defaults(annot_clean=None)

    parser.add_argument(
        "--annot_clean_test_mode",
        action="store_true",
        help="Enable evaluation-only mode for annotation cleaner",
    )
    parser.add_argument(
        "--no_annot_clean_test_mode",
        action="store_false",
        dest="annot_clean_test_mode",
        help="Disable evaluation-only mode for annotation cleaner",
    )
    parser.set_defaults(annot_clean_test_mode=None)

    # YOLO Cropper flags
    parser.add_argument(
        "--yolo_crop",
        action="store_true",
        help="Enable YOLO-based cropping",
    )
    parser.add_argument(
        "--no_yolo_crop",
        action="store_false",
        dest="yolo_crop",
        help="Disable YOLO-based cropping",
    )
    parser.set_defaults(yolo_crop=None)

    parser.add_argument(
        "--yolo_model",
        type=str,
        default=None,
        choices=["yolov2", "yolov4", "yolov5", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="Select YOLO backbone architecture",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Configuration Management
    # --------------------------------------------------------
    cfg_mgr = ConfigManager(args.config)
    updated_cfg = cfg_mgr.update_paths(
        annot_clean=args.annot_clean,
        yolo_crop=args.yolo_crop,
        yolo_model=args.yolo_model,
        annot_clean_test_mode=args.annot_clean_test_mode,
    )
    cfg_mgr.save()

    # Extract pipeline settings
    main_cfg = updated_cfg.get("main", {})
    annot_clean = main_cfg.get("annot_clean", True)
    yolo_crop = main_cfg.get("yolo_crop", True)
    yolo_model = main_cfg.get("yolo_model", "yolov5")
    classify_model = main_cfg.get("classify_model", "vgg16")
    
    yolo_result = Path(updated_cfg["yolo_cropper"]["dataset"]["results_dir"]) / yolo_model / "result.json"

    setup_logging("logs/main")
    logger = get_logger("main")

    logger.info("Starting Crop for Trust Pipeline")
    logger.info(f"Configuration: {args.config}")

    # --------------------------------------------------------
    # Step 1. Annotation Cleaning
    # --------------------------------------------------------
    if annot_clean:
        try:
            logger.info("[Step 1] Running Annotation Cleaner")
            cleaner = AnnotationCleaner(config_path=args.config)
            cleaner.run()
        except Exception as e:
            logger.error(f"AnnotationCleaner failed: {e}")
            traceback.print_exc()
    else:
        logger.info("[Step 1] Annotation Cleaner skipped")

    # --------------------------------------------------------
    # Step 2. YOLO Cropping
    # --------------------------------------------------------
    if yolo_crop:
        try:
            logger.info(f"[Step 2] Running YOLOCropper ({yolo_model})")
            yolo_cropper = YOLOCropperController(config_path=args.config)
            yolo_cropper.run(save_image=True)
        except Exception as e:
            logger.error(f"YOLOCropper failed: {e}")
            traceback.print_exc()
    else:
        logger.info("[Step 2] YOLOCropper skipped")

    # --------------------------------------------------------
    # Step 3. Annotation Evaluation (using YOLO results)
    # --------------------------------------------------------
    if annot_clean:
        try:
            if not yolo_result.exists():
                logger.info("[Step 3] Generating YOLO metadata for evaluation (Image save skipped)")
                yolo_cropper = YOLOCropperController(config_path=args.config)
                yolo_cropper.run(save_image=False)
            else:
                logger.info("[Step 3] Using existing YOLO metadata for evaluation")

            cleaner.step_evaluate()
        except Exception as e:
            logger.error(f"Cleaner evaluation failed: {e}")
            traceback.print_exc()

    # --------------------------------------------------------
    # Step 4. Data Augmentation
    # --------------------------------------------------------
    try:
        logger.info("[Step 4] Running Data Augmentor")
        augmentor = DataAugmentor(config_path=args.config)
        augmentor.run()
    except Exception as e:
        logger.error(f"DataAugmentor failed: {e}")
        traceback.print_exc()

    # --------------------------------------------------------
    # Step 5. Classification
    # --------------------------------------------------------
    try:
        logger.info(f"[Step 5] Running Classifier ({classify_model})")
        classifier = Classifier(config_path=args.config)
        classifier.run()
    except Exception as e:
        logger.error(f"Classifier failed: {e}")
        traceback.print_exc()

    logger.info("Pipeline execution completed")


if __name__ == "__main__":
    main()