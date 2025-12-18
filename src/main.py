#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import traceback
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.annotation_cleaner.annotation_cleaner import AnnotationCleaner
from src.yolo_cropper.yolo_cropper import YOLOCropperController
from src.data_augmentor.data_augmentor import DataAugmentor
from src.classifier.classifier import Classifier
from utils.config_manager import ConfigManager
from utils.logging import get_logger, setup_logging


def main():

    # --------------------------------------------------------
    # CLI Arguments
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(description="Full AI Pipeline Controller")

    parser.add_argument(
        "--config",
        type=str,
        default="utils/config.yaml",
        help="Path to configuration YAML file",
    )

    parser.add_argument("--annot_clean", action="store_true")
    parser.add_argument("--no_annot_clean", action="store_false", dest="annot_clean")
    parser.set_defaults(annot_clean=None)

    parser.add_argument("--annot_clean_test_mode", action="store_true")
    parser.add_argument("--no_annot_clean_test_mode", action="store_false", dest="annot_clean_test_mode")
    parser.set_defaults(annot_clean_test_mode=None)

    parser.add_argument("--yolo_crop", action="store_true")
    parser.add_argument("--no_yolo_crop", action="store_false", dest="yolo_crop")
    parser.set_defaults(yolo_crop=None)

    parser.add_argument("--yolo_model", type=str, default=None)

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load & Update Config
    # --------------------------------------------------------
    cfg_mgr = ConfigManager(args.config)
    updated_cfg = cfg_mgr.update_paths(
        annot_clean=args.annot_clean,
        yolo_crop=args.yolo_crop,
        yolo_model=args.yolo_model,
        annot_clean_test_mode=args.annot_clean_test_mode,
    )
    cfg_mgr.save()

    main_cfg = updated_cfg.get("main", {})
    annot_clean_test_mode = main_cfg.get("annot_clean_test_mode", True)
    annot_clean = main_cfg.get("annot_clean", True)
    yolo_crop = main_cfg.get("yolo_crop", True)
    yolo_model = main_cfg.get("yolo_model", "yolov8s")
    classify_model = main_cfg.get("classify_model", "vgg16")
    demo_mode = main_cfg.get("demo", False)
    yolo_result = Path(updated_cfg["yolo_cropper"]["dataset"]["results_dir"]) / yolo_model / "result.json"
    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    setup_logging("logs/main")
    logger = get_logger("main")

    logger.info("Crop for Trust Pipeline Starting...")
    logger.info(f"demo mode      : {demo_mode}")
    logger.info(f"annot_clean_test_mode : {annot_clean_test_mode}")
    logger.info(f"annot_clean    : {annot_clean}")
    logger.info(f"yolo_crop      : {yolo_crop}")
    logger.info(f"yolo_model     : {yolo_model}")
    logger.info(f"classify_model : {classify_model}")

    # --------------------------------------------------------
    # Step 1. AnnotationCleaner
    # --------------------------------------------------------
    if annot_clean:
        try:
            logger.info("[STEP 1] Starting Annotation Cleaner...")
            cleaner = AnnotationCleaner(config_path=args.config)
            cleaner.run()
        except Exception as e:
            logger.error(f"[AnnotationCleaner] Failed: {e}")
            traceback.print_exc()
    else:
        logger.info("[STEP 1] Annotation Cleaner skipped.")

    # --------------------------------------------------------
    # Step 2. YOLOCropper
    # --------------------------------------------------------
    if yolo_crop:
        try:
            logger.info(f"[STEP 2] Starting YOLOCropper ({yolo_model})...")
            yolo_cropper = YOLOCropperController(config_path=args.config)
            yolo_cropper.run(save_image = True)
        except Exception as e:
            logger.error(f"[YOLOCropper] Failed: {e}")
            traceback.print_exc()
    else:
        print("[STEP 2] YOLOCropper skipped")

    # --------------------------------------------------------
    # Step 3. Annotation Cleaner Evaluation (Crop Evaluation)
    # --------------------------------------------------------
    if annot_clean:
        try:
            if yolo_result.exists():
                logger.info("[STEP 3] Using existing YOLO metadata for evaluation.")
            else:
                logger.info("[STEP 3] YOLO metadata missing. Running YOLO (save_image=False) for evaluation only.")
                yolo_cropper = YOLOCropperController(config_path=args.config)
                yolo_cropper.run(save_image = False)

            cleaner.step_evaluate()
        except Exception as e:
            logger.error(f"[AnnotationCleaner] Evaluation Failed: {e}")
            traceback.print_exc()

    # --------------------------------------------------------
    # Step 4. DataAugmentor
    # --------------------------------------------------------
    try:
        logger.info("[STEP 4] Starting DataAugmentor...")
        augmentor = DataAugmentor(config_path=args.config)
        augmentor.run()
    except Exception as e:
        logger.error(f"[DataAugmentor] Failed: {e}")
        traceback.print_exc()

    # --------------------------------------------------------
    # Step 5. Classifier
    # --------------------------------------------------------
    try:
        logger.info(f"[STEP 5] Starting Classifier ({classify_model})...")
        classifier = Classifier(config_path=args.config)
        classifier.run()
    except Exception as e:
        logger.error(f"[Classifier] Failed: {e}")
        traceback.print_exc()


    logger.info("All pipeline stages completed successfully.")


if __name__ == "__main__":
    main()
