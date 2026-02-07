#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import traceback
import warnings
import shutil
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

from src.annotation_cleaner.annotation_cleaner import AnnotationCleaner
from src.classifier.classifier import Classifier
from src.data_augmentor.data_augmentor import DataAugmentor
from src.yolo_cropper.yolo_cropper import YOLOCropperController
from utils.config_manager import ConfigManager
from utils.logging import get_logger, setup_logging

def ensure_dataset_integrity(dataset_root: str, logger) -> None:
    """
    Checks if train/valid/test folders exist and are populated.
    If valid or test are empty, copies data from train to prevent Classifier crash.
    """
    root = Path(dataset_root)
    subsets = ["train", "valid", "test"]
    
    # 1. Check if dataset root exists
    if not root.exists():
        logger.warning(f"[Integrity] Dataset root not found: {root}")
        return

    # 2. Get categories from train
    train_dir = root / "train"
    if not train_dir.exists():
        return 

    categories = [d.name for d in train_dir.iterdir() if d.is_dir()]

    # 3. Fill missing data
    for cat in categories:
        src_dir = train_dir / cat
        if not src_dir.exists() or not any(src_dir.iterdir()):
            continue 

        for split in ["valid", "test"]:
            target_dir = root / split / cat
            target_dir.mkdir(parents=True, exist_ok=True)
            
            if not any(target_dir.iterdir()):
                logger.info(f"[Integrity] Healing empty dataset: Copying {cat} from train to {split}")
                for file_path in src_dir.iterdir():
                    if file_path.is_file():
                        shutil.copy(file_path, target_dir / file_path.name)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop for Trust: Reliability-Aware Visual Debiasing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, default="utils/config.yaml", help="Path to configuration file")
    
    parser.add_argument("--data_dir", type=str, default=None, help="Override input data directory for demo/sandbox mode")

    parser.add_argument("--annot_clean", action="store_true", help="Enable annotation cleaning")
    parser.add_argument("--no_annot_clean", action="store_false", dest="annot_clean", help="Disable annotation cleaning")
    parser.set_defaults(annot_clean=None)

    parser.add_argument("--annot_clean_test_mode", action="store_true", help="Enable evaluation-only mode")
    parser.add_argument("--no_annot_clean_test_mode", action="store_false", dest="annot_clean_test_mode", help="Disable evaluation-only mode")
    parser.set_defaults(annot_clean_test_mode=None)

    parser.add_argument("--yolo_crop", action="store_true", help="Enable YOLO-based cropping")
    parser.add_argument("--no_yolo_crop", action="store_false", dest="yolo_crop", help="Disable YOLO-based cropping")
    parser.set_defaults(yolo_crop=None)

    parser.add_argument("--yolo_model", type=str, default=None, choices=["yolov2", "yolov4", "yolov5", "yolov8s", "yolov8m", "yolov8l", "yolov8x"], help="Select YOLO backbone")

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
        new_data_root=args.data_dir
    )
    cfg_mgr.save()

    main_cfg = updated_cfg.get("main", {})
    annot_clean = main_cfg.get("annot_clean", True)
    yolo_crop = main_cfg.get("yolo_crop", True)
    yolo_model = main_cfg.get("yolo_model", "yolov5")
    classify_model = main_cfg.get("classify_model", "vgg16")
    
    setup_logging("logs/main")
    logger = get_logger("main")

    logger.info("Starting Crop for Trust Pipeline")
    
    # Step 1. Annotation Cleaning
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

    # Step 2. YOLO Cropping
    if yolo_crop:
        try:
            logger.info(f"[Step 2] Running YOLOCropper ({yolo_model})")
            yolo_cropper = YOLOCropperController(config_path=args.config)
            yolo_cropper.run()
        except Exception as e:
            logger.error(f"YOLOCropper failed: {e}")
            traceback.print_exc()
    else:
        logger.info("[Step 2] YOLOCropper skipped")

    # Step 3. Data Augmentation
    try:
        logger.info("[Step 3] Running Data Augmentor")
        augmentor = DataAugmentor(config_path=args.config)
        augmentor.run()
    except Exception as e:
        logger.error(f"DataAugmentor failed: {e}")
        traceback.print_exc()

    # Step 4. Classification
    try:    
        logger.info(f"[Step 4] Running Classifier ({classify_model})")
        
        classifier_input_dir = updated_cfg.get("classifier", {}).get("data", {}).get("input_dir")
        if classifier_input_dir:
            ensure_dataset_integrity(classifier_input_dir, logger)
        else:
            logger.warning("[Step 4] Classifier input_dir not found in config. Skipping integrity check.")

        classifier = Classifier(config_path=args.config)
        classifier.run()

    except Exception as e:
        logger.error(f"Classifier failed: {e}")
        traceback.print_exc()

    logger.info("Pipeline execution completed")


if __name__ == "__main__":
    main()