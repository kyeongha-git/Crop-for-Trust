#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py
-------------------
Unified AI Pipeline Entrypoint (with Dynamic ConfigManager)

example:
# 기본 config.yaml 기반 실행
python src/main.py

# CLI override (annot_clean off)
python src/main.py --annot_clean off

# CLI override (annot_clean on + yolo_crop off)
python src/main.py --annot_clean on --yolo_crop off

# 테스트 모드 실행
python src/main.py --test
"""

import argparse
import traceback
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from utils.logging import setup_logging, get_logger
from utils.config_manager import ConfigManager
from src.annotation_cleaner.annotation_cleaner import AnnotationCleaner
from src.yolo_cropper.yolo_cropper import YOLOCropperController
from src.data_augmentor.data_augmentor import DataAugmentor
from src.classifier.classifier import Classifier


def main():
    # --------------------------------------------------------
    # 1️⃣ CLI Arguments
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(description="Full AI Pipeline Controller")

    parser.add_argument("--config", type=str, default="utils/config.yaml", help="Path to config.yaml")
    parser.add_argument("--annot_clean", type=str, choices=["on", "off"], default=None)
    parser.add_argument("--yolo_crop", type=str, choices=["on", "off"], default=None)
    parser.add_argument("--yolo_model", type=str, default=None)
    parser.add_argument("--test", type=str, choices=["on", "off"], default="off", help="AnnotationCleaner test mode (3 images only)")

    args = parser.parse_args()

    # --------------------------------------------------------
    # 2️⃣ ConfigManager: Load + Apply Overrides + Update Paths
    # --------------------------------------------------------
    cfg_mgr = ConfigManager(args.config)
    updated_cfg = cfg_mgr.update_paths(
        annot_clean=args.annot_clean,
        yolo_crop=args.yolo_crop,
        yolo_model=args.yolo_model,
        test_mode=args.test
    )
    cfg_mgr.save()  # 🔹 save updated version for reproducibility

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

    logger.info("🚀 Unified AI Pipeline Starting")
    logger.info(f"annot_clean : {annot_clean}")
    logger.info(f"yolo_crop   : {yolo_crop}")
    logger.info(f"yolo_model  : {yolo_model}")
    logger.info(f"classify_model : {classify_model}")

    # --------------------------------------------------------
    # 4️⃣ AnnotationCleaner
    # --------------------------------------------------------
    if annot_clean == "on":
        try:
            print("\n🧼 [1단계] AnnotationCleaner 시작...")
            cleaner = AnnotationCleaner(config_path=args.config)
            cleaner.run(test_mode=(args.test == "on"))
        except Exception as e:
            logger.error(f"[AnnotationCleaner] 실패: {e}")
            traceback.print_exc()
    else:
        print("⚪ [1단계] AnnotationCleaner 스킵됨")

    # --------------------------------------------------------
    # 5️⃣ YOLOCropper
    # --------------------------------------------------------
    if yolo_crop == "on":
        try:
            print(f"\n🔍 [2단계] YOLOCropper ({yolo_model}) 시작...")
            yolo_cropper = YOLOCropperController(config_path=args.config)
            yolo_cropper.run()
        except Exception as e:
            logger.error(f"[YOLOCropper] 실패: {e}")
            traceback.print_exc()
    else:
        print("⚪ [2단계] YOLOCropper 스킵됨")

    # --------------------------------------------------------
    # 6️⃣ DataAugmentor
    # --------------------------------------------------------
    # try:
    #     print("\n🧩 [3단계] DataAugmentor 시작...")
    #     augmentor = DataAugmentor(config_path=args.config)
    #     augmentor.run()
    # except Exception as e:
    #     logger.error(f"[DataAugmentor] 실패: {e}")
    #     traceback.print_exc()

    # --------------------------------------------------------
    # 7️⃣ Classifier
    # --------------------------------------------------------
    # try:
    #     print(f"\n🎯 [4단계] Classifier ({classify_model}) 시작...")
    #     classifier = Classifier(config_path=args.config)
    #     classifier.run()
    # except Exception as e:
    #     logger.error(f"[Classifier] 실패: {e}")
    #     traceback.print_exc()

    # --------------------------------------------------------
    # ✅ 완료
    # --------------------------------------------------------
    print("\n🎉 전체 파이프라인 완료!")
    logger.info("✅ All pipeline stages completed successfully.")


if __name__ == "__main__":
    main()
