#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
annotation_cleaner.py
-------------------
AnnotationCleaner 전체 파이프라인 클래스 (config 섹션별 관리)
"""

import shutil
import yaml
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]  # Research/
sys.path.append(str(ROOT_DIR))

from src.annotation_cleaner.core.image_padding import ImagePadder
from src.annotation_cleaner.core.clean_annotation import CleanAnnotation
from src.annotation_cleaner.core.restore_crop import RestoreCropper
from src.annotation_cleaner.evaluate import Evaluator
from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging


class AnnotationCleaner:
    """AnnotationCleaner 전체 프로세스 관리"""

    def __init__(self, config_path="./utils/config.yaml"):
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("AnnotationCleaner")

        # ------------------------------
        # Load Config
        # ------------------------------
        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)

        # Section slicing
        cleaner_cfg = self.cfg.get("annotation_cleaner", {})
        self.main_cfg = cleaner_cfg.get("main", {})
        self.img_padd_cfg = cleaner_cfg.get("image_padding", {})
        self.annot_clean_cfg = cleaner_cfg.get("annotation_clean", {})
        self.restore_crop_cfg = cleaner_cfg.get("restore_crop", {})
        self.evaluate_cfg = cleaner_cfg.get("evaluate", {})

        # Common values
        self.categories = self.main_cfg.get("categories", ["repair", "replace"])
        self.metadata_name = self.main_cfg.get("metadata_name", "padding_info.json")

        self.input_dir = Path(self.main_cfg.get("input_dir", "./data/original"))
        self.output_dir = Path(self.main_cfg.get("output_dir", "./data/generation"))

        self.logger.info("⚙️ [INIT] AnnotationCleaner 초기화 완료")
        self.logger.info(f"📄 설정 파일: {self.config_path}")
        self.logger.info(f"📂 입력 폴더: {self.input_dir}")
        self.logger.info(f"📦 출력 폴더: {self.output_dir}")


    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------
    def cleanup_temp_dirs(self):
        """패딩 / 생성 중간 폴더 정리"""
        main_cfg = self.restore_crop_cfg
        temp_dirs = [
            Path(self.img_padd_cfg.get("output_dir", "")),
            Path(self.annot_clean_cfg.get("output_dir", "")),
        ]

        if main_cfg.get("keep_metadata", False):
            self.logger.info("🧩 keep_metadata=True → padding 폴더 유지")
            temp_dirs.pop(0)

        for d in temp_dirs:
            if not d.exists():
                continue
            try:
                shutil.rmtree(d)
                self.logger.info(f"✅ 삭제 완료: {d}")
            except Exception as e:
                self.logger.error(f"⚠️ 삭제 실패: {d} ({e})")

    # --------------------------------------------------------
    # Replace & Export
    # --------------------------------------------------------
    def replace_and_export(self):
        """복원본을 원본 이미지와 병합"""
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

        self.logger.info(f"✅ 결과 병합 완료 → {output_dir}")

    # --------------------------------------------------------
    # Main Pipeline
    # --------------------------------------------------------
    def run(self, test_mode: bool = False):
        """전체 파이프라인 실행"""
        self.logger.info("===== 🚀 Annotation Cleaner Pipeline 시작 =====")

        # 1️⃣ Padding
        self.logger.info("[1/4] 🧱 IMAGE PADDING 단계")
        ImagePadder(
            input_dir=self.img_padd_cfg["input_dir"],
            output_dir=self.img_padd_cfg["output_dir"],
            categories=self.categories,
            target_size=self.img_padd_cfg.get("target_size", 1024),
            metadata_name=self.metadata_name,
        ).run()

        # 2️⃣ Annotation Clean
        self.logger.info("[2/4] 🎨 ANNOTATION CLEAN 단계")

        # 🔹 main.py에서 받은 test_mode가 True이면 강제로 활성화
        if test_mode:
            self.logger.info("⚙️ 테스트 모드 활성화 (이미지 3장만 처리)")
            self.annot_clean_cfg["test_mode"] = True
            self.annot_clean_cfg["test_limit"] = 3

        test_mode_flag = self.annot_clean_cfg.get("test_mode", False)
        test_limit = self.annot_clean_cfg.get("test_limit", 3) if test_mode_flag else None

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
        self.logger.info("[3/4] ✂️ RESTORE CROP 단계")
        RestoreCropper(
            input_dir=self.restore_crop_cfg["input_dir"],
            output_dir=self.restore_crop_cfg["output_dir"],
            meta_dir=self.restore_crop_cfg["metadata_root"],
            categories=self.categories,
            metadata_name=self.metadata_name,
        ).run()

        # 4️⃣ Merge & Clean
        self.logger.info("[4/4] 🔄 결과 병합 및 폴더 정리")
        self.replace_and_export()
        self.cleanup_temp_dirs()

        # 5️⃣ Evaluate
        # self.logger.info("[5/5] 📊 EVALUATION 단계")
        # Evaluator(
        #     orig_dir=self.evaluate_cfg["orig_dir"],
        #     gen_dir=self.evaluate_cfg["gen_dir"],
        #     metric_dir=self.evaluate_cfg["metric_dir"],
        #     metrics=self.evaluate_cfg.get("metrics", ["ssim", "l1", "edge_iou"]),
        #     yolo_model=self.evaluate_cfg.get("yolo_model", "./saved_model/yolo_cropper/yolov8s.pt"),
        #     imgsz=self.evaluate_cfg.get("imgsz", 416),
        #     categories=self.categories,
        # ).run()

        self.logger.info("🎉 Annotation Cleaner 전체 파이프라인 완료!")



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
