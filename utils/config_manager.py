#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
config_manager.py
-----------------
Dynamic Config Manager (with CLI overrides)

💡 핵심 특징
- main.input_dir 기준으로 전체 경로를 자동 갱신
- annotation_clean이 'on'일 때만 annotation_cleaner 경로 갱신
- yolo_crop, yolo_model에 따라 하위 모듈 입출력 경로 자동 수정
- 'off' 상태인 모듈의 경로는 절대 건드리지 않음
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Dynamic Config Manager that updates config.yaml paths safely."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.cfg = self._load_yaml()

        main_cfg = self.cfg.get("main", {})
        self.base_dir = Path(main_cfg.get("input_dir", "data/original")).resolve()
        self.test_mode = self.cfg.get("annotation_cleaner", {}).get("annotation_clean", {}).get("test_mode", "off")
        self.annot_clean = main_cfg.get("annot_clean", "on")
        self.yolo_crop = main_cfg.get("yolo_crop", "on")
        self.yolo_model = main_cfg.get("yolo_model", "yolov8s")

    # --------------------------------------------------------
    def _load_yaml(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # --------------------------------------------------------
    def update_paths(
        self,
        annot_clean: Optional[str] = None,
        yolo_crop: Optional[str] = None,
        yolo_model: Optional[str] = None,
        test_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update paths dynamically based on given overrides."""

        # CLI override 반영
        if annot_clean is not None:
            self.annot_clean = annot_clean
        if yolo_crop is not None:
            self.yolo_crop = yolo_crop
        if yolo_model is not None:
            self.yolo_model = yolo_model
        if test_mode is not None:
            self.test_mode = test_mode

        # base_dir 최신화
        self.base_dir = Path(self.cfg.get("main", {}).get("input_dir", "data/original")).resolve()

        # === Base paths ===
        base_root = self.base_dir.parent  # e.g., data/sample
        annot_root = base_root / "annotation_cleaner"

        # === AnnotationCleaner 관련 하위 경로 ===
        annot_only = annot_root / "only_annotation_image"
        annot_only_padded = annot_root / "only_annotation_image_padded"
        generated_padded = annot_root / "generated_image_padded"
        generated_final = annot_root / "generated_image"
        
         # === AnnotationCleaner Output Dir (on/off 따라 다르게 계산)
        if self.annot_clean == "on":
            annot_output_dir = base_root / "generation"
        else:
            annot_output_dir = self.base_dir  # 원본 그대로 사용

        # === YOLO Cropper Output Dir ===
        if self.yolo_crop == "on":
            crop_output_dir = annot_output_dir.parent / f"{annot_output_dir.name}_crop" / self.yolo_model
        else:
            crop_output_dir = annot_output_dir

        # ==================================================================
        # 🧼 AnnotationCleaner (on일 때만 갱신)
        # ==================================================================
        if self.annot_clean == "on":
            annotation_cfg = self.cfg.get("annotation_cleaner", {})
            annotation_cfg.setdefault("main", {})
            annotation_cfg["main"]["input_dir"] = str(self.base_dir)
            annotation_cfg["main"]["output_dir"] = str(annot_output_dir)

            annotation_cfg.setdefault("image_padding", {})
            annotation_cfg["image_padding"]["input_dir"] = str(annot_only)
            annotation_cfg["image_padding"]["output_dir"] = str(annot_only_padded)

            annotation_cfg.setdefault("annotation_clean", {})
            annotation_cfg["annotation_clean"]["input_dir"] = str(annot_only_padded)
            annotation_cfg["annotation_clean"]["output_dir"] = str(generated_padded)
            annotation_cfg["annotation_clean"]["test_mode"] = self.test_mode

            annotation_cfg.setdefault("restore_crop", {})
            annotation_cfg["restore_crop"]["input_dir"] = str(generated_padded)
            annotation_cfg["restore_crop"]["output_dir"] = str(generated_final)
            annotation_cfg["restore_crop"]["metadata_root"] = str(annot_only_padded)

            annotation_cfg.setdefault("evaluate", {})
            annotation_cfg["evaluate"]["orig_dir"] = str(annot_only)
            annotation_cfg["evaluate"]["gen_dir"] = str(generated_final)

            self.cfg["annotation_cleaner"] = annotation_cfg
        else:
            print("[⚪] AnnotationCleaner OFF → 기존 경로 유지")

        # ==================================================================
        # 🔍 YOLO Cropper
        # ==================================================================
        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        yolo_cropper_cfg.setdefault("main", {})
        yolo_cropper_cfg["main"]["input_dir"] = str(annot_output_dir)
        yolo_cropper_cfg["main"]["output_dir"] = str(crop_output_dir)
        yolo_cropper_cfg["main"]["model_name"] = self.yolo_model

        # ==================================================================
        # 🧩 DataAugmentor
        # ==================================================================
        data_augmentor_cfg = self.cfg.get("data_augmentor", {})
        data_augmentor_cfg.setdefault("data", {})
        data_augmentor_cfg["data"]["input_dir"] = str(crop_output_dir)
        data_augmentor_cfg["data"]["output_dir"] = str(crop_output_dir)

        # ==================================================================
        # 🎯 Classifier
        # ==================================================================
        classifier_cfg = self.cfg.get("classifier", {})
        classifier_cfg.setdefault("data", {})
        classifier_cfg["data"]["input_dir"] = str(crop_output_dir)

        # ==================================================================
        # 🧩 Main Config 갱신
        # ==================================================================
        self.cfg["main"]["annot_clean"] = self.annot_clean
        self.cfg["main"]["yolo_crop"] = self.yolo_crop
        self.cfg["main"]["yolo_model"] = self.yolo_model
        self.cfg["yolo_cropper"] = yolo_cropper_cfg
        self.cfg["data_augmentor"] = data_augmentor_cfg
        self.cfg["classifier"] = classifier_cfg

        return self.cfg

    # --------------------------------------------------------
    def save(self, output_path: Optional[str] = None):
        """Save updated config to YAML file."""
        target_path = Path(output_path or self.config_path)
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False, allow_unicode=True)
        print(f"[✓] Updated config saved → {target_path}")


# --------------------------------------------------------
# ✅ CLI Debug Entry
# --------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic Config Manager CLI")
    parser.add_argument("--config", type=str, default="utils/config.yaml")
    parser.add_argument("--annot_clean", type=str, choices=["on", "off"], default=None)
    parser.add_argument("--yolo_crop", type=str, choices=["on", "off"], default=None)
    parser.add_argument("--yolo_model", type=str, default=None)
    parser.add_argument("--test_mode", type=str, choices=["on", "off"], default=None)
    args = parser.parse_args()

    mgr = ConfigManager(args.config)
    updated = mgr.update_paths(
        annot_clean=args.annot_clean,
        yolo_crop=args.yolo_crop,
        yolo_model=args.yolo_model,
        test_mode=args.test_mode,
    )
    mgr.save()
