#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
config_manager.py
-----------------
Dynamic Config Manager (with CLI Overrides)

Responsibilities:
- Single source of truth for YOLO model selection (main.yolo_model)
- Automatic synchronization across yolo_cropper / darknet / yolov5 / yolov8
- Path propagation for annotation cleaner, YOLO cropper, data augmentor, classifier
- Demo / full mode aware behavior
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigManager:
    """
    Dynamic Config Manager that safely updates YAML configurations.
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.cfg = self._load_yaml()

        main_cfg = self.cfg.get("main", {})

        # Base input directory (e.g., data/sample/original)
        self.base_dir = Path(main_cfg.get("input_dir", "data/original")).resolve()

        # Mode flags
        self.demo_mode = main_cfg.get("demo", False)
        self.annot_clean = main_cfg.get("annot_clean", "on")
        self.yolo_crop = main_cfg.get("yolo_crop", "on")

        # Single source of truth
        self.yolo_model = main_cfg.get("yolo_model", "yolov8s").lower()

        self.test_mode = (
            self.cfg.get("annotation_cleaner", {})
            .get("annotation_clean", {})
            .get("test_mode", "off")
        )

    # --------------------------------------------------------
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # --------------------------------------------------------
    def _build_classifier_save_dir(self, classifier_input: Path) -> Path:
        """
        Classifier input_dir → hierarchical save_dir path.

        Example:
        input_dir = data/sample/original_crop/yolov2/dataset
        → saved_model/classifier/sample/original_crop/yolov2
        """
        if classifier_input.name == "dataset":
            classifier_input = classifier_input.parent

        parts = classifier_input.parts
        try:
            data_index = parts.index("data")
            subpath = Path(*parts[data_index + 1 :])
        except ValueError:
            subpath = Path(*parts[-3:])

        return Path("saved_model/classifier") / subpath

    # --------------------------------------------------------
    def update_paths(
        self,
        annot_clean: Optional[str] = None,
        yolo_crop: Optional[str] = None,
        yolo_model: Optional[str] = None,
        test_mode: Optional[str] = None,
    ) -> Dict[str, Any]:

        # ---------------------------
        # Apply CLI overrides
        # ---------------------------
        if annot_clean is not None:
            self.annot_clean = annot_clean
        if yolo_crop is not None:
            self.yolo_crop = yolo_crop
        if yolo_model is not None:
            self.yolo_model = yolo_model.lower()
        if test_mode is not None:
            self.test_mode = test_mode

        # Refresh base input dir
        self.base_dir = Path(
            self.cfg.get("main", {}).get("input_dir", "data/original")
        ).resolve()

        base_root = self.base_dir.parent

        # ======================================================
        # Annotation Cleaner
        # ======================================================
        annot_root = base_root / "annotation_cleaner"
        annot_only = annot_root / "only_annotation_image"
        annot_only_padded = annot_root / "only_annotation_image_padded"
        generated_padded = annot_root / "generated_image_padded"
        generated_final = annot_root / "generated_image"

        if self.annot_clean == "on":
            annot_output_dir = base_root / "generation"
        else:
            annot_output_dir = self.base_dir

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

        # ======================================================
        # YOLO Cropper (Single Source of Truth)
        # ======================================================
        if self.yolo_crop == "on":
            crop_output_dir = (
                annot_output_dir.parent
                / f"{annot_output_dir.name}_crop"
                / self.yolo_model
            )
        else:
            crop_output_dir = annot_output_dir

        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        yolo_cropper_cfg.setdefault("main", {})
        yolo_cropper_cfg["main"]["input_dir"] = str(annot_output_dir)
        yolo_cropper_cfg["main"]["output_dir"] = str(crop_output_dir)
        yolo_cropper_cfg["main"]["model_name"] = self.yolo_model

        # --- Darknet sync ---
        darknet_cfg = yolo_cropper_cfg.get("darknet", {})
        darknet_cfg["model_name"] = self.yolo_model
        yolo_cropper_cfg["darknet"] = darknet_cfg

        # --- Optional explicit sync ---
        if self.yolo_model.startswith("yolov5"):
            yolo_cropper_cfg.setdefault("yolov5", {})
            yolo_cropper_cfg["yolov5"]["model_name"] = self.yolo_model

        if self.yolo_model.startswith("yolov8"):
            yolo_cropper_cfg.setdefault("yolov8", {})
            yolo_cropper_cfg["yolov8"]["model_name"] = self.yolo_model

        self.cfg["yolo_cropper"] = yolo_cropper_cfg

        # ======================================================
        # Data Augmentor
        # ======================================================
        data_aug_cfg = self.cfg.get("data_augmentor", {})
        data_aug_cfg.setdefault("data", {})
        data_aug_cfg["data"]["input_dir"] = str(crop_output_dir)

        if self.demo_mode:
            aug_output_dir = crop_output_dir / "dataset"
        else:
            aug_output_dir = crop_output_dir

        data_aug_cfg["data"]["output_dir"] = str(aug_output_dir)
        self.cfg["data_augmentor"] = data_aug_cfg

        # ======================================================
        # Classifier
        # ======================================================
        classifier_cfg = self.cfg.get("classifier", {})
        classifier_cfg.setdefault("data", {})

        classifier_cfg["data"]["input_dir"] = str(aug_output_dir)

        dynamic_save_dir = self._build_classifier_save_dir(aug_output_dir)
        classifier_cfg.setdefault("train", {})
        classifier_cfg["train"]["save_dir"] = str(dynamic_save_dir)

        metric_root = Path("metrics/classifier")
        classifier_cfg["train"]["metric_dir"] = str(
            metric_root / dynamic_save_dir.relative_to("saved_model/classifier")
        )

        self.cfg["classifier"] = classifier_cfg

        # ======================================================
        # Persist main flags
        # ======================================================
        self.cfg["main"]["annot_clean"] = self.annot_clean
        self.cfg["main"]["yolo_crop"] = self.yolo_crop
        self.cfg["main"]["yolo_model"] = self.yolo_model

        return self.cfg

    # --------------------------------------------------------
    def save(self, output_path: Optional[str] = None):
        target = Path(output_path or self.config_path)
        with open(target, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False, allow_unicode=True)
        print(f"Updated config saved → {target}")
