#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
config_manager.py
-----------------
Dynamic Config Manager (with CLI Overrides)

Features
---------------
- Automatically updates all submodule paths relative to `main.input_dir`
- Updates annotation_cleaner paths only when `annot_clean == "on"`
- Dynamically sets YOLO input/output directories based on `yolo_crop` and `yolo_model`
- Preserves existing paths for modules that are turned "off"
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigManager:
    """
    Dynamic Config Manager that safely updates YAML configurations.

    This class centralizes all path and mode management logic for the pipeline.
    It loads `config.yaml`, applies CLI overrides, and automatically adjusts
    directory paths for downstream modules such as:
    - AnnotationCleaner
    - YOLOCropper
    - DataAugmentor
    - Classifier

    """

    def __init__(self, config_path: str):
        """
        Initialize the ConfigManager and load the YAML file.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config_path = Path(config_path)
        self.cfg = self._load_yaml()

        main_cfg = self.cfg.get("main", {})
        self.base_dir = Path(main_cfg.get("input_dir", "data/original")).resolve()
        self.test_mode = (
            self.cfg.get("annotation_cleaner", {})
            .get("annotation_clean", {})
            .get("test_mode", "off")
        )
        self.annot_clean = main_cfg.get("annot_clean", "on")
        self.yolo_crop = main_cfg.get("yolo_crop", "on")
        self.yolo_model = main_cfg.get("yolo_model", "yolov8s")

    # --------------------------------------------------------
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # --------------------------------------------------------
    def update_paths(
        self,
        annot_clean: Optional[str] = None,
        yolo_crop: Optional[str] = None,
        yolo_model: Optional[str] = None,
        test_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dynamically update module paths and parameters based on CLI overrides.

        Returns:
            Dict[str, Any]: Updated configuration dictionary ready for saving.
        """
        # Apply CLI overrides
        if annot_clean is not None:
            self.annot_clean = annot_clean
        if yolo_crop is not None:
            self.yolo_crop = yolo_crop
        if yolo_model is not None:
            self.yolo_model = yolo_model
        if test_mode is not None:
            self.test_mode = test_mode

        # Refresh base_dir
        self.base_dir = Path(
            self.cfg.get("main", {}).get("input_dir", "data/original")
        ).resolve()

        # === Base Paths ===
        base_root = self.base_dir.parent  # e.g., data/sample
        annot_root = base_root / "annotation_cleaner"

        # === AnnotationCleaner Paths ===
        annot_only = annot_root / "only_annotation_image"
        annot_only_padded = annot_root / "only_annotation_image_padded"
        generated_padded = annot_root / "generated_image_padded"
        generated_final = annot_root / "generated_image"

        # === Determine Output Directory ===
        if self.annot_clean == "on":
            annot_output_dir = base_root / "generation"
        else:
            annot_output_dir = self.base_dir  # Use original images if cleaner is off

        # === YOLO Cropper Output ===
        if self.yolo_crop == "on":
            crop_output_dir = (
                annot_output_dir.parent
                / f"{annot_output_dir.name}_crop"
                / self.yolo_model
            )
        else:
            crop_output_dir = annot_output_dir

        # ==================================================================
        # AnnotationCleaner
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
            print("AnnotationCleaner OFF → Skipping path updates")

        # ==================================================================
        # YOLO Cropper
        # ==================================================================
        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        yolo_cropper_cfg.setdefault("main", {})
        yolo_cropper_cfg["main"]["input_dir"] = str(annot_output_dir)
        yolo_cropper_cfg["main"]["output_dir"] = str(crop_output_dir)
        yolo_cropper_cfg["main"]["model_name"] = self.yolo_model

        # ==================================================================
        # DataAugmentor
        # ==================================================================
        data_augmentor_cfg = self.cfg.get("data_augmentor", {})
        data_augmentor_cfg.setdefault("data", {})
        data_augmentor_cfg["data"]["input_dir"] = str(crop_output_dir)
        data_augmentor_cfg["data"]["output_dir"] = str(crop_output_dir)

        # ==================================================================
        # Classifier
        # ==================================================================
        classifier_cfg = self.cfg.get("classifier", {})
        classifier_cfg.setdefault("data", {})
        classifier_cfg["data"]["input_dir"] = str(crop_output_dir)

        # ==================================================================
        # Main Config Update
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
        """
        Save the updated configuration to a YAML file.

        Args:
            output_path (Optional[str]): Optional custom output path.
                Defaults to overwriting the original configuration file.
        """
        target_path = Path(output_path or self.config_path)
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False, allow_unicode=True)
        print(f"Updated config saved → {target_path}")
