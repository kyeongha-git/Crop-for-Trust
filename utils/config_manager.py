#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic Configuration Manager.

Handles the loading, updating, and saving of YAML configurations.
Resolves relative paths dynamically based on pipeline stages and CLI overrides.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigManager:
    """
    Manages configuration state, applying CLI overrides and resolving
    inter-module path dependencies.
    """

    def __init__(self, config_path: str) -> None:
        self.config_path = Path(config_path)
        self.cfg = self._load_yaml()

        main_cfg = self.cfg.get("main", {})

        # Base paths (kept relative)
        self.base_dir = Path(main_cfg.get("input_dir", "data/original"))
        self.saved_model_path = Path(main_cfg.get("saved_model", "saved_model"))

        # Pipeline flags
        self.demo_mode = main_cfg.get("demo", False)
        self.annot_clean = main_cfg.get("annot_clean", True)
        self.yolo_crop = main_cfg.get("yolo_crop", True)

        self.yolo_model = main_cfg.get("yolo_model", "yolov8s").lower()
        self.annot_clean_test_mode = main_cfg.get("annot_clean_test_mode", False)
        
        assert isinstance(self.annot_clean_test_mode, bool)

    def _load_yaml(self) -> Dict[str, Any]:
        """Loads YAML configuration from disk."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_classifier_save_dir(self, classifier_input: Path) -> Path:
        """
        Constructs the hierarchical save directory for the classifier.

        Args:
            classifier_input (Path): Input directory path for the classifier.

        Returns:
            Path: Resolved relative path for saving classifier models.
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

    def update_paths(
        self,
        annot_clean: Optional[bool] = None,
        yolo_crop: Optional[bool] = None,
        yolo_model: Optional[str] = None,
        annot_clean_test_mode: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Updates configuration paths based on active flags and CLI overrides.

        Args:
            annot_clean (Optional[bool]): Override for annotation cleaning flag.
            yolo_crop (Optional[bool]): Override for YOLO cropping flag.
            yolo_model (Optional[str]): Override for YOLO model name.
            annot_clean_test_mode (Optional[bool]): Override for annotation test mode.

        Returns:
            Dict[str, Any]: The updated configuration dictionary.
        """
        # Apply CLI overrides
        if annot_clean is not None:
            self.annot_clean = annot_clean
        if yolo_crop is not None:
            self.yolo_crop = yolo_crop
        if yolo_model is not None:
            self.yolo_model = yolo_model.lower()
        if annot_clean_test_mode is not None:
            self.annot_clean_test_mode = annot_clean_test_mode

        # Refresh base input directory
        self.base_dir = Path(
            self.cfg.get("main", {}).get("input_dir", "data/original")
        )
        base_root = self.base_dir.parent

        # 1. Annotation Cleaner Configuration
        annot_root = base_root / "annotation_cleaner"
        annot_only = annot_root / "only_annotation_image"
        annot_only_padded = annot_root / "only_annotation_image_padded"

        if self.annot_clean_test_mode:
            generated_padded = annot_root / "generated_image_padded_test"
            generated_final = annot_root / "generated_image_test"
            annot_output_dir = base_root / "generation_test"
        else:
            generated_padded = annot_root / "generated_image_padded"
            generated_final = annot_root / "generated_image"
            annot_output_dir = base_root / "generation"

        if not self.annot_clean:
            annot_output_dir = self.base_dir

        if self.annot_clean:
            annotation_cfg = self.cfg.get("annotation_cleaner", {})

            annotation_cfg.setdefault("main", {})
            annotation_cfg["main"]["input_dir"] = self.base_dir.as_posix()
            annotation_cfg["main"]["output_dir"] = annot_output_dir.as_posix()

            annotation_cfg.setdefault("image_padding", {})
            annotation_cfg["image_padding"]["input_dir"] = annot_only.as_posix()
            annotation_cfg["image_padding"]["output_dir"] = annot_only_padded.as_posix()

            annotation_cfg.setdefault("annotation_clean", {})
            annotation_cfg["annotation_clean"]["input_dir"] = annot_only_padded.as_posix()
            annotation_cfg["annotation_clean"]["output_dir"] = generated_padded.as_posix()
            annotation_cfg["annotation_clean"]["test_mode"] = self.annot_clean_test_mode

            annotation_cfg.setdefault("restore_crop", {})
            annotation_cfg["restore_crop"]["input_dir"] = generated_padded.as_posix()
            annotation_cfg["restore_crop"]["output_dir"] = generated_final.as_posix()
            annotation_cfg["restore_crop"]["metadata_root"] = annot_only_padded.as_posix()

            annotation_cfg.setdefault("evaluate", {})
            annotation_cfg["evaluate"]["orig_dir"] = annot_only.as_posix()
            annotation_cfg["evaluate"]["gen_dir"] = generated_final.as_posix()
            annotation_cfg["evaluate"]["yolo_model"] = (
                Path("saved_model")
                / "yolo_cropper"
                / f"{self.yolo_model}.pt"
            ).as_posix()

            self.cfg["annotation_cleaner"] = annotation_cfg

        # 2. YOLO Cropper Configuration
        if self.yolo_crop:
            crop_output_dir = (
                annot_output_dir.parent
                / f"{annot_output_dir.name}_crop"
                / self.yolo_model
            )
        else:
            crop_output_dir = annot_output_dir

        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        yolo_cropper_cfg.setdefault("main", {})
        yolo_cropper_cfg["main"]["input_dir"] = annot_output_dir.as_posix()
        yolo_cropper_cfg["main"]["output_dir"] = crop_output_dir.as_posix()
        yolo_cropper_cfg["main"]["model_name"] = self.yolo_model

        darknet_cfg = yolo_cropper_cfg.get("darknet", {})
        darknet_cfg["model_name"] = self.yolo_model
        yolo_cropper_cfg["darknet"] = darknet_cfg

        if self.yolo_model.startswith("yolov5"):
            yolo_cropper_cfg.setdefault("yolov5", {})
            yolo_cropper_cfg["yolov5"]["model_name"] = self.yolo_model

        if self.yolo_model.startswith("yolov8"):
            yolo_cropper_cfg.setdefault("yolov8", {})
            yolo_cropper_cfg["yolov8"]["model_name"] = self.yolo_model

        self.cfg["yolo_cropper"] = yolo_cropper_cfg

        # 3. Data Augmentor Configuration
        data_aug_cfg = self.cfg.get("data_augmentor", {})
        data_aug_cfg.setdefault("data", {})
        data_aug_cfg["data"]["input_dir"] = crop_output_dir.as_posix()

        if self.demo_mode:
            aug_output_dir = crop_output_dir / "dataset"
        else:
            aug_output_dir = crop_output_dir

        data_aug_cfg["data"]["output_dir"] = aug_output_dir.as_posix()
        self.cfg["data_augmentor"] = data_aug_cfg

        # 4. Classifier Configuration
        classifier_cfg = self.cfg.get("classifier", {})
        classifier_cfg.setdefault("data", {})
        classifier_cfg.setdefault("train", {})

        classifier_cfg["data"]["input_dir"] = aug_output_dir.as_posix()

        dynamic_save_dir = self._build_classifier_save_dir(aug_output_dir)
        relative_subpath = dynamic_save_dir.relative_to("saved_model/classifier")

        classifier_cfg["train"]["save_dir"] = dynamic_save_dir.as_posix()
        classifier_cfg["train"]["metric_dir"] = (
            Path("metrics/classifier") / relative_subpath
        ).as_posix()
        classifier_cfg["train"]["check_dir"] = (
            Path("checkpoints/classifier") / relative_subpath
        ).as_posix()

        self.cfg["classifier"] = classifier_cfg

        # Update main flags in config
        self.cfg["main"]["annot_clean"] = self.annot_clean
        self.cfg["main"]["yolo_crop"] = self.yolo_crop
        self.cfg["main"]["yolo_model"] = self.yolo_model

        return self.cfg

    def save(self, output_path: Optional[str] = None) -> None:
        """
        Saves the current configuration state to a YAML file.

        Args:
            output_path (Optional[str]): Target path. Defaults to initial config path.
        """
        target = Path(output_path or self.config_path)
        with open(target, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False, allow_unicode=True)
        print(f"Configuration saved to: {target}")