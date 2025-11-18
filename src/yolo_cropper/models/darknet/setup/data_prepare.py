#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_prepare.py
---------------
This module prepares dataset splits and metadata files for Darknet-based YOLO models.
It generates train/valid/test file lists, class label files, and configuration files
(obj.data and obj.names) based on the provided configuration object.
"""

import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[5]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class DarknetDataPreparer:
    """
    Handles dataset preparation for YOLOv2 or YOLOv4 training with Darknet.

    This class automates the creation of data split lists and configuration
    files, making the dataset compatible with the expected Darknet directory
    structure. It supports both flat and Roboflow-style folder layouts.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset preparer using a configuration dictionary.

        Args:
            config (Dict[str, Any]): The full configuration object injected
                from the main controller, containing dataset and Darknet settings.

        """
        self.logger = get_logger("yolo_cropper.DarknetDataPreparer")

        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.darknet_cfg = self.yolo_cropper_cfg.get("darknet", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        self.base_dir = Path(
            self.dataset_cfg.get("train_data_dir", "data/yolo_cropper")
        )
        self.darknet_data_dir = Path(
            self.darknet_cfg.get("darknet_data_dir", "third_party/darknet/data")
        )
        self.model_name = self.main_cfg.get("model_name", "yolov2").lower()

        self.darknet_data_dir.mkdir(parents=True, exist_ok=True)

        if not self.base_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.base_dir}")

        self.logger.info(
            f"Initialized DarknetDataPreparer for {self.model_name.upper()} "
            f"→ base_dir={self.base_dir}/{self.model_name}, darknet_data_dir={self.darknet_data_dir}"
        )

    def prepare(self):
        """
        Prepare all Darknet-compatible dataset files.

        This method generates:
            - train.txt / valid.txt / test.txt (image paths)
            - obj.data and obj.names (Darknet configuration files)

        Returns:
            dict: A dictionary containing paths to all generated files.
        """
        self.logger.info(
            f"Preparing Darknet dataset for {self.model_name.upper()} → {self.darknet_data_dir}"
        )
        self._generate_split_lists()
        class_names = self._get_class_names()
        self._generate_obj_files(class_names)
        self.logger.info("Darknet dataset preparation complete.")
        return {
            "train_txt": str(self.darknet_data_dir / "train.txt"),
            "valid_txt": str(self.darknet_data_dir / "valid.txt"),
            "test_txt": str(self.darknet_data_dir / "test.txt"),
            "obj_data": str(self.darknet_data_dir / "obj.data"),
            "obj_names": str(self.darknet_data_dir / "obj.names"),
        }

    def _generate_split_lists(self):
        """
        Generate image list files (train.txt, valid.txt, test.txt).

        The method scans dataset subfolders and writes full image paths
        for each split, ensuring compatibility with Darknet’s expected structure.
        """
        exts = (".jpg", ".jpeg", ".png", ".bmp")

        for split in ["train", "valid", "test"]:
            base_dir = self.base_dir / self.model_name / split
            if not base_dir.exists():
                self.logger.warning(f"Split folder missing: {base_dir}")
                continue

            img_dir = (
                base_dir / "images" if (base_dir / "images").exists() else base_dir
            )
            output_file = self.darknet_data_dir / f"{split}.txt"

            images = [
                str(p.resolve())
                for p in sorted(img_dir.glob("*"))
                if p.suffix.lower() in exts
            ]
            if not images:
                raise ValueError(f"No images found in {img_dir}")

            output_file.write_text("\n".join(images) + "\n", encoding="utf-8")
            self.logger.info(
                f"  └─ [{split}] {len(images)} images listed → {output_file.name}"
            )

    def _get_class_names(self):
        """
        Retrieve class names from existing label files.

        This method checks for `_darknet.labels` or `_classes.txt` files
        inside the dataset directory and loads class names from the first
        available file.

        Returns:
            list[str]: A list of class names.

        """
        candidates = [
            self.base_dir / self.model_name / "train" / "_darknet.labels",
            self.base_dir / "_classes.txt",
        ]
        for path in candidates:
            if path.exists():
                class_names = [
                    c.strip()
                    for c in path.read_text(encoding="utf-8").splitlines()
                    if c.strip()
                ]
                self.logger.info(
                    f"Loaded {len(class_names)} classes from {path.name}: {class_names}"
                )
                return class_names
        raise FileNotFoundError("No _darknet.labels or _classes.txt found.")

    def _generate_obj_files(self, class_names):
        """
        Create obj.data and obj.names files for Darknet training.

        """
        obj_data = self.darknet_data_dir / "obj.data"
        obj_names = self.darknet_data_dir / "obj.names"

        num_classes = len(class_names)
        backup_dir = Path(f"checkpoints/yolo_cropper/{self.model_name}")
        backup_dir.mkdir(parents=True, exist_ok=True)

        obj_data_content = (
            f"classes = {num_classes}\n"
            f"train = data/train.txt\n"
            f"valid = data/valid.txt\n"
            f"names = data/obj.names\n"
            f"backup = {backup_dir.resolve()}\n"
        )

        obj_data.write_text(obj_data_content, encoding="utf-8")
        obj_names.write_text("\n".join(class_names) + "\n", encoding="utf-8")

        self.logger.info(f"obj.data / obj.names created ({num_classes} classes)")
        self.logger.info(f"Backup path set to: {backup_dir.resolve()}")

    def update_backup_path(self, backup_dir: str):
        """
        Update the backup path defined in obj.data.

        """
        obj_data_path = self.darknet_data_dir / "obj.data"
        if not obj_data_path.exists():
            raise FileNotFoundError(f"obj.data not found: {obj_data_path}")

        lines = obj_data_path.read_text(encoding="utf-8").splitlines()
        new_lines = [
            f"backup = {backup_dir}" if line.strip().startswith("backup") else line
            for line in lines
        ]
        obj_data_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        self.logger.info(f"updated backup path → {backup_dir}")
