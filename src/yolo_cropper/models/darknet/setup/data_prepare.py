#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Darknet Data Preparation Module.

Generates dataset manifests (train/val/test lists) and configuration files
(obj.data, obj.names) required for Darknet training.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[5]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class DarknetDataPreparer:
    """
    Manages the creation of Darknet-compatible dataset files and directory structures.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = get_logger("yolo_cropper.DarknetDataPreparer")

        self.cfg = config
        global_main_cfg = self.cfg.get("main", {})
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.darknet_cfg = self.yolo_cropper_cfg.get("darknet", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        self.categories = global_main_cfg.get("categories", [])
        if not self.categories:
            raise ValueError("Categories must be defined in configuration.")

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

        self.logger.info(f"Initialized DataPreparer (Model: {self.model_name.upper()})")

    def _generate_split_lists(self) -> None:
        """
        Generates text files listing absolute image paths for train, valid, and test splits.
        """
        exts = (".jpg", ".jpeg", ".png")

        for split in ["train", "valid", "test"]:
            base_dir = self.base_dir / self.model_name / split
            if not base_dir.exists():
                self.logger.warning(f"Split folder missing: {base_dir}")
                continue

            # Support both flat structure and images/ labels/ structure
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
            self.logger.info(f"Generated {split}.txt ({len(images)} images)")

    def _generate_obj_files(self, class_names: List[str]) -> None:
        """
        Creates 'obj.data' and 'obj.names' configuration files.
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

        self.logger.info(f"Generated obj.data and obj.names ({num_classes} classes)")

    def run(self) -> Dict[str, str]:
        """
        Executes the dataset preparation process.

        Returns:
            Dict[str, str]: A dictionary of generated file paths.
        """
        self.logger.info("Starting dataset preparation")

        self._generate_split_lists()
        self._generate_obj_files(self.categories)

        self.logger.info("Dataset preparation complete")
        return {
            "train_txt": str(self.darknet_data_dir / "train.txt"),
            "valid_txt": str(self.darknet_data_dir / "valid.txt"),
            "test_txt": str(self.darknet_data_dir / "test.txt"),
            "obj_data": str(self.darknet_data_dir / "obj.data"),
            "obj_names": str(self.darknet_data_dir / "obj.names"),
        }