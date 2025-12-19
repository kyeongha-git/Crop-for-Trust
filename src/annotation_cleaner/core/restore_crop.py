#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for restoring images to their original dimensions.

This script uses the metadata generated during the padding stage to crop
processed images back to their original size and aspect ratio, ensuring
spatial consistency with the input dataset.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger, setup_logging


class RestoreCropper:
    """
    Controller for the image restoration phase.
    
    Reverses the padding operation by cropping images based on saved metadata.
    """

    def __init__(self, config: Dict) -> None:
        """
        Args:
            config (Dict): Configuration dictionary containing paths and parameters.
        """
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("annotation_cleaner.RestoreCropper")

        # Configuration setup
        self.cfg = config
        global_main_cfg = self.cfg.get("main", {})
        cleaner_cfg = self.cfg.get("annotation_cleaner", {})
        self.main_cfg = cleaner_cfg.get("main", {})
        self.restore_cfg = cleaner_cfg.get("restore_crop", {})

        # Path setup
        self.input_root = Path(
            self.restore_cfg.get("input_dir", "data/annotation_cleaner/generated_image_padded")
        ).resolve()

        self.output_root = Path(
            self.restore_cfg.get("output_dir", "data/annotation_cleaner/generated_image")
        ).resolve()

        self.meta_root = Path(
            self.restore_cfg.get("metadata_root", "data/annotation_cleaner/only_annotation_image_padded")
        ).resolve()

        self.categories: List[str] = global_main_cfg.get(
            "categories", ["repair", "replace"]
        )

        self.meta_name: str = self.main_cfg.get(
            "metadata_name", "padding_info.json"
        )

        test_mode = "test" in self.input_root.name

        self.logger.info("Initialized RestoreCropper")
        self.logger.info(f" - Input dir   : {self.input_root}")
        self.logger.info(f" - Output dir  : {self.output_root}")
        self.logger.info(f" - Metadata dir: {self.meta_root}")
        self.logger.info(f" - Test mode   : {test_mode}")

    def _load_metadata(self, meta_path: Path) -> Optional[Dict[str, Any]]:
        """
        Loads and normalizes padding metadata from a JSON file.

        The keys in the returned dictionary are normalized to filenames without 
        extensions to ensure robust matching against image files.

        Args:
            meta_path (Path): Path to the JSON metadata file.

        Returns:
            Optional[Dict[str, Any]]: Metadata dictionary or None if loading fails.
        """
        if not meta_path.exists():
            self.logger.warning(f"Metadata not found: {meta_path}")
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Normalize keys: remove extensions for consistent lookups
            return {os.path.splitext(k)[0]: v for k, v in data.items()}
        except Exception as e:
            self.logger.error(f"Failed to load metadata ({meta_path}): {e}")
            return None

    def _restore_single_image(
        self, img_path: Path, meta: Dict[str, Any], save_path: Path
    ) -> bool:
        """
        Crops a single image back to its original dimensions.

        Args:
            img_path (Path): Path to the padded image.
            meta (Dict[str, Any]): Padding info containing 'orig_size' and 'pad_info'.
            save_path (Path): Destination path for the restored image.

        Returns:
            bool: True if successful, False otherwise.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            self.logger.warning(f"Failed to read image: {img_path.name}")
            return False

        # Retrieve original dimensions and padding offsets
        h_orig, w_orig = meta["orig_size"]
        top = meta["pad_info"]["top"]
        left = meta["pad_info"]["left"]

        # Crop to Region of Interest (ROI)
        roi = img[top : top + h_orig, left : left + w_orig]

        success = cv2.imwrite(str(save_path), roi)
        if success:
            self.logger.info(f"Restored: {save_path.name}")
            return True
        else:
            self.logger.error(f"Failed to save: {save_path.name}")
            return False

    def run(self) -> None:
        """
        Executes the restoration process for all categories.
        Iterates through files and applies cropping based on loaded metadata.
        """
        if not self.input_root.exists():
            raise FileNotFoundError(f"Input folder not found: {self.input_root}")

        self.output_root.mkdir(parents=True, exist_ok=True)
        total_restored = 0

        for category in self.categories:
            in_dir = self.input_root / category
            out_dir = self.output_root / category
            meta_path = self.meta_root / category / self.meta_name

            out_dir.mkdir(parents=True, exist_ok=True)

            if not in_dir.exists():
                self.logger.warning(f"Input folder missing: {in_dir}")
                continue

            metadata = self._load_metadata(meta_path)
            if not metadata:
                self.logger.warning(f"No metadata for category: {category}")
                continue

            restored_count = 0

            # Process each file in the category
            for file in sorted(os.listdir(in_dir)):
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                name = os.path.splitext(file)[0]
                input_path = in_dir / file
                save_path = out_dir / file

                # Fallback: Copy if no metadata exists for this file
                if name not in metadata:
                    shutil.copy(input_path, save_path)
                    self.logger.info(f"{file}: Copied (no padding metadata)")
                    continue

                success = self._restore_single_image(
                    input_path, metadata[name], save_path
                )
                restored_count += int(success)

            self.logger.info(
                f"{category}: {restored_count} images restored → {out_dir}"
            )
            total_restored += restored_count

        self.logger.info(f"Restoration complete ({total_restored} total files)")