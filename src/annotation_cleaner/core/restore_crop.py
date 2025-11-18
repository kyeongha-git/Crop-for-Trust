#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
restore_crop.py
-------------------
This module restores generated images (1024×1024) to their original size
using stored padding metadata from the preprocessing stage.
Each restored image matches the original dimensions before padding.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import cv2

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger, setup_logging


class RestoreCropper:
    """
    Restores images to their original dimensions using padding metadata.

    This class reverses the padding applied during preprocessing by cropping
    the central region of 1024×1024 generated images back to their original size.
    It uses the JSON metadata produced during image padding to determine
    the crop coordinates for each image.
    """

    def __init__(
        self,
        input_dir: str,  # generated_image_padded
        output_dir: str,  # generated_image
        meta_dir: str,  # only_annotation_image_padded
        categories: Optional[List[str]] = None,
        metadata_name: str = "padding_info.json",
    ):
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("RestoreCrop")

        # --- Directory and configuration setup ---
        self.input_root = Path(input_dir)
        self.meta_root = Path(meta_dir)
        self.output_root = Path(output_dir)
        self.categories = categories or ["repair", "replace"]
        self.meta_name = metadata_name

        self.logger.info(f"Input folder: {self.input_root}")
        self.logger.info(f"Metadata folder: {self.meta_root}")
        self.logger.info(f"Output folder: {self.output_root}")

    # ============================================================
    # Internal Utilities
    # ============================================================
    def _load_metadata(self, meta_path: Path) -> Optional[dict]:
        """
        Loads the padding metadata (JSON) for a given category.

        Returns:
            dict: A mapping of image names to padding information.
            None: If the metadata file is missing or cannot be loaded.
        """
        if not meta_path.exists():
            self.logger.warning(f"Metadata not found: {meta_path}")
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {os.path.splitext(k)[0]: v for k, v in data.items()}
        except Exception as e:
            self.logger.error(f"Failed to load metadata ({meta_path}): {e}")
            return None

    def _restore_single_image(
        self, img_path: Path, meta: dict, save_path: Path
    ) -> bool:
        """
        Restores a single image to its original dimensions.

        Uses the stored padding information to crop out the valid region
        from the 1024×1024 generated image.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            self.logger.warning(f"Failed to read image: {img_path.name}")
            return False

        h_orig, w_orig = meta["orig_size"]
        top, left = meta["pad_info"]["top"], meta["pad_info"]["left"]
        roi = img[top : top + h_orig, left : left + w_orig]

        success = cv2.imwrite(str(save_path), roi)
        if success:
            self.logger.info(f"Restored: {save_path.name}")
            return True
        else:
            self.logger.error(f"Failed to save: {save_path.name}")
            return False

    # ============================================================
    # Public API
    # ============================================================
    def run(self):
        """
        Executes the full restoration process for all categories.

        - Iterates through each category folder.
        - Loads corresponding padding metadata.
        - Restores each generated image to its original size.
        - Copies images without padding metadata directly.
        """
        if not self.input_root.exists():
            raise FileNotFoundError(f"Input folder not found: {self.input_root}")

        self.output_root.mkdir(parents=True, exist_ok=True)
        total_restored = 0

        for category in self.categories:
            in_dir = self.input_root / category
            meta_path = self.meta_root / category / self.meta_name
            out_dir = self.output_root / category
            out_dir.mkdir(parents=True, exist_ok=True)

            if not in_dir.exists():
                self.logger.warning(f"Input folder missing: {in_dir}")
                continue

            metadata = self._load_metadata(meta_path)
            if not metadata:
                continue

            restored_count = 0
            for file in sorted(os.listdir(in_dir)):
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                name = os.path.splitext(file)[0]
                input_path = in_dir / file
                save_path = out_dir / file

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
