#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
image_padding.py
----------------
This module pads input images to a fixed square size (default: 1024×1024).
Each image is centered within the new canvas, and padding information is saved
as JSON metadata for later reference or restoration.
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


class ImagePadder:
    """
    Pads images to a target square size and records metadata about the padding.

    Features:
    - Centers each image on a black background.
    - Automatically skips images that are already large enough.
    - Saves padding information (top, bottom, left, right) in a JSON file.
    - Handles corrupted or unreadable image files safely.
    """

    DEFAULT_PADDING_COLOR = (0, 0, 0)

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        categories: Optional[List[str]] = None,
        target_size: int = 1024,
        metadata_name: str = "padding_info.json",
    ):
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("ImagePadder")

        self.input_root = Path(input_dir)
        self.output_root = Path(output_dir)
        self.categories = categories or ["repair", "replace"]
        self.target_size = target_size
        self.metadata_name = metadata_name
        self.padding_color = self.DEFAULT_PADDING_COLOR

        self.logger.info(f"Input directory: {self.input_root}")
        self.logger.info(f"Output directory: {self.output_root}")
        self.logger.info(f"Target size: {self.target_size}")

    # ============================================================
    # Internal Utility: Padding Single Image
    # ============================================================
    def _pad_image(self, image_path: Path, save_path: Path) -> Optional[dict]:
        """
        Pads a single image to the target size while keeping it centered.

        Returns:
            dict: A dictionary containing the original image size and padding details.
            None: If the image failed to load or save.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.error(f"{image_path.name}: Failed to load image.")
            return None

        h, w = img.shape[:2]

        # Skip padding if already larger than target
        if h >= self.target_size and w >= self.target_size:
            self.logger.info(f"{image_path.name}: Already large enough → copy only.")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy(str(image_path), str(save_path))
            except Exception as e:
                self.logger.error(f"{image_path.name}: Copy failed ({e})")
            return None

        # Calculate padding (preventing negative values)
        top = max(0, (self.target_size - h) // 2)
        bottom = max(0, self.target_size - h - top)
        left = max(0, (self.target_size - w) // 2)
        right = max(0, self.target_size - w - left)

        try:
            padded = cv2.copyMakeBorder(
                img,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=self.padding_color,
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(save_path), padded)

            if not success:
                self.logger.error(f"{image_path.name}: Failed to save padded image.")
                return None

            return {
                "orig_size": [h, w],
                "pad_info": {
                    "top": top,
                    "left": left,
                    "bottom": bottom,
                    "right": right,
                },
            }

        except Exception as e:
            self.logger.error(f"{image_path.name}: Error during padding ({e})")
            return None

    # ============================================================
    # Public API
    # ============================================================
    def run(self):
        """
        Pads all images within each category folder and saves metadata.

        - Iterates through all category subfolders (e.g., "repair", "replace").
        - Pads smaller images to the target size.
        - Saves both the padded images and a JSON file containing padding info.
        - Logs skipped, copied, and processed files for transparency.
        """
        if not self.input_root.exists():
            raise FileNotFoundError(f"Input folder not found: {self.input_root}")

        self.output_root.mkdir(parents=True, exist_ok=True)

        for category in self.categories:
            in_dir = self.input_root / category
            out_dir = self.output_root / category
            meta_path = out_dir / self.metadata_name

            if not in_dir.exists():
                self.logger.warning(f"Missing folder: {in_dir}")
                continue

            self.logger.info(f"Processing category: {category}")
            metadata = {}

            for file in sorted(os.listdir(in_dir)):
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                input_path = in_dir / file
                save_path = out_dir / file
                info = self._pad_image(input_path, save_path)
                if info:
                    metadata[file] = info

            # Save metadata
            if metadata:
                try:
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=4, ensure_ascii=False)
                    self.logger.info(f"Padding complete → {out_dir}")
                    self.logger.info(f"Metadata saved → {meta_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save metadata ({meta_path}): {e}")
            else:
                self.logger.info(f"{category}: No new padded images created.")
