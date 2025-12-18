#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
image_padding.py
----------------
Pads input images to a fixed square size (default: 1024×1024).

This module is fully configuration-driven:
- Receives only `config` at initialization
- Reads required paths and parameters internally
- Exposes a single public API: `run()`
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger, setup_logging


class ImagePadder:
    """
    Pads images to a target square size and records padding metadata.

    Configuration section used:
        annotation_cleaner:
            main:
                categories
                metadata_name
            image_padding:
                input_dir
                output_dir
                target_size
    """

    DEFAULT_PADDING_COLOR = (0, 0, 0)

    def __init__(self, config: Dict):
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("annotation_cleaner.ImagePadder")

        self.cfg = config
        global_main_cfg = self.cfg.get("main", {})
        cleaner_cfg = self.cfg.get("annotation_cleaner", {})
        self.main_cfg = cleaner_cfg.get("main", {})
        self.pad_cfg = cleaner_cfg.get("image_padding", {})

        self.input_root = Path(
            self.pad_cfg.get("input_dir", self.main_cfg.get("input_dir", "data/original"))
        ).resolve()

        self.output_root = Path(
            self.pad_cfg.get("output_dir", "data/annotation_cleaner/only_annotation_image_padded")
        ).resolve()

        self.categories: List[str] = global_main_cfg.get(
            "categories", ["repair", "replace"]
        )

        self.target_size: int = int(self.pad_cfg.get("target_size", 1024))
        self.metadata_name: str = self.main_cfg.get(
            "metadata_name", "padding_info.json"
        )

        self.padding_color = self.DEFAULT_PADDING_COLOR

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        self.logger.info("Initialized ImagePadder")
        self.logger.info(f" - Input dir  : {self.input_root}")
        self.logger.info(f" - Output dir : {self.output_root}")
        self.logger.info(f" - Target size: {self.target_size}")

    def _pad_image(self, image_path: Path, save_path: Path) -> Optional[dict]:
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.error(f"{image_path.name}: Failed to load image.")
            return None

        h, w = img.shape[:2]

        # Skip padding if already large enough
        if h >= self.target_size and w >= self.target_size:
            self.logger.info(f"{image_path.name}: Already large enough → copy only.")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy(str(image_path), str(save_path))
            except Exception as e:
                self.logger.error(f"{image_path.name}: Copy failed ({e})")
            return None

        # Calculate padding
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

    def run(self):
        """
        Pads all images under each category directory and saves metadata.
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
