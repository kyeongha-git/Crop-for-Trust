#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for image padding operations.

Resizes images to a fixed square resolution (default: 1024x1024) by adding
borders (padding) while preserving the original aspect ratio. Metadata regarding
the padding offsets is saved to allow for precise restoration later.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from utils.logging import get_logger, setup_logging


class ImagePadder:
    """
    Handles image padding to a target square size and records padding metadata.
    """

    DEFAULT_PADDING_COLOR: Tuple[int, int, int] = (0, 0, 0)

    def __init__(self, config: Dict) -> None:
        """
        Args:
            config (Dict): Configuration dictionary containing paths and parameters.
        """
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("annotation_cleaner.ImagePadder")

        # Configuration setup
        self.cfg = config
        global_main_cfg = self.cfg.get("main", {})
        cleaner_cfg = self.cfg.get("annotation_cleaner", {})
        self.main_cfg = cleaner_cfg.get("main", {})
        self.pad_cfg = cleaner_cfg.get("image_padding", {})

        # Path setup
        self.input_root = Path(
            self.pad_cfg.get("input_dir", self.main_cfg.get("input_dir", "data/original"))
        ).resolve()

        self.output_root = Path(
            self.pad_cfg.get("output_dir", "data/annotation_cleaner/only_annotation_image_padded")
        ).resolve()

        self.categories: List[str] = global_main_cfg.get(
            "categories", ["repair", "replace"]
        )

        # Padding parameters
        self.target_size: int = int(self.pad_cfg.get("target_size", 1024))
        self.metadata_name: str = self.main_cfg.get(
            "metadata_name", "padding_info.json"
        )
        self.padding_color = self.DEFAULT_PADDING_COLOR

        self.logger.info("Initialized ImagePadder")
        self.logger.info(f" - Input dir   : {self.input_root}")
        self.logger.info(f" - Output dir  : {self.output_root}")
        self.logger.info(f" - Target size : {self.target_size}")

    def _pad_image(self, image_path: Path, save_path: Path) -> Optional[Dict[str, Any]]:
        """
        Pads a single image to the target square size.
        
        If the image is already larger than the target size, it is copied as-is.
        Otherwise, symmetric padding is applied to center the image.

        Args:
            image_path (Path): Path to the source image.
            save_path (Path): Path where the processed image will be saved.

        Returns:
            Optional[Dict[str, Any]]: Metadata dict with original size and padding offsets,
                                      or None if processing failed.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.error(f"Failed to load image: {image_path.name}")
            return None

        h, w = img.shape[:2]

        # Case 1: Image dimensions meet or exceed target size
        if h >= self.target_size and w >= self.target_size:
            self.logger.debug(f"Skipping padding (size sufficient): {image_path.name}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy(str(image_path), str(save_path))
            except Exception as e:
                self.logger.error(f"Copy failed for {image_path.name}: {e}")
            return None

        # Case 2: Apply padding (Center the image)
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
                self.logger.error(f"Failed to save padded image: {image_path.name}")
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
            self.logger.error(f"Error padding {image_path.name}: {e}")
            return None

    def run(self) -> None:
        """
        Executes the padding process for all configured categories.
        Aggregates and saves metadata for restoration.
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

            # Iterate and process images
            for file in sorted(os.listdir(in_dir)):
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                input_path = in_dir / file
                save_path = out_dir / file

                info = self._pad_image(input_path, save_path)
                if info:
                    metadata[file] = info

            # Save aggregation metadata
            if metadata:
                try:
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=4, ensure_ascii=False)
                    self.logger.info(f"Padding complete for {category}. Metadata saved.")
                except Exception as e:
                    self.logger.error(f"Failed to save metadata at {meta_path}: {e}")
            else:
                self.logger.info(f"No padded images generated for {category}.")

def main() -> None:
    """
    Entry point for standalone execution.

    Usage:
        python image_padder.py --config path/to/config.yaml
        python image_padder.py  (uses default: config.yaml)
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Pad images to a fixed square size and save metadata.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (yaml or json). Default: config.yaml",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    # Load config (supports YAML/JSON)
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load .yaml/.yml configs. Install with: pip install pyyaml"
            ) from e
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    elif cfg_path.suffix.lower() == ".json":
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {cfg_path.suffix} (use .yaml/.yml/.json)")

    padder = ImagePadder(config)
    padder.run()


if __name__ == "__main__":
    main()