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
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

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



def main():
    """
    Entry point for standalone execution.

    Examples:
      python restore_cropper.py --config config.yaml
      python restore_cropper.py --config config.yaml --override_input data/...
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Restore padded images to original dimensions using padding metadata."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )

    # (선택) 실행 시 경로를 오버라이드하고 싶을 때
    parser.add_argument("--override_input", type=str, default=None, help="Override input_dir")
    parser.add_argument("--override_output", type=str, default=None, help="Override output_dir")
    parser.add_argument("--override_meta", type=str, default=None, help="Override metadata_root")

    args = parser.parse_args()

    # config 로더: 프로젝트에 이미 있는 로더를 우선 사용
    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = None

    # 1) (권장) PyYAML이 설치되어 있으면 yaml 로드
    try:
        import yaml  # type: ignore

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    except Exception:
        # 2) 프로젝트에 전용 config loader가 있다면 여기서 가져와서 사용하세요.
        #    (예: from utils.config import load_config)
        raise RuntimeError(
            "Failed to load YAML config. Install PyYAML (pip install pyyaml) "
            "or plug in your project's config loader."
        )

    # override 적용(필요 시)
    if args.override_input is not None:
        cfg.setdefault("annotation_cleaner", {}).setdefault("restore_crop", {})["input_dir"] = args.override_input
    if args.override_output is not None:
        cfg.setdefault("annotation_cleaner", {}).setdefault("restore_crop", {})["output_dir"] = args.override_output
    if args.override_meta is not None:
        cfg.setdefault("annotation_cleaner", {}).setdefault("restore_crop", {})["metadata_root"] = args.override_meta

    cropper = RestoreCropper(cfg)
    cropper.run()


if __name__ == "__main__":
    main()