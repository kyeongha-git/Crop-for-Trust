#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict.py
----------
This module performs YOLOv2 or YOLOv4 inference using the Darknet binary.

It automatically gathers image paths, executes Darknet detection, and
exports the resulting predictions (in JSON format) to an organized output
directory. All parameters are provided through a configuration object.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class DarknetPredictor:
    """
    Runs object detection inference using Darknet for YOLOv2 or YOLOv4.

    This class manages the entire prediction process — gathering images,
    executing the Darknet command, saving prediction results, and
    logging outputs for review.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Darknet predictor with a configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration object injected by the main
                controller, containing Darknet, dataset, and model parameters.
        """
        self.logger = get_logger("yolo_cropper.DarknetPredictor")

        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.darknet_cfg = self.yolo_cropper_cfg.get("darknet", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        self.darknet_dir = Path(
            self.darknet_cfg.get("darknet_dir", "third_party/darknet")
        ).resolve()
        self.data_dir = self.darknet_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.input_dir = Path(self.main_cfg.get("input_dir", "data/original"))
        self.output_dir = Path(
            self.dataset_cfg.get("results_dir", "outputs/json_results")
        )
        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        )
        self.model_name = self.main_cfg.get("model_name", "yolov2").lower()

        self.logger.info(f"Initialized DarknetPredictor for {self.model_name.upper()}")
        self.logger.debug(f"Darknet dir : {self.darknet_dir}")
        self.logger.debug(f"Data dir    : {self.data_dir}")

    def _list_images(self, input_dir: Path):
        """
        Recursively list all image files in the input directory.

        This includes images in nested folders (e.g., repair/replace or other
        subcategories).

        Returns:
            list[str]: A sorted list of absolute image paths.
        """
        exts = [".jpg", ".jpeg", ".png", ".bmp"]
        image_paths = [
            str(p.resolve()) for p in input_dir.rglob("*") if p.suffix.lower() in exts
        ]

        if not image_paths:
            raise FileNotFoundError(f"No images found recursively under {input_dir}")

        self.logger.info(f"Found {len(image_paths)} images under {input_dir}")
        return sorted(image_paths)

    def run(self):
        """
        Run YOLO object detection using the Darknet executable.

        This method:
            1. Collects image paths recursively from the input directory.
            2. Generates a `predict.txt` list for Darknet.
            3. Executes the `detector test` command with the model’s weights.
            4. Saves the output JSON to both internal and external directories.
            5. Copies the prediction list for recordkeeping.

        Returns:
            tuple[str, str]: Paths to the JSON result file and the predict.txt file.

        """
        input_dir = self.input_dir
        predict_path = self.data_dir / "predict.txt"

        # Generate image list file
        images = self._list_images(input_dir)
        predict_path.write_text("\n".join(images) + "\n", encoding="utf-8")
        self.logger.info(
            f"Generated predict.txt ({len(images)} images) → {predict_path}"
        )

        # Configure paths
        cfg_path = f"cfg/{self.model_name}-obj.cfg"
        obj_data = "data/obj.data"
        weights_path = Path(
            f"{self.saved_model_dir}/{self.model_name}.weights"
        ).resolve()
        internal_result = (self.data_dir / f"result_{self.model_name}.json").resolve()
        external_result = Path(f"{self.output_dir}/{self.model_name}/result.json")
        external_result.parent.mkdir(parents=True, exist_ok=True)

        # Build Darknet detection command
        command = (
            f"./darknet detector test {obj_data} {cfg_path} {weights_path} "
            f"-thresh 0.25 -dont_show -ext_output -out {internal_result} < data/{predict_path.name}"
        )

        self.logger.info(f"Starting YOLO detection ({self.model_name.upper()})")
        self.logger.debug(f"[CMD] {command}")

        process = subprocess.run(
            ["bash", "-lc", command], cwd=self.darknet_dir, shell=False
        )

        # Handle Darknet return code
        if process.returncode not in (0, 1):
            raise RuntimeError(
                f"Darknet detection failed (code: {process.returncode})"
            )
        elif process.returncode == 1:
            self.logger.warning(
                "Darknet exited with code 1 (non-fatal). Detection likely succeeded."
            )

        # Copy JSON results to external output directory
        if internal_result.exists():
            shutil.copy2(internal_result, external_result)
            self.logger.info(f"Copied result → {external_result.resolve()}")
        else:
            self.logger.warning("result.json not found in darknet/data folder!")

        # Copy predict.txt to output directory for reference
        predict_copy_path = self.output_dir / "predict.txt"
        try:
            shutil.copy2(predict_path, predict_copy_path)
            self.logger.info(f"Copied predict.txt → {predict_copy_path.resolve()}")
        except Exception as e:
            self.logger.warning(f"Failed to copy predict.txt → {e}")

        return str(external_result.resolve()), str(predict_path)
