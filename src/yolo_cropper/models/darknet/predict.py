#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Darknet Inference Module.

Manages batch object detection using Darknet (YOLOv2/v4), handling
input list generation, configuration file creation, and result parsing.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger
from utils.model_hub import download_fine_tuned_weights


class DarknetPredictor:
    """
    Orchestrates Darknet execution for batch image inference.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = get_logger("yolo_cropper.DarknetPredictor")
        self.cfg = config

        # Configuration
        self.global_main_cfg = self.cfg.get("main", {})
        self.demo_mode = self.global_main_cfg.get("demo", False)

        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.train_cfg = self.yolo_cropper_cfg.get("train", {})
        self.darknet_cfg = self.yolo_cropper_cfg.get("darknet", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        self.model_name = self.main_cfg.get("model_name", "yolov2").lower()

        # Path setup
        self.darknet_dir = Path(
            self.darknet_cfg.get("darknet_dir", "third_party/darknet")
        ).resolve()

        self.data_dir = self.darknet_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        self.input_dir = Path(
            self.main_cfg.get("input_dir", "data/original")
        ).resolve()

        self.results_root = Path(
            self.dataset_cfg.get("results_dir", "outputs/json_results")
        ).resolve()

        self.output_dir = (self.results_root / self.model_name).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.saved_model_dir = Path(
            self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")
        ).resolve()

        self.weights_path = (
            self.saved_model_dir / f"{self.model_name}.weights"
        ).resolve()

        self.conf_thresh = self.train_cfg.get("conf_thresh", 0.25)

        self.logger.info(f"Initialized Predictor (Model: {self.model_name.upper()})")

    def _list_images(self, input_dir: Path) -> List[str]:
        """
        Recursively retrieves image paths from the input directory.

        Returns:
            List[str]: A sorted list of absolute image paths.
        """
        exts = [".jpg", ".jpeg", ".png", ".bmp"]
        images = [
            str(p.resolve())
            for p in input_dir.rglob("*")
            if p.suffix.lower() in exts
        ]

        if not images:
            raise FileNotFoundError(f"No images found in {input_dir}")

        self.logger.info(f"Found {len(images)} images")
        return sorted(images)

    def _ensure_obj_data(self) -> None:
        """
        Generates temporary configuration files (`obj.data`, `obj.names`) required by Darknet.
        """
        obj_data_path = self.data_dir / "obj.data"
        obj_names_path = self.data_dir / "obj.names"

        categories = self.global_main_cfg.get("categories", [])
        if not categories:
            raise ValueError("Categories not defined in configuration.")

        # Ensure backup directory exists
        backup_dir = self.darknet_dir / "backup"
        backup_dir.mkdir(exist_ok=True)

        # Generate obj.names
        obj_names_path.write_text(
            "\n".join(categories) + "\n", encoding="utf-8"
        )

        # Generate obj.data
        content = [
            f"classes = {len(categories)}",
            "names = data/obj.names",
            "backup = backup/",
        ]
        obj_data_path.write_text("\n".join(content) + "\n", encoding="utf-8")

        self.logger.info("Generated Darknet config files (obj.data, obj.names)")

    def run(self) -> Tuple[str, str]:
        """
        Executes the inference pipeline via Darknet subprocess.

        Returns:
            Tuple[str, str]: Paths to the result JSON and the prediction image list.
        """
        self._ensure_obj_data()

        if self.demo_mode:
            self.logger.info("Demo mode: Downloading fine-tuned weights")
            download_fine_tuned_weights(
                cfg=self.cfg,
                model_name=self.model_name,
                saved_model_path=self.weights_path,
                logger=self.logger,
            )

        images = self._list_images(self.input_dir)
        predict_path = self.data_dir / "predict.txt"
        predict_path.write_text("\n".join(images) + "\n", encoding="utf-8")

        self.logger.info(f"Generated prediction list: {predict_path}")

        # Darknet CLI setup
        cfg_path = f"cfg/{self.model_name}-obj.cfg"
        obj_data = "data/obj.data"

        internal_result = (
            self.data_dir / f"result_{self.model_name}.json"
        ).resolve()

        external_result = self.output_dir / "result.json"

        # Construct command
        command = (
            f"./darknet detector test {obj_data} {cfg_path} {self.weights_path} "
            f"-thresh {self.conf_thresh} -dont_show -ext_output "
            f"-out {internal_result} < data/predict.txt"
        )

        self.logger.info(f"Starting Inference ({self.model_name.upper()})")

        process = subprocess.run(
            ["bash", "-lc", command],
            cwd=self.darknet_dir,
            shell=False,
        )

        # Darknet often returns 1 even on success if image window can't open
        if process.returncode not in (0, 1):
            raise RuntimeError(f"Darknet detection failed (Code: {process.returncode})")
        elif process.returncode == 1:
            self.logger.warning("Darknet exited with code 1 (Non-fatal)")

        if internal_result.exists():
            shutil.copy2(internal_result, external_result)
            self.logger.info(f"Result saved to {external_result}")
        else:
            self.logger.warning("Result JSON not generated by Darknet.")

        predict_copy = self.output_dir / "predict.txt"
        shutil.copy2(predict_path, predict_copy)

        return str(external_result), str(predict_path)