#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict.py
----------
Darknet-based YOLOv2 / YOLOv4 inference module.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger
from utils.model_hub import download_fine_tuned_weights


class DarknetPredictor:
    """
    Runs object detection inference using Darknet (YOLOv2 / YOLOv4).
    """

    def __init__(self, config: Dict[str, Any]):
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

        # Paths
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

        # Data parameters
        self.conf_thresh = self.train_cfg.get("conf_thresh", 0.25)

        self.logger.info(
            f"Initialized DarknetPredictor ({self.model_name.upper()})"
        )
        self.logger.debug(f" - Darknet dir : {self.darknet_dir}")
        self.logger.debug(f" - Weights    : {self.weights_path}")
        self.logger.debug(f" - Input dir  : {self.input_dir}")

    # --------------------------------------------------------
    def _list_images(self, input_dir: Path) -> list[str]:
        exts = [".jpg", ".jpeg", ".png", ".bmp"]
        images = [
            str(p.resolve())
            for p in input_dir.rglob("*")
            if p.suffix.lower() in exts
        ]

        if not images:
            raise FileNotFoundError(
                f"No images found recursively under {input_dir}"
            )

        self.logger.info(f"Found {len(images)} images under {input_dir}")
        return sorted(images)
    
    # --------------------------------------------------------
    def _ensure_obj_data(self):
        obj_data_path = self.data_dir / "obj.data"
        obj_names_path = self.data_dir / "obj.names"

        categories = self.global_main_cfg.get("categories", [])
        if not categories:
            raise ValueError("main.categories is empty in config.yaml")

        # Ensure backup directory exists
        backup_dir = self.darknet_dir / "backup"
        backup_dir.mkdir(exist_ok=True)

        # obj.names
        obj_names_path.write_text(
            "\n".join(categories) + "\n", encoding="utf-8"
        )

        # obj.data (minimal, inference-only)
        content = [
            f"classes = {len(categories)}",
            "names = data/obj.names",
            "backup = backup/",
        ]
        obj_data_path.write_text("\n".join(content) + "\n", encoding="utf-8")

        self.logger.info(
            "Generated minimal data/obj.data, obj.names, and ensured backup/ for inference"
        )


    # --------------------------------------------------------
    def run(self):
        """
        Run Darknet-based YOLO inference.
        """
        self._ensure_obj_data()

        if self.demo_mode:
            self.logger.info(
                "Demo mode → Download fine-tuned Darknet YOLO weights"
            )
            download_fine_tuned_weights(
                cfg=self.cfg,
                model_name=self.model_name,
                saved_model_path=self.weights_path,
                logger=self.logger,
            )

        images = self._list_images(self.input_dir)
        predict_path = self.data_dir / "predict.txt"
        predict_path.write_text("\n".join(images) + "\n", encoding="utf-8")

        self.logger.info(
            f"Generated predict.txt ({len(images)} images) → {predict_path}"
        )

        # Darknet command setup
        cfg_path = f"cfg/{self.model_name}-obj.cfg"
        obj_data = "data/obj.data"

        internal_result = (
            self.data_dir / f"result_{self.model_name}.json"
        ).resolve()

        external_result = self.output_dir / "result.json"

        command = (
            f"./darknet detector test {obj_data} {cfg_path} {self.weights_path} "
            f"-thresh {self.conf_thresh} -dont_show -ext_output "
            f"-out {internal_result} < data/predict.txt"
        )

        self.logger.info(
            f"Starting Darknet detection ({self.model_name.upper()})"
        )
        self.logger.debug(f"[CMD] {command}")

        process = subprocess.run(
            ["bash", "-lc", command],
            cwd=self.darknet_dir,
            shell=False,
        )

        if process.returncode not in (0, 1):
            raise RuntimeError(
                f"Darknet detection failed (code: {process.returncode})"
            )
        elif process.returncode == 1:
            self.logger.warning(
                "Darknet exited with code 1 (non-fatal). Detection likely succeeded."
            )

        if internal_result.exists():
            shutil.copy2(internal_result, external_result)
            self.logger.info(f"Copied result → {external_result}")
        else:
            self.logger.warning(
                "result.json not found in darknet/data folder!"
            )

        predict_copy = self.output_dir / "predict.txt"
        shutil.copy2(predict_path, predict_copy)
        self.logger.info(
            f"Copied predict.txt → {predict_copy.resolve()}"
        )

        return str(external_result), str(predict_path)
