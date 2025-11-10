#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predictor.py
------------
YOLOv8 Predictor (Config-driven)
- config.yaml 기반으로 YOLOv8 추론 수행
- outputs/json_results/{model_name}/result/ 에 결과 저장
- outputs/json_results/predict.txt 를 입력으로 사용
"""

from pathlib import Path
from ultralytics import YOLO
from typing import Dict, Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger, setup_logging
from utils.load_config import load_yaml_config


class YOLOv8Predictor:
    """Handles YOLOv8 inference using Ultralytics YOLO library (config-driven)."""

    def __init__(self, config: Dict[str, Any]):

        # --------------------------------------------------------
        # Parse config
        # --------------------------------------------------------
        self.logger = get_logger("yolo_cropper.YOLOv8Predictor")
        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.yolov8_cfg = self.yolo_cropper_cfg.get("yolov8", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.tain_cfg = self.yolo_cropper_cfg.get("train", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})
        self.model_name = self.main_cfg.get("model_name", "yolov8s")

        # --------------------------------------------------------
        # Directories
        # --------------------------------------------------------
        saved_model_dir = Path(self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")).resolve()
        self.weights_path = (saved_model_dir / f"{self.model_name}.pt").resolve()
        self.detect_root = Path(self.dataset_cfg.get("detect_output_dir", "runs/detect")).resolve()
        self.output_root = Path(self.dataset_cfg.get("results_dir", "outputs/json_results"))
        self.output_dir = (self.output_root / self.model_name).resolve()


        self.imgsz = self.tain_cfg.get("imgsz", 416)
        self.save_crop = self.tain_cfg.get("save_crop", False)
        self.save_txt = self.tain_cfg.get("save_txt", True)
        self.save_conf = self.tain_cfg.get("save_conf", True)
        self.quiet = self.yolov8_cfg.get("quiet", True)
        self.predict_txt = (self.output_root / "predict.txt").resolve()
        

        self.logger.info(f"Initialized YOLOv8Predictor ({self.model_name.upper()})")
        self.logger.debug(f" - Weights : {self.weights_path}")
        self.logger.debug(f" - Predict : {self.predict_txt}")
        self.logger.debug(f" - Output  : {self.output_dir}")

    # --------------------------------------------------------
    # 🔹 Main Prediction
    # --------------------------------------------------------
    def run(self):
        """Run YOLOv8 detection using predict.txt"""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"❌ Model weight not found: {self.weights_path}")
        if not self.predict_txt.exists():
            raise FileNotFoundError(f"❌ predict.txt not found: {self.predict_txt}")

        # 모델 로드
        model = YOLO(self.weights_path)
        self.logger.info(f"🚀 Starting YOLOv8 detection ({self.model_name.upper()})")

        # 예측 실행
        results = model.predict(
            source=str(self.predict_txt),
            imgsz=self.imgsz,
            save=True,
            save_crop=self.save_crop,
            save_txt=self.save_txt,
            save_conf=self.save_conf,
            project=str(self.detect_root),
            name=self.model_name,
            exist_ok=True,
            verbose=not self.quiet,
        )

        result_dir = Path(results[0].save_dir).resolve()
        self.logger.info(f"[✓] Detection complete → {result_dir}")

        return str(result_dir), str(self.predict_txt)