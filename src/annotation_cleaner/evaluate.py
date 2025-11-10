#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate.py
-------------------
Evaluator 클래스 기반 통합 평가 모듈
- Full Image Metric
- YOLO Crop Metric
- metrics.py 기반 동적 metric 매핑
- config.yaml은 main.py에서 주입받음
"""
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from ultralytics import YOLO
import sys
import tempfile

ROOT_DIR = Path(__file__).resolve().parents[2]  # Research/
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger, setup_logging
from src.annotation_cleaner.metrics.metrics import l1_distance, ssim_score, edge_iou


# ============================================================
# 🧠 Evaluator 클래스
# ============================================================
class Evaluator:
    """
    Full Image 및 YOLO Crop 기반 평가 수행기

    Args:
        orig_dir (str): 원본 이미지 폴더
        gen_dir (str): 생성 이미지 폴더
        metric_dir (str): 평가 결과 저장 폴더
        metrics (List[str]): 사용할 metric 이름 리스트 (e.g., ["ssim", "l1", "edge_iou"])
        yolo_model (str): YOLO weight 파일 경로
        imgsz (int): YOLO 입력 이미지 크기
        categories (List[str], optional): 평가할 클래스 목록
        conf_thres (float, optional): YOLO confidence threshold
    """

    # ------------------------------------------------------------
    # Metric 이름 → 함수 매핑 테이블
    # ------------------------------------------------------------
    METRIC_MAP = {
        "l1": l1_distance,
        "ssim": ssim_score,
        "edge_iou": edge_iou,
    }

    def __init__(
        self,
        orig_dir: str,
        gen_dir: str,
        metric_dir: str,
        metrics: List[str],
        yolo_model: str,
        imgsz: int,
        categories: Optional[List[str]] = None,
        conf_thres: float = 0.25,
    ):
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("Evaluator")

        # --- 기본 설정 ---
        self.orig_dir = Path(orig_dir)
        self.gen_dir = Path(gen_dir)
        self.metric_dir = Path(metric_dir)
        self.metrics = metrics
        self.categories = categories or ["repair", "replace"]

        # --- YOLO 설정 ---
        self.yolo_model = yolo_model
        self.imgsz = imgsz
        self.conf_thres = conf_thres

        # --- 로그 출력 ---
        self.logger.info(f"📂 원본 폴더: {self.orig_dir}")
        self.logger.info(f"💾 생성 폴더: {self.gen_dir}")
        self.logger.info(f"📁 결과 저장 폴더: {self.metric_dir}")
        self.logger.info(f"🧠 YOLO 모델: {self.yolo_model}")
        self.logger.info(f"📏 활성 Metrics: {', '.join(self.metrics)}")

    # ============================================================
    # 📏 Metric 계산
    # ============================================================
    def _compute_metrics(self, orig_img, gen_img) -> Dict[str, float]:
        """
        metrics.py의 함수를 동적으로 호출하여 결과 계산.
        (새로운 metric이 metrics.py에 추가되어도 자동 반영 가능)
        """
        results = {}
        for metric_name in self.metrics:
            func = self.METRIC_MAP.get(metric_name)
            if not func:
                self.logger.warning(f"⚠️ 지원되지 않는 metric: {metric_name} — 스킵")
                continue

            try:
                val = func(orig_img, gen_img)
                key = metric_name.upper() if metric_name != "edge_iou" else "Edge_IoU"
                results[key] = float(val)
            except Exception as e:
                self.logger.error(f"❌ Metric 계산 오류 ({metric_name}): {e}")
        return results

    # ============================================================
    # 🧩 Full Image 평가
    # ============================================================
    def evaluate_full_images(self, save_path: Path) -> Optional[Dict[str, float]]:
        self.logger.info("📊 [1/2] Full Image Evaluation 시작...")
        results = []

        for split in self.categories:
            orig_split = self.orig_dir / split
            gen_split = self.gen_dir / split

            if not orig_split.exists() or not gen_split.exists():
                self.logger.warning(f"⚠️ {split} 폴더가 존재하지 않아 스킵")
                continue

            o_files = {f.stem: f for f in orig_split.glob("*.[jp][pn]g")}
            g_files = {f.stem: f for f in gen_split.glob("*.[jp][pn]g")}
            common = set(o_files.keys()) & set(g_files.keys())

            for name in tqdm(common, desc=f"{split}"):
                o_img = cv2.imread(str(o_files[name]))
                g_img = cv2.imread(str(g_files[name]))
                if o_img is None or g_img is None:
                    continue
                if o_img.shape != g_img.shape:
                    g_img = cv2.resize(g_img, (o_img.shape[1], o_img.shape[0]))

                metric_vals = self._compute_metrics(o_img, g_img)
                results.append({"split": split, "file": name, **metric_vals})

        if not results:
            self.logger.warning("❌ 평가할 이미지가 없습니다.")
            return None

        df = pd.DataFrame(results)
        avg = df.drop(columns=["split", "file"]).mean().to_dict()

        avg_row = {**{k: "" for k in df.columns}, **avg}
        avg_row["split"] = "AVG"
        avg_row["file"] = "AVG" 
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        self.metric_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        self.logger.info(f"📁 Full Image 결과 저장 → {save_path}")
        return avg

    # ============================================================
    # 🧩 YOLO Crop 평가 (Temp 디렉토리 기반)
    # ============================================================
    def evaluate_with_yolo_crop(self, save_path: Path) -> Optional[Dict[str, float]]:
        self.logger.info("📊 [2/2] YOLO Crop Evaluation 시작...")
        yolo = YOLO(self.yolo_model)
        results = []

        # ✅ 임시 작업 폴더 생성
        with tempfile.TemporaryDirectory(prefix="eval_yolo_") as temp_root:
            temp_root = Path(temp_root)
            crop_dir = temp_root / "crops"
            bbox_dir = temp_root / "bboxes"
            crop_dir.mkdir(parents=True, exist_ok=True)
            bbox_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"🧩 임시 폴더 생성: {temp_root}")

            image_list = [img for c in self.categories for img in (self.gen_dir / c).glob("*.[jp][pn]g")]

            for img_path in tqdm(image_list, desc="YOLO inference"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                preds = yolo.predict(source=str(img_path), imgsz=self.imgsz, conf=self.conf_thres, save=False, verbose=False)
                if not preds or not preds[0].boxes.xyxy.numel():
                    continue

                split_name = img_path.parent.name
                base_name = img_path.stem
                bbox_txt = bbox_dir / split_name / f"{base_name}.txt"
                crop_split_dir = crop_dir / split_name
                bbox_txt.parent.mkdir(parents=True, exist_ok=True)
                crop_split_dir.mkdir(parents=True, exist_ok=True)

                # ✅ bounding box 정보 기록 (필요시 로깅용)
                with open(bbox_txt, "w") as f:
                    for idx, box in enumerate(preds[0].boxes.xyxy):
                        x1, y1, x2, y2 = map(int, box)
                        f.write(f"abs {x1} {y1} {x2} {y2}\n")

                        # crop 이미지는 메모리에 저장 후 평가용으로만 사용
                        crop = img[y1:y2, x1:x2]
                        if crop.size > 0:
                            cv2.imwrite(str(crop_split_dir / f"{base_name}_crop{idx}.jpg"), crop)

                orig_path = self.orig_dir / split_name / f"{base_name}.jpg"
                if not orig_path.exists():
                    continue

                o_img = cv2.imread(str(orig_path))
                if o_img is None:
                    continue

                # ✅ 각 crop에 대해 metric 계산
                for idx, box in enumerate(preds[0].boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    c1, c2 = o_img[y1:y2, x1:x2], img[y1:y2, x1:x2]
                    if c1.size == 0 or c2.size == 0:
                        continue
                    if c1.shape != c2.shape:
                        c2 = cv2.resize(c2, (c1.shape[1], c1.shape[0]))

                    metric_vals = self._compute_metrics(c1, c2)
                    results.append({
                        "split": split_name,
                        "file": base_name,
                        "crop_idx": idx,
                        **metric_vals
                    })

            # ✅ 평가 완료 후 자동 정리
            self.logger.info(f"🧹 YOLO Crop 임시 데이터 삭제: {temp_root}")
        
        self.metric_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ 결과 집계
        if not results:
            self.logger.warning("❌ YOLO Crop 평가 결과 없음.")
            return None

        df = pd.DataFrame(results)
        avg = df.drop(columns=["split", "file", "crop_idx"]).mean().to_dict()

        avg_row = {**{k: "" for k in df.columns}, **avg}
        avg_row["split"] = "AVG"
        avg_row["file"] = "AVG"
        avg_row["crop_idx"] = "AVG"
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        df.to_csv(save_path, index=False)
        self.logger.info(f"📁 YOLO Crop 결과 저장 → {save_path}")
        return avg


    # ============================================================
    # 🚀 전체 실행
    # ============================================================
    def run(self) -> Dict[str, Optional[Dict[str, float]]]:
        """Full + YOLO Crop 평가 전체 수행"""
        full_path = self.metric_dir / "metrics_full_image.csv"
        crop_path = self.metric_dir / "metrics_yolo_crop.csv"

        avg_full = self.evaluate_full_images(full_path)
        avg_crop = self.evaluate_with_yolo_crop(crop_path)

        self.logger.info("\n=== ✅ 최종 평균 결과 ===")
        self.logger.info(f"Full Image: {avg_full}")
        self.logger.info(f"YOLO Crop:  {avg_crop}")

        return {"full": avg_full, "crop": avg_crop}
