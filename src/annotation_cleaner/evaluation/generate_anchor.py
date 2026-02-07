#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict
from tqdm import tqdm

from src.annotation_cleaner.evaluation.crop_file import crop_dataset
from utils.logging import get_logger      

logger = get_logger("AnchorGen")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

class AnchorGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eval_conf = config["annotation_cleaner"]["evaluate"]
        self.dirs = self.eval_conf["dirs"]
        self.params = self.eval_conf.get("params", {})
        
        # --- Paths ---
        self.label_dir = Path(self.dirs["label"])
        self.original_full_dir = Path(self.dirs["original"])
        
        # GAC & Marking Full Paths
        self.gac_full_dir = Path(self.dirs.get("gac_full", "data/experiments/gac_eval/gac"))
        self.marked_full_dir = Path(self.dirs.get("marked_full", "data/experiments/gac_eval/marked"))
        
        # Other Crack (Mix Anchor source)
        self.other_crack_dir = Path(self.dirs.get("other_crack", "data/experiments/gac_eval/other_crack"))
        
        # Output Paths (Crops)
        self.baseline_crop_dir = Path(self.dirs["baseline_crop"])
        self.anchor_root = Path(self.dirs["anchor_root"])
        
        # Seed Setting
        seed = self.params.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)

    def run(self):
        logger.info("=== Starting Evaluation Data Preparation ===")

        # Step 1: Generate Marked Full Images (Original + Label -> Marked Full)
        self._generate_full_marking()

        # Step 2: Crop Everything (Full -> Crop)
        self._run_all_crops()

        # Step 3: Generate Synthetic Anchors (Mix, Blur, JPEG from Baseline Crop)
        self._generate_synthetic_anchors()

        logger.info("=== All Data Preparation Finished ===")

    # Full Image Marking
    def _generate_full_marking(self):
        logger.info("[Step 1] Generating Marked Full Images...")
        self.marked_full_dir.mkdir(parents=True, exist_ok=True)
        
        label_files = sorted(list(self.label_dir.glob("*.txt")))
        mark_params = self.params.get("marking", {})
        
        for lbl_path in tqdm(label_files, desc="Marking Full"):
            stem = lbl_path.stem
            # Find original image
            img_path = None
            for ext in VALID_EXTS:
                if (self.original_full_dir / (stem + ext)).exists():
                    img_path = self.original_full_dir / (stem + ext)
                    break
            
            if not img_path: continue
            
            img = cv2.imread(str(img_path))
            if img is None: continue
            h, w = img.shape[:2]

            # Parse Label to find where to draw
            bboxes = self._parse_yolo_for_marking(lbl_path, w, h)
            if not bboxes: continue

            # Draw Marking
            marked_img = self._draw_synthetic_marking(img, bboxes, mark_params)
            cv2.imwrite(str(self.marked_full_dir / img_path.name), marked_img)

    def _parse_yolo_for_marking(self, txt_path, img_w, img_h):
        bboxes = []
        try:
            with open(txt_path, 'r') as f: lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                coords = list(map(float, parts[1:]))
                if len(coords) == 4: # BBox
                    cx, cy, bw, bh = coords
                    x1 = int((cx - bw/2) * img_w)
                    y1 = int((cy - bh/2) * img_h)
                    x2 = int((cx + bw/2) * img_w)
                    y2 = int((cy + bh/2) * img_h)
                    bboxes.append((x1, y1, x2, y2))
                elif len(coords) > 4: # Polygon -> BBox
                    xs = coords[0::2]
                    ys = coords[1::2]
                    bboxes.append((int(min(xs)*img_w), int(min(ys)*img_h), int(max(xs)*img_w), int(max(ys)*img_h)))
        except: pass
        return bboxes

    def _draw_synthetic_marking(self, img, bboxes, params):
        marked = img.copy()
        color = (255, 255, 255)
        t_min = params.get("thickness_min", 4)
        t_max = params.get("thickness_max", 6)
        
        for (x1, y1, x2, y2) in bboxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            if x2 <= x1 or y2 <= y1: continue

            thickness = random.randint(t_min, t_max)
            start_pt = (random.randint(x1, x1 + (x2-x1)//3), random.randint(y1, y2))
            end_pt = (random.randint(x1 + 2*(x2-x1)//3, x2), random.randint(y1, y2))
            control_pt = (random.randint(x1, x2), random.randint(y1, y2))
            
            pts = self._get_bezier_curve([start_pt, control_pt, end_pt], 50)
            cv2.polylines(marked, [pts], False, color, thickness, cv2.LINE_AA)
        return marked

    @staticmethod
    def _get_bezier_curve(points, num_points=100):
        points = np.array(points)
        n = len(points) - 1
        t = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 2))
        for i in range(n + 1):
            coef = math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
            bernstein = coef * (t ** i) * ((1 - t) ** (n - i))
            curve += np.outer(bernstein, points[i])
        return curve.astype(np.int32)

    # Cropping
    def _run_all_crops(self):
        logger.info("[Step 2] Cropping Images...")
        
        # 1. Original -> Baseline
        crop_dataset(self.label_dir, self.original_full_dir, self.baseline_crop_dir)
        
        # 2. Marked Full -> Anchors/Marked
        crop_dataset(self.label_dir, self.marked_full_dir, self.anchor_root / "marked")
        
        # 3. GAC Full -> Anchors/GAC
        if self.gac_full_dir.exists():
            crop_dataset(self.label_dir, self.gac_full_dir, self.anchor_root / "gac")
        else:
            logger.warning(f"GAC Full dir not found ({self.gac_full_dir}). Skipping GAC crop.")

    # Synthetic Anchors (Mix, Blur, JPEG)
    def _generate_synthetic_anchors(self):
        logger.info("[Step 3] Generating Synthetic Anchors from Baseline...")
        
        if not list(self.baseline_crop_dir.glob("*")):
             logger.warning(f"No baseline crops found in {self.baseline_crop_dir}. Skipping synthetic generation.")
             return

        self._generate_mix_anchors()    # Output -> self.anchor_root / "mix"
        self._generate_benign_anchors() # Output -> self.anchor_root / "blur", "jpeg"
    
    # --- Mix Anchor Logic (Fixed) ---
    def _generate_mix_anchors(self):
        mix_conf = self.params.get("mix", {})
        out_dir = self.anchor_root / "mix"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        files = sorted([p for p in self.baseline_crop_dir.glob("*") if p.suffix.lower() in VALID_EXTS])
        
        other_crack_files = []
        if self.other_crack_dir.exists():
            other_crack_files = sorted([p for p in self.other_crack_dir.glob("*") if p.suffix.lower() in VALID_EXTS])
            logger.info(f"[Mix] Using {len(other_crack_files)} other crack source images.")
        
        # Fallback
        if not other_crack_files:
            logger.warning("[Mix] No other sources found. Fallback to intra-baseline mixing.")
            other_crack_files = files
            if len(files) < 2:
                return

        # 3. Loop & Generate
        logger.info(f"[Mix] Generating {len(files)} mixed anchors...")
        for img_a_path in tqdm(files, desc="Gen Mix"):
            candidates = [p for p in other_crack_files if p.name != img_a_path.name]
            if not candidates: candidates = other_crack_files
            img_b_path = random.choice(candidates)
            
            self._create_mix_crack(img_a_path, img_b_path, out_dir / img_a_path.name, mix_conf)

    def _create_mix_crack(self, path_a, path_b, out_path, conf):
        img_a = cv2.imread(str(path_a))
        img_b = cv2.imread(str(path_b))
        if img_a is None or img_b is None: return

        h, w = img_a.shape[:2]
        img_b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_LINEAR)

        # Extract Mask
        mask = self._build_crack_mask(
            img_b, 
            conf.get("canny1", 30), 
            conf.get("canny2", 100),
            conf.get("dilate_iter", 4),
            conf.get("close_ksize", 7),
            conf.get("min_area", 200)
        )
        if np.count_nonzero(mask) == 0: return

        # Center Align
        img_b_shift, mask_shift = self._shift_to_center(img_b, mask, (w//2, h//2))
        if img_b_shift is None: return

        # Seamless Clone
        try:
            flag = cv2.MIXED_CLONE if conf.get("use_mixed_clone", False) else cv2.NORMAL_CLONE
            center = (w//2, h//2)
            blended = cv2.seamlessClone(img_b_shift, img_a, mask_shift, center, flag)
            
            # Alpha Boost
            alpha = conf.get("alpha_boost", 0.0)
            if alpha > 0:
                soft = cv2.GaussianBlur(mask_shift, (0, 0), sigmaX=3.0).astype(np.float32) / 255.0
                soft = soft[..., None]
                out = blended.astype(np.float32)
                src = img_b_shift.astype(np.float32)
                out = out * (1 - alpha * soft) + src * (alpha * soft)
                blended = np.clip(out, 0, 255).astype(np.uint8)

            cv2.imwrite(str(out_path), blended)
        except Exception as e:
            logger.debug(f"Mix failed for {path_a.name}: {e}")

    def _build_crack_mask(self, img, c1, c2, d_iter, ksize, min_area):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, c1, c2)
        if d_iter > 0:
            edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=d_iter)
        if ksize > 1:
            k = np.ones((ksize, ksize), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
        mask = (edges > 0).astype(np.uint8) * 255
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num > 1:
            clean = np.zeros_like(mask)
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    clean[labels == i] = 255
            return clean
        return mask

    def _shift_to_center(self, img, mask, target_center):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0: return None, None
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        tx, ty = target_center
        M = np.float32([[1, 0, tx - cx], [0, 1, ty - cy]])
        h, w = img.shape[:2]
        img_s = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        mask_s = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img_s, mask_s

    # --- Benign Anchors ---
    def _generate_benign_anchors(self):
        conf = self.params.get("benign", {})
        blur_sigma = conf.get("blur_sigma", 1.0)
        jpeg_q = conf.get("jpeg_quality", 50)
        
        blur_dir = self.anchor_root / "blur"
        jpeg_dir = self.anchor_root / "jpeg"
        blur_dir.mkdir(parents=True, exist_ok=True)
        jpeg_dir.mkdir(parents=True, exist_ok=True)
        
        files = sorted([p for p in self.baseline_crop_dir.glob("*") if p.suffix.lower() in VALID_EXTS])
        logger.info(f"[Benign] Generating Blur/JPEG for {len(files)} images...")
        
        for p in tqdm(files, desc="Gen Benign"):
            img = cv2.imread(str(p))
            if img is None: continue
            
            # Blur
            img_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=float(blur_sigma))
            cv2.imwrite(str(blur_dir / p.name), img_blur)
            
            # JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)]
            _, buf = cv2.imencode(".jpg", img, encode_param)
            img_jpeg = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            cv2.imwrite(str(jpeg_dir / p.name), img_jpeg)

def run_anchor_generation(config):
    AnchorGenerator(config).run()