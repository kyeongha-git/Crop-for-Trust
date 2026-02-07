#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import random
from glob import glob
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# Common settings
# ----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ============================================================
# Part A) Mix-crack anchor (SeamlessClone with Heavy Source)
# ============================================================
def build_crack_mask_from_source(
    img_b_bgr: np.ndarray,
    canny1: int = 30,    
    canny2: int = 100,   
    dilate_iter: int = 4,
    close_ksize: int = 7,
    min_area: int = 200,
):
    gray = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny1, canny2)

    if dilate_iter > 0:
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=dilate_iter)

    if close_ksize and close_ksize > 1:
        k = np.ones((close_ksize, close_ksize), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    mask = (edges > 0).astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num > 1:
        clean = np.zeros_like(mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == i] = 255
        mask = clean

    return mask


def compute_mask_centroid(mask_1ch: np.ndarray):
    ys, xs = np.where(mask_1ch > 0)
    if len(xs) == 0:
        return None
    return (int(np.mean(xs)), int(np.mean(ys)))


def shift_image_and_mask_to_center(img_b, mask, target_center):
    h, w = mask.shape[:2]
    c = compute_mask_centroid(mask)
    if c is None:
        return None, None

    cx, cy = c
    tx, ty = target_center
    dx = tx - cx
    dy = ty - cy

    M = np.float32([[1, 0, dx], [0, 1, dy]])

    img_b_shift = cv2.warpAffine(
        img_b,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    mask_shift = cv2.warpAffine(
        mask,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return img_b_shift, mask_shift


def create_mix_crack_anchor(
    img_path_a,
    img_path_b,
    output_path,
    canny1=30,
    canny2=100,
    dilate_iter=4,
    close_ksize=7,
    min_area=200,
    use_mixed_clone=False,
    alpha_boost=0.0,
):
    img_a = cv2.imread(str(img_path_a), cv2.IMREAD_COLOR)
    img_b = cv2.imread(str(img_path_b), cv2.IMREAD_COLOR)

    if img_a is None or img_b is None:
        return False

    h, w = img_a.shape[:2]
    img_b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_LINEAR)

    # 1. Mask extraction (Heavy Source Image)
    mask = build_crack_mask_from_source(
        img_b,
        canny1=canny1,
        canny2=canny2,
        dilate_iter=dilate_iter,
        close_ksize=close_ksize,
        min_area=min_area,
    )
    if np.count_nonzero(mask) == 0:
        # print(f"[WARN] Empty crack mask from heavy source: {img_path_b.name}")
        return False

    center = (w // 2, h // 2)
    img_b_shift, mask_shift = shift_image_and_mask_to_center(img_b, mask, center)
    
    if img_b_shift is None or mask_shift is None or np.count_nonzero(mask_shift) == 0:
        return False

    try:
        flag = cv2.MIXED_CLONE if use_mixed_clone else cv2.NORMAL_CLONE
        
        blended = cv2.seamlessClone(img_b_shift, img_a, mask_shift, center, flag)

        if alpha_boost and alpha_boost > 0:
            a = float(alpha_boost)
            soft = cv2.GaussianBlur(mask_shift, (0, 0), sigmaX=3.0).astype(np.float32) / 255.0
            soft = soft[..., None]
            
            out = blended.astype(np.float32)
            src = img_b_shift.astype(np.float32)
            out = out * (1 - a * soft) + src * (a * soft)
            blended = np.clip(out, 0, 255).astype(np.uint8)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return cv2.imwrite(str(output_path), blended)

    except Exception as e:
        print(f"[ERROR] Blending failed: {output_path} ({e})")
        return False


def generate_mix_crack_anchors(
    baseline_dir,
    output_dir,
    heavy_source_dir=None, # Heavy check source directory
    seed=42,
    canny1=30,
    canny2=100,
    dilate_iter=4,
    close_ksize=7,
    min_area=200,
    alpha_boost=0.0,
    use_mixed_clone=False, # Normal Clone
):
    baseline_dir = Path(baseline_dir)
    output_dir = Path(output_dir)

    baseline_imgs = sorted([p for p in baseline_dir.glob("*") if p.suffix.lower() in IMG_EXTS])

    # [New] Heavy Source Loading
    heavy_imgs = []
    if heavy_source_dir:
        heavy_source_dir = Path(heavy_source_dir)
        heavy_imgs = sorted([p for p in heavy_source_dir.glob("*") if p.suffix.lower() in IMG_EXTS])
        print(f"[INFO] Using HEAVY source pool: {len(heavy_imgs)} images found in {heavy_source_dir}")
    
    # Fallback check
    if not heavy_imgs:
        print("[WARN] No heavy source images found or directory not provided.")
        print("       Falling back to Intra-Baseline mixing (weaker results expected).")
        heavy_imgs = baseline_imgs # fallback to self
        if len(baseline_imgs) < 2:
            raise RuntimeError("Need at least 2 images for intra-mixing.")

    random.seed(seed)

    print("Generating Mix Anchors (Heavy Source Strategy)")
    print(f"- Method: SeamlessClone ({'MIXED' if use_mixed_clone else 'NORMAL'})")
    print(f"- Canny: {canny1}-{canny2} | Dilate: {dilate_iter}")

    for img_a_path in tqdm(baseline_imgs):
        candidates = [p for p in heavy_imgs if p.name != img_a_path.name]
        if not candidates: candidates = heavy_imgs 
        
        img_b_path = random.choice(candidates)

        save_path = output_dir / img_a_path.name 

        ok = create_mix_crack_anchor(
            img_a_path,
            img_b_path,
            save_path,
            canny1=canny1,
            canny2=canny2,
            dilate_iter=dilate_iter,
            close_ksize=close_ksize,
            min_area=min_area,
            use_mixed_clone=use_mixed_clone,
            alpha_boost=alpha_boost,
        )
        
# ==========================================
# Part B) Benign anchors (Blur / JPEG)
# ==========================================
def create_benign_anchors(input_dir, output_root, blur_sigma=1.0, jpeg_quality=90):
    input_dir = Path(input_dir)
    output_root = Path(output_root)

    blur_dir = output_root / "blur"
    jpeg_dir = output_root / "jpeg"
    blur_dir.mkdir(parents=True, exist_ok=True)
    jpeg_dir.mkdir(parents=True, exist_ok=True)

    image_paths = glob(str(input_dir / "*.*"))
    image_paths = [p for p in image_paths if Path(p).suffix.lower() in VALID_EXTS]

    print(f"Generating Benign Anchors...")

    for img_path in tqdm(image_paths):
        img_path = Path(img_path)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None: continue

        # --- Blur ---
        img_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=float(blur_sigma))
        cv2.imwrite(str(blur_dir / img_path.name), img_blur)

        # --- JPEG ---
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        ok, buf = cv2.imencode(".jpg", img, encode_param)
        if ok:
            img_jpeg = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            cv2.imwrite(str(jpeg_dir / img_path.name), img_jpeg)


# ==========================================
# Unified runner
# ==========================================
def run_all_anchors(
    baseline_dir,
    output_root,
    heavy_source_dir=None, # [New]
    seed=42,
    # mix params
    canny1=30, canny2=100, dilate_iter=4, 
    use_mixed_clone=False,
    # benign params
    blur_sigma=1.0, jpeg_quality=50,
):
    output_root = Path(output_root)
    mix_dir = output_root / "mix"
    mix_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mix Anchor (Heavy Strategy)
    generate_mix_crack_anchors(
        baseline_dir=baseline_dir,
        output_dir=mix_dir,
        heavy_source_dir=heavy_source_dir,
        seed=seed,
        canny1=canny1,
        canny2=canny2,
        dilate_iter=dilate_iter,
        use_mixed_clone=use_mixed_clone
    )

    # 2. Benign Anchor
    create_benign_anchors(
        input_dir=baseline_dir,
        output_root=output_root,
        blur_sigma=blur_sigma,
        jpeg_quality=jpeg_quality,
    )

    print("\n[Done] All anchors generated.")


if __name__ == "__main__":
    BASELINE_DIR = "data/experiments/gac_eval/baseline"
    OUTPUT_ROOT = "data/experiments/gac_eval/anchors"

    HEAVY_SOURCE_DIR = "data/experiments/gac_eval/heavy_check" 

    run_all_anchors(
        baseline_dir=BASELINE_DIR,
        output_root=OUTPUT_ROOT,
        heavy_source_dir=HEAVY_SOURCE_DIR,
        seed=777,
        
        # Mix Params (Natural but Strong)
        use_mixed_clone=False, # False = Normal Clone, True = Mixed Clone
        dilate_iter=4,         
        canny1=30, canny2=100, 
        
        # Benign Params
        blur_sigma=1.0,
        jpeg_quality=50
    )