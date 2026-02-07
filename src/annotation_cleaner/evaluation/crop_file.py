#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
from tqdm import tqdm

from utils.logging import get_logger

warnings.filterwarnings("ignore")
logger = get_logger("EvalUtils")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def get_crop_coords(yolo_line: str, img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int]]:
    """Parses YOLO label to pixel coordinates."""
    parts = yolo_line.strip().split()
    if len(parts) < 5: return None
    try:
        nums = list(map(float, parts))
    except ValueError: return None
    
    coords = nums[1:]
    if len(coords) == 4: # BBox
        xc, yc, w, h = coords
        box_w, box_h = w * img_w, h * img_h
        left, top = (xc * img_w) - (box_w/2), (yc * img_h) - (box_h/2)
        right, bottom = left + box_w, top + box_h
    else: # Polygon
        xs, ys = coords[0::2], coords[1::2]
        if not xs: return None
        left, right = min(xs) * img_w, max(xs) * img_w
        top, bottom = min(ys) * img_h, max(ys) * img_h

    left, top = max(0, int(left)), max(0, int(top))
    right, bottom = min(img_w, int(right)), min(img_h, int(bottom))
    if right <= left or bottom <= top: return None
    return (left, top, right, bottom)

def crop_and_save(img_path: Path, coords: Tuple, save_dir: Path, stem: str, idx: int) -> bool:
    if idx != 0: return True # Only crop first object
    save_dir.mkdir(parents=True, exist_ok=True)
    out_ext = img_path.suffix.lower()
    save_path = save_dir / f"{stem}{out_ext}"
    try:
        with Image.open(img_path) as img:
            crop = img.crop(coords)
            if out_ext in {".jpg", ".jpeg"} and crop.mode != "RGB":
                crop = crop.convert("RGB")
            crop.save(save_path)
        return True
    except Exception as e:
        logger.warning(f"Crop failed {img_path}: {e}")
        return False

def crop_dataset(label_dir: Path, input_img_dir: Path, output_crop_dir: Path) -> None:
    """
    Crops all images in input_img_dir based on labels in label_dir,
    saving results to output_crop_dir.
    """
    if not input_img_dir.exists():
        logger.warning(f"Input dir missing: {input_img_dir}. Skipping cropping.")
        return

    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        logger.warning(f"No labels found in {label_dir}")
        return

    output_crop_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cropping from {input_img_dir.name} -> {output_crop_dir.name}")

    count = 0
    for label_path in tqdm(label_files, desc=f"Crop {input_img_dir.name}"):
        stem = label_path.stem
        # Find corresponding image
        img_path = None
        for ext in IMG_EXTS:
            if (input_img_dir / (stem + ext)).exists():
                img_path = input_img_dir / (stem + ext)
                break
        
        if not img_path: continue

        try:
            with Image.open(img_path) as img: W, H = img.size
            # Read label
            try: lines = label_path.read_text(encoding="utf-8").splitlines()
            except: lines = label_path.read_text(encoding="cp949").splitlines()

            for idx, line in enumerate(lines):
                if not line.strip(): continue
                coords = get_crop_coords(line, W, H)
                if coords:
                    if crop_and_save(img_path, coords, output_crop_dir, stem, idx):
                        count += 1
        except Exception: continue
    
    logger.info(f"Finished cropping {count} images to {output_crop_dir}")