import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent.parent


# ================= CONFIGURATION =================
INPUT_ROOT = BASE_DIR / 'data' / 'yolo_cropper' / 'yolov5_no_aug'
OUTPUT_ROOT_BLUR = BASE_DIR / 'data' / 'experiments' / 'baseline_yolo' / 'blur'
OUTPUT_ROOT_BLACKOUT = BASE_DIR / 'data' / 'experiments' / 'baseline_yolo' / 'black-out'

SPLITS = ['train', 'test', 'valid']

# Gaussian Blur Parameter
BLUR_KSIZE = (51, 51)
BLUR_SIGMA = 20

CLASSES = ['repair', 'replace']
# =================================================


def process_image(img_path, label_path, mode='blur'):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 5: continue
            
            cls_id = int(parts[0])
            coords = parts[1:]
            
            if len(coords) == 4: # Standard YOLO BBox (xc, yc, w, h)
                xc, yc, bw, bh = coords
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
            else: # Polygon Format (x1, y1, x2, y2, ...)
                points = []
                for i in range(0, len(coords), 2):
                    px = int(coords[i] * w)
                    py = int(coords[i+1] * h)
                    points.append([px, py])
                
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)

    mask_3ch = cv2.merge([mask, mask, mask])
    
    if mode == 'blur':
        blurred_img = cv2.GaussianBlur(img, BLUR_KSIZE, BLUR_SIGMA)
        result = np.where(mask_3ch == 255, img, blurred_img)
        
    elif mode == 'black-out':
        result = cv2.bitwise_and(img, mask_3ch)
        
    else:
        result = img

    return result


def run_experiment_gen(target_mode):
    output_root = OUTPUT_ROOT_BLUR if target_mode == 'blur' else OUTPUT_ROOT_BLACKOUT
    
    print(f"\nStarting Data Generation: [{target_mode.upper()}]")
    print(f"   Input: {INPUT_ROOT}")
    print(f"   Output: {output_root}")

    for split in SPLITS:
        src_img_dir = Path(INPUT_ROOT) / split / 'images'
        src_lbl_dir = Path(INPUT_ROOT) / split / 'labels'
        
        dst_img_dir = Path(output_root) / split / 'images'
        dst_lbl_dir = Path(output_root) / split / 'labels'
        
        print("== will make dirs ==\n   dst_img_dir:", dst_img_dir)
        print("   dst_lbl_dir:", dst_lbl_dir)

        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        if not src_img_dir.exists():
            print(f"Warning: {src_img_dir} does not exist. Skipping.")
            continue
            
        images = list(src_img_dir.glob('*'))
        
        print(f"Processing {split} set... ({len(images)} images)")
        
        for img_path in tqdm(images):
            label_name = img_path.stem + '.txt'
            label_path = src_lbl_dir / label_name
            
            processed_img = process_image(img_path, label_path, mode=target_mode)
            
            if processed_img is not None:
                save_path = dst_img_dir / img_path.name
                cv2.imwrite(str(save_path), processed_img)
            
            if label_path.exists():
                shutil.copy(label_path, dst_lbl_dir / label_name)


def reorganize_dataset(root_dir):
    stats = defaultdict(lambda: defaultdict(int))
    
    for split in SPLITS:
        split_dir = root_dir / split
        
        labels_dir = split_dir / 'labels'
        if labels_dir.exists():
            shutil.rmtree(labels_dir)
            print(f"Deleted {labels_dir}")
        
        images_dir = split_dir / 'images'
        if not images_dir.exists():
            print(f"No images in {images_dir}")
            continue
            
        print(f"\nProcessing {split} images... ({len(list(images_dir.glob('*')))} files)")
        
        for class_name in CLASSES:
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)
        
        moved_count = 0
        for img_path in tqdm(list(images_dir.glob('*')), desc=f"{split}"):
            if img_path.is_file() and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                filename = img_path.name.lower()
                
                if 'repair' in filename:
                    target_dir = split_dir / 'repair'
                    stats[split]['repair'] += 1
                elif 'replace' in filename:
                    target_dir = split_dir / 'replace'
                    stats[split]['replace'] += 1
                else:
                    print(f"Unknown class: {img_path.name}")
                    continue
                
                shutil.move(str(img_path), str(target_dir / img_path.name))
                moved_count += 1
        
        remaining = len(list(images_dir.glob('*')))
        if remaining == 0:
            images_dir.rmdir()
            print(f"Deleted empty {images_dir}")
        else:
            print(f"{remaining} files remain in {images_dir}")
        
        print(f"   → Moved: {moved_count} images")
    
    return stats


if __name__ == "__main__":
    run_experiment_gen('blur')
    run_experiment_gen('black-out')
    print("\nAll experiments datasets generated successfully!")
    
    print("\n" + "="*60)
    
    print("Starting dataset reorganization...")
    blur_stats = reorganize_dataset(OUTPUT_ROOT_BLUR)
    blackout_stats = reorganize_dataset(OUTPUT_ROOT_BLACKOUT)
    
    print("\n" + "="*60)
    print("FINAL DATASET STATISTICS")
    print("="*60)
    for dataset, stats in [('BLUR', blur_stats), ('BLACK-OUT', blackout_stats)]:
        print(f"\n{dataset}:")
        for split in SPLITS:
            total = sum(stats[split].values())
            print(f"  {split:6}: repair={stats[split]['repair']:4}, replace={stats[split]['replace']:4}, total={total:4}")
    
    print("\nAll done! Datasets ready for training.")