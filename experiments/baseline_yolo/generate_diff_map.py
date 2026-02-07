import shutil
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Set
import cv2
import numpy as np
import imagehash
from PIL import Image
from tqdm import tqdm 

# ================= CONFIGURATION =================
FILE_PATH = Path(__file__).resolve()
BASE_PATH = FILE_PATH.parent 

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}
ORIGINAL_ROOT = BASE_PATH / "data/experiments/diff_map/original_label_mapping"
GEN_ROOT = BASE_PATH / "data/experiments/diff_map/generation"
SAVE_ORIG_CROP = BASE_PATH / "data/experiments/diff_map/original_gt_crop"
SAVE_GEN_CROP = BASE_PATH / "data/experiments/diff_map/generation_gt_crop"
CLASSES = ['repair', 'replace']

warnings.filterwarnings('ignore')

def safe_imread(image_path: Path):
    try:
        data = np.fromfile(str(image_path), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def _collect_images(folder: Path):
    files = []
    for ext in IMG_EXTS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    uniq = {p.name: p for p in files if p.is_file()}
    return dict(sorted(uniq.items(), key=lambda kv: kv[0].lower()))   
    

def get_crop_coords(yolo_line, img_w, img_h):
    parts = list(map(float, yolo_line.strip().split()))
    
    if len(parts) < 5:
        return None

    coords = parts[1:]
    left, top, right, bottom = 0, 0, 0, 0

    # CASE A: Standard YOLO BBox (x_c, y_c, w, h)
    if len(coords) == 4:
        xc, yc, w, h = coords
        
        box_w = w * img_w
        box_h = h * img_h
        
        left = (xc * img_w) - (box_w / 2)
        top = (yc * img_h) - (box_h / 2)
        right = left + box_w
        bottom = top + box_h

    # CASE B: Polygon Format (x1, y1, x2, y2, ...)
    else:
        xs = coords[0::2]
        ys = coords[1::2]
        
        min_x = min(xs) * img_w
        max_x = max(xs) * img_w
        min_y = min(ys) * img_h
        max_y = max(ys) * img_h
        
        left = min_x
        top = min_y
        right = max_x
        bottom = max_y

    left = max(0, int(left))
    top = max(0, int(top))
    right = min(img_w, int(right))
    bottom = min(img_h, int(bottom))
    
    if right <= left or bottom <= top:
        return None

    return (left, top, right, bottom)

def crop_and_save(img_path, coords, save_dir, filename_stem, obj_idx):
    try:
        with Image.open(img_path) as img:
            # (left, top, right, bottom)
            crop_img = img.crop(coords)
            
            save_name = f"{filename_stem}_crop{obj_idx}.png"
            save_path = save_dir / save_name
            
            crop_img.save(save_path)
            return True
    except Exception as e:
        return False

def run_crop_process():
    for cls in CLASSES:
        (SAVE_ORIG_CROP / cls).mkdir(parents=True, exist_ok=True)
        (SAVE_GEN_CROP / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        print(f"\nProcessing Class: {cls}")
        
        orig_image_dir = ORIGINAL_ROOT / cls / "images"
        orig_label_dir = ORIGINAL_ROOT / cls / "labels"
        gen_image_dir = GEN_ROOT / cls

        label_files = list(orig_label_dir.glob("*.txt"))
        
        if not label_files:
            print(f"No label files found in {orig_label_dir}")
            continue

        for label_path in tqdm(label_files, desc=f"{cls} Cropping"):
            stem = label_path.stem 
            
            orig_img_candidates = list(orig_image_dir.glob(f"{stem}.*"))
            if not orig_img_candidates:
                continue
            
            orig_img_path = orig_img_candidates[0]
            
            try:
                with Image.open(orig_img_path) as temp_img:
                    W, H = temp_img.size
            except:
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            # --- Multi-Object Loop ---
            for obj_idx, line in enumerate(lines):
                if not line.strip(): continue
                
                coords = get_crop_coords(line, W, H)
                
                if coords is None:
                    continue

                crop_and_save(
                    orig_img_path, 
                    coords, 
                    SAVE_ORIG_CROP / cls, 
                    stem, 
                    obj_idx 
                )
                
                gen_img_candidates = list(gen_image_dir.glob(f"{stem}.*"))
                if gen_img_candidates:
                    gen_img_path = gen_img_candidates[0]
                    crop_and_save(
                        gen_img_path, 
                        coords, 
                        SAVE_GEN_CROP / cls, 
                        stem, 
                        obj_idx
                    )

    print("\nAll jobs finished.")
    print(f"Original Crops saved to: {SAVE_ORIG_CROP}")
    print(f"Generation Crops saved to: {SAVE_GEN_CROP}")

def calculate_diff(orig_path: Path, gen_path: Path, out_dir: Path, mapping_file: Path):
    orig_img = safe_imread(orig_path)
    gen_img = safe_imread(gen_path)

    if orig_img is None or gen_img is None:
        return

    if orig_img.shape != gen_img.shape:
        gen_img = cv2.resize(gen_img, (orig_img.shape[1], orig_img.shape[0]))

    diff = cv2.absdiff(orig_img, gen_img)

    safe_name = f"temp_{len(list(out_dir.glob('temp_*.jpg'))):06d}_diff.jpg"
    temp_path = out_dir / safe_name

    ok = cv2.imwrite(str(temp_path), diff)
    if not ok:
        return

    with open(mapping_file, "a", encoding="utf-8") as f:
        f.write(f"{safe_name}\t{orig_path.name}\n")

    return True

def restore_original_names(out_dir: Path, mapping_file: Path):
    if not mapping_file.exists():
        print("mapping.txt not found")
        return

    with open(mapping_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    restored = 0
    for line in tqdm(lines, desc="Restoring names"):
        safe_name, orig_name = line.split("\t")
        temp_path = out_dir / safe_name
        final_path = out_dir / orig_name

        if temp_path.exists():
            if final_path.exists():
                final_path.unlink()
            temp_path.rename(final_path)
            restored += 1

    print(f"{restored}/{len(lines)} file restored.")
    mapping_file.unlink()

def create_diff_map_full(original_root: str, generation_root: str, output_root):
    orig_root = Path(original_root)
    gen_root = Path(generation_root)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for subdir in ["repair", "replace"]:
        orig_dir = orig_root / subdir
        gen_dir = gen_root / subdir
        out_dir = out_root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        mapping_file = out_dir / "mapping.txt"
        if mapping_file.exists():
            mapping_file.unlink()

        if not orig_dir.exists() or not gen_dir.exists():
            print(f"Skipping {subdir} (missing dir)")
            continue

        orig_map = _collect_images(orig_dir)
        gen_map = _collect_images(gen_dir)

        common_names = sorted(set(orig_map.keys()) & set(gen_map.keys()), key=lambda s: s.lower())

        print(f"Processing {subdir}: {len(common_names)} matched pairs "
              f"(orig={len(orig_map)}, gen={len(gen_map)})")

        success = 0
        
        for name in tqdm(common_names, total=len(common_names)):
            if calculate_diff(orig_map[name], gen_map[name], out_dir, mapping_file):
                success += 1

        restore_original_names(out_dir, mapping_file)
        print(f"{subdir} Done.\n")


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    print(f"Script Location: {BASE_PATH}")
    
    print("Starting Crop Process...")
    run_crop_process()
    print("\n")

    print("Starting ROI DIFF MAP...")
    
    # Full Image
    create_diff_map_full(
        BASE_PATH / "data/experiments/diff_map/original", 
        BASE_PATH / "data/experiments/diff_map/generation", 
        BASE_PATH / "results/baseline_yolo/full_images"
    )

    # Local Representation
    # create_diff_map_full(
    #     BASE_PATH / "data/experiments/diff_map/original_gt_crop", 
    #     BASE_PATH / "data/experiments/diff_map/generation_gt_crop", 
    #     BASE_PATH / "results/baseline_yolo/local_representation"
    # )

if __name__ == "__main__":
    main()