import os
import shutil
import warnings
import imagehash
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

# ================= CONFIGURATION =================
FILE_PATH = Path(__file__).resolve()
BASE_PATH = FILE_PATH.parent 

# 경로 설정
real_label_dataset = BASE_PATH / "data/experiments/diff_map/yolov5_no_aug"
orig_dataset = BASE_PATH / "data/experiments/diff_map/original"
gen_dataset = BASE_PATH / "data/experiments/diff_map/generation"
output_dataset = BASE_PATH / "data/experiments/diff_map/original_label_mapping"

CLASSES = ['repair', 'replace']
HASH_SIZE = 8
MATCH_THRESH = 10

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 0. Initialization
# ---------------------------------------------------------
def initialize_output_folder():
    print(f"{'='*10} Initializing Output Sandbox {'='*10}")
    print(f"Base Path: {BASE_PATH}")
    print(f"Output Target: {output_dataset}")
    
    if output_dataset.exists():
        print(f"🧹 Removing existing output directory: {output_dataset}")
        shutil.rmtree(output_dataset)
    
    output_dataset.mkdir(parents=True, exist_ok=True)
    
    for cls in CLASSES:
        src_dir = orig_dataset / cls
        dst_dir = output_dataset / cls / "images"
        dst_dir.mkdir(parents=True, exist_ok=True)
        (output_dataset / cls / "labels").mkdir(parents=True, exist_ok=True)
        
        if not src_dir.exists():
            print(f"Warning: Source directory not found: {src_dir}")
            continue
            
        print(f"Copying images for class '{cls}' to output folder...")
        files = list(src_dir.glob('*.*'))
        valid_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG'}
        
        count = 0
        for f in tqdm(files, desc=f"Copying {cls}"):
            if f.suffix in valid_exts:
                shutil.copy2(f, dst_dir / f.name)
                count += 1
        print(f"   -> Copied {count} images.")
    print("Initialization Complete.\n")

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def compute_hash(img_path, target_size=(416, 416)):
    try:
        with Image.open(img_path) as img:
            img = img.convert("L")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            return imagehash.phash(img, hash_size=HASH_SIZE)
    except Exception as e:
        print(f"Hash Error {img_path}: {e}")
        return None

def build_hash_db(img_files, desc_name="Images"):
    hash_db = {}
    for p in tqdm(img_files, desc=f"Hashing {desc_name}"):
        h = compute_hash(p)
        if h:
            hash_db[p.stem] = h
    return hash_db

def get_image_files(folder_path):
    folder_path = Path(folder_path)
    files = list(folder_path.glob('*.*'))
    valid_exts = {'.jpg', '.jpeg', '.png'}
    img_files = [f for f in files if f.suffix.lower() in valid_exts]
    return img_files

def delete_image_and_label(img_path):
    try:
        if img_path.exists():
            img_path.unlink()
            # print(f"   Deleted Image: {img_path.name}") 
    except Exception as e:
        print(f"Error deleting image {img_path}: {e}")
        return

    parts = list(img_path.parts)
    if 'images' in parts:
        idx = len(parts) - 1 - parts[::-1].index('images')
        parts[idx] = 'labels'
        label_path = Path(*parts).with_suffix('.txt')
        
        if label_path.exists():
            try:
                label_path.unlink()
                # print(f"       └── Deleted Label: {label_path.name}")
            except Exception as e:
                print(f"        Error deleting label {label_path}: {e}")

# ---------------------------------------------------------
# 1. Renaming Logic
# ---------------------------------------------------------
def resolve_conflicts_by_renaming(target_root, cls_name):
    img_dir = target_root / cls_name / "images"
    gen_img_dir = gen_dataset / cls_name
    orig_img_dir = orig_dataset / cls_name
    
    all_files = [f for f in img_dir.glob('*.*') if not f.name.startswith('.')]
    stem_groups = defaultdict(list)
    for f in all_files:
        stem_groups[f.stem].append(f)
        
    renamed_count = 0
    
    for stem, files in tqdm(stem_groups.items(), desc=f" resolving conflicts {cls_name}"):
        if len(files) > 1:
            files.sort(key=lambda x: str(x))
            
            # original = files[0] 
            duplicates = files[1:]
            
            for i, dup_file in enumerate(duplicates, start=1):
                new_stem = f"{stem}{i}"
                new_name = f"{new_stem}{dup_file.suffix}"
                new_path = img_dir / new_name
                
                while new_path.exists():
                    i += 1
                    new_stem = f"{stem}{i}"
                    new_name = f"{new_stem}{dup_file.suffix}"
                    new_path = img_dir / new_name
                
                try:
                    old_name = dup_file.name
                    dup_file.rename(new_path)
                    renamed_count += 1

                    # Sync Rename in Original Dataset
                    orig_src_file = orig_img_dir / old_name
                    orig_dst_file = orig_img_dir / new_name

                    if orig_src_file.exists():
                        if not orig_dst_file.exists():
                            orig_src_file.rename(orig_dst_file)
                    
                    # Sync Rename in Generation Dataset
                    gen_src_file = gen_img_dir / old_name
                    gen_dst_file = gen_img_dir / new_name
                    
                    if gen_src_file.exists():
                        gen_src_file.rename(gen_dst_file)
                        
                except Exception as e:
                    print(f"   Error renaming {dup_file.name}: {e}")

    if renamed_count > 0:
        print(f"[{cls_name}] Resolved {renamed_count} conflicts")

# ---------------------------------------------------------
# 2. Match & Copy Logic
# ---------------------------------------------------------
def match_and_copy(target_root):
    total_copied = 0
    
    for cls in CLASSES:
        print(f"\nProcessing Class: {cls}")
        
        src_img_dir = real_label_dataset / cls / "images"
        src_lab_dir = real_label_dataset / cls / "labels"
        
        tgt_img_dir = target_root / cls / "images"
        tgt_lab_dir = target_root / cls / "labels"
        
        if not src_img_dir.exists():
            print(f"Skipping {cls}: Source not found ({src_img_dir}).")
            continue
            
        # Build Source DB
        print("   -> Building Source DB...")
        src_files = get_image_files(src_img_dir)
        src_db = build_hash_db(src_files, desc_name=src_img_dir.name)
        
        tgt_files = get_image_files(tgt_img_dir)
        copied_cnt = 0
        
        for tgt_file in tqdm(tgt_files, desc="Matching"):
            tgt_hash = compute_hash(tgt_file)
            if not tgt_hash: continue
            
            best_dist = 1000
            best_src_stem = None
            
            for src_stem, src_hash in src_db.items():
                dist = tgt_hash - src_hash
                if dist < best_dist:
                    best_dist = dist
                    best_src_stem = src_stem
            
            if best_dist <= MATCH_THRESH:
                src_label_path = src_lab_dir / f"{best_src_stem}.txt"
                if src_label_path.exists():
                    tgt_label_path = tgt_lab_dir / f"{tgt_file.stem}.txt"
                    shutil.copy2(src_label_path, tgt_label_path)
                    copied_cnt += 1
        
        print(f"{cls}: {copied_cnt}/{len(tgt_files)} labels recovered.")
        total_copied += copied_cnt

# ---------------------------------------------------------
# 3. Diagnostic & Clean Logic
# ---------------------------------------------------------
def run_diagnostic_and_clean(target_root):
    print(f"\nStarting Diagnostic & Cleaning on OUTPUT folder...")
    
    for cls in CLASSES:
        print(f"\n{'='*20} CLASS: {cls.upper()} {'='*20}")
        
        src_img_dir = real_label_dataset / cls / "images"
        tgt_img_dir = target_root / cls / "images"
        
        src_files = get_image_files(src_img_dir)
        tgt_files = get_image_files(tgt_img_dir)
        
        src_db = build_hash_db(src_files, desc_name="Source DB")
        
        match_log = defaultdict(list)
        failures = []
        
        for tgt_file in tqdm(tgt_files, desc="Checking Integrity"):
            tgt_hash = compute_hash(tgt_file)
            if not tgt_hash:
                failures.append(tgt_file)
                continue
                
            best_dist = 1000
            best_src = None
            for src_stem, src_hash in src_db.items():
                dist = tgt_hash - src_hash
                if dist < best_dist:
                    best_dist = dist
                    best_src = src_stem
            
            if best_dist <= MATCH_THRESH:
                match_log[best_src].append(tgt_file)
            else:
                failures.append(tgt_file)
        
        if failures:
            print(f"Deleting {len(failures)} unmatched files (Failures)...")
            for fail_path in failures:
                delete_image_and_label(fail_path)
        
        duplicates_count = 0
        for src, targets in match_log.items():
            if len(targets) > 1:
                duplicates_count += 1
                targets.sort()
                
                remove_files = targets[1:]
                for rm_file in remove_files:
                    delete_image_and_label(rm_file)
                    
        print(f"Processed duplicates for {duplicates_count} sources.")

        final_img_files = get_image_files(tgt_img_dir)
        final_lab_files = list((target_root / cls / "labels").glob("*.txt"))
        
        img_stems = set(f.stem for f in final_img_files)
        lab_stems = set(f.stem for f in final_lab_files)
        
        orphans = img_stems - lab_stems
        if orphans:
            print(f"Cleaning {len(orphans)} images without labels...")
            for f in final_img_files:
                if f.stem in orphans:
                    delete_image_and_label(f)

        print(f"{cls.upper()} Cleaning Complete.")


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    print("Code running from:", FILE_PATH)
    
    initialize_output_folder()
    
    for cls in CLASSES:
        resolve_conflicts_by_renaming(output_dataset, cls)
        
    match_and_copy(output_dataset)
    
    run_diagnostic_and_clean(output_dataset)

if __name__ == "__main__":
    main()