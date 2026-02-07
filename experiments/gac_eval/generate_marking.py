import cv2
import numpy as np
import os
import random
import math
from pathlib import Path

# ================= CONFIGURATION =================
base_dir = Path("data/experiments/gac_eval")
img_dir = base_dir / "original" 
label_dir = base_dir / "label"
output_dir = base_dir / "marked"


# ================= MAIN FUNCTIONS =================
def get_bezier_curve(points, num_points=100):
    points = np.array(points)
    n = len(points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    
    for i in range(n + 1):
        coef = math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
        bernstein = coef * (t ** i) * ((1 - t) ** (n - i))
        curve += np.outer(bernstein, points[i])
        
    return curve.astype(np.int32)

def parse_yolo_label(txt_path, img_w, img_h):
    bboxes = []
    if not os.path.exists(txt_path):
        return bboxes

    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: continue
        
        cx, cy, w, h = map(float, parts[1:5])
        
        # Un-normalize
        box_w = w * img_w
        box_h = h * img_h
        box_cx = cx * img_w
        box_cy = cy * img_h
        
        x1 = int(box_cx - box_w / 2)
        y1 = int(box_cy - box_h / 2)
        x2 = int(box_cx + box_w / 2)
        y2 = int(box_cy + box_h / 2)
        
        bboxes.append((x1, y1, x2, y2))
        
    return bboxes

def draw_synthetic_marking(img, bboxes):
    marked_img = img.copy()
    
    color = (255, 255, 255) # White
    thickness = random.randint(4, 6) # 4~6px
    
    for (x1, y1, x2, y2) in bboxes:
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        start_pt = (random.randint(x1, x1 + (x2-x1)//3), random.randint(y1, y2))
        end_pt = (random.randint(x1 + 2*(x2-x1)//3, x2), random.randint(y1, y2))
        control_pt = (random.randint(x1, x2), random.randint(y1, y2))
        
        control_points = [start_pt, control_pt, end_pt]
        curve_points = get_bezier_curve(control_points, num_points=50)
        
        cv2.polylines(marked_img, [curve_points], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        
    return marked_img

def main_synthetic_generation():
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")))
    
    print(f"Found {len(img_files)} images.")

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        h, w = img.shape[:2]
        
        lbl_name = img_path.stem + ".txt"
        lbl_path = label_dir / lbl_name
        
        if not lbl_path.exists():
            print(f"Warning: No label file for {img_path.name}")
            continue
            
        bboxes = parse_yolo_label(str(lbl_path), w, h)
        
        if not bboxes:
            print(f"Skipping {img_path.name}: No valid bbox found.")
            continue
            
        syn_marked_img = draw_synthetic_marking(img, bboxes)
        
        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), syn_marked_img)
        print(f"Saved: {save_path.name} (Boxes: {len(bboxes)})")

if __name__ == "__main__":
    main_synthetic_generation()