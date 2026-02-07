import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
from math import exp
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================
DIR_ROOT = Path("data/experiments/gac_eval")
DIR_REF = DIR_ROOT / "baseline"  # Ground Truth (Clean)
DIR_ANCHORS = DIR_ROOT / "anchors"

VGG_WEIGHT_PATH = "saved_model/classifier/original_crop/yolov5/vgg.pt"

OUT_DIR = Path("results/annotation_cleaner")
OUT_SUMMARY_CSV = OUT_DIR / "metrics_summary.csv"
OUT_DETAIL_CSV = OUT_DIR / "metrics_details.csv"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ANCHOR_TYPES = ["marked", "blur", "jpeg", "mix", "gac", "ood"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= METRIC FUNCTIONS =================
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))

def calculate_edge_distance(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    sobel_x1 = cv2.Sobel(g1, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y1 = cv2.Sobel(g1, cv2.CV_64F, 0, 1, ksize=3)
    mag1 = np.sqrt(sobel_x1**2 + sobel_y1**2)
    
    sobel_x2 = cv2.Sobel(g2, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y2 = cv2.Sobel(g2, cv2.CV_64F, 0, 1, ksize=3)
    mag2 = np.sqrt(sobel_x2**2 + sobel_y2**2)
    
    return float(np.mean(np.abs(mag1 - mag2)))

def create_window(window_size, channel):
    _1D_window = torch.tensor([exp(-(x - window_size//2)**2/float(2*1.5**2)) for x in range(window_size)])
    _1D_window = _1D_window.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def calculate_ssim(img1_bgr, img2_bgr):
    t1 = torch.from_numpy(img1_bgr).permute(2,0,1).float().unsqueeze(0) / 255.0
    t2 = torch.from_numpy(img2_bgr).permute(2,0,1).float().unsqueeze(0) / 255.0
    if torch.cuda.is_available():
        t1 = t1.cuda()
        t2 = t2.cuda()
    window = create_window(11, 3)
    if torch.cuda.is_available():
        window = window.cuda()
    score = _ssim(t1, t2, window, 11, 3)
    return float(score.item())

# ================= MODEL & UTILS =================
class VGGFeatureExtractor(nn.Module):
    def __init__(self, weight_path=None, device="cpu"):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        if weight_path and os.path.exists(weight_path):
            try:
                checkpoint = torch.load(weight_path, map_location="cpu")
                state_dict = checkpoint.state_dict() if isinstance(checkpoint, nn.Module) else checkpoint
                if "model_state_dict" in state_dict: state_dict = state_dict["model_state_dict"]
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("backbone.features") or k.startswith("features."):
                        new_k = k.replace("backbone.", "")
                        new_state_dict[new_k] = v
                vgg.load_state_dict(new_state_dict, strict=False)
                # print("[INFO] Custom VGG weights loaded.")
            except:
                print("[WARN] Failed to load custom weights. Using ImageNet.")
        
        self.slice1 = nn.Sequential(*list(vgg.features.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.features.children())[4:9])
        self.slice3 = nn.Sequential(*list(vgg.features.children())[9:16])
        self.slice4 = nn.Sequential(*list(vgg.features.children())[16:23])
        for p in self.parameters(): p.requires_grad = False
        self.to(device)
        self.eval()

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return (h1, h2, h3, h4)

def normalize_tensor(x, eps=1e-10):
    norm = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm + eps)

def cv2_to_tensor(img_bgr, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return transform(img_rgb).unsqueeze(0).to(device)

def calc_feature_distance(model, t_ref, t_img):
    feats_ref = model(t_ref)
    feats_img = model(t_img)
    total = 0.0
    for f0, f1 in zip(feats_ref, feats_img):
        total += torch.mean((normalize_tensor(f0) - normalize_tensor(f1)) ** 2)
    return float(total.item())

def calculate_l1_distance(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(g1, g2)))

def safe_imread(path):
    try:
        data = np.fromfile(str(path), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except:
        return None
    
def save_visualization(img_ref, img_anchor, anchor_type, filename, output_dir):
    vis_dir = output_dir / "visualizations" / filename
    vis_dir.mkdir(parents=True, exist_ok=True)

    def get_edge_map(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
        mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return mag_vis

    edge_ref = get_edge_map(img_ref)
    edge_anchor = get_edge_map(img_anchor)

    cv2.imwrite(str(vis_dir / f"clean.png"), edge_ref)
    cv2.imwrite(str(vis_dir / f"{anchor_type}.png"), edge_anchor)

# ================= MAIN PROCESS =================
def process_full_metrics():
    print("=" * 80)
    print(" Starting Comprehensive Metric Evaluation")
    print(f"Reference: {DIR_REF}")
    print(f"Anchors:   {DIR_ANCHORS}")
    print("=" * 80)
    
    feat_model = VGGFeatureExtractor(weight_path=str(VGG_WEIGHT_PATH), device=DEVICE)
    ref_files = sorted([p for p in DIR_REF.glob("*") if p.suffix.lower() in IMG_EXTS])
    
    if not ref_files:
        print("[ERROR] No reference images found.")
        return

    anchor_maps = {}
    ood_single_image_path = None

    for anchor_type in ANCHOR_TYPES:
        anchor_dir = DIR_ANCHORS / anchor_type
        if not anchor_dir.exists():
            print(f"[WARN] Anchor directory missing: {anchor_dir}")
            continue
        
        if anchor_type == "ood":
            ood_files = [p for p in anchor_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
            if ood_files:
                ood_single_image_path = ood_files[0]
            else:
                print(f"[WARN] No images found in OOD directory.")
        
        file_map = {}
        for p in anchor_dir.glob("*"):
            if p.suffix.lower() in IMG_EXTS:
                file_map[p.stem] = p 
        anchor_maps[anchor_type] = file_map

    all_data = []

    # --- Loop through Reference Images ---
    for ref_path in tqdm(ref_files, desc="Comparing Images"):
        filename = ref_path.name
        stem = ref_path.stem
        
        img_ref = safe_imread(ref_path)
        if img_ref is None: continue
        t_ref = cv2_to_tensor(img_ref, DEVICE)

        for anchor_type in ANCHOR_TYPES:
            anchor_file = None
            if anchor_type == "ood":
                anchor_file = ood_single_image_path
            else:
                if anchor_type in anchor_maps and stem in anchor_maps[anchor_type]:
                    anchor_file = anchor_maps[anchor_type][stem]
            
            if anchor_file is None or not anchor_file.exists():
                continue
            
            img_anchor = safe_imread(anchor_file)
            if img_anchor is None: continue

            if img_anchor.shape != img_ref.shape:
                img_anchor = cv2.resize(img_anchor, (img_ref.shape[1], img_ref.shape[0]), interpolation=cv2.INTER_LINEAR)

            # --- Calculate Metrics ---
            l1 = calculate_l1_distance(img_ref, img_anchor)
            save_visualization(img_ref, img_anchor, anchor_type, output_dir=OUT_DIR, filename=filename)
            psnr = calculate_psnr(img_ref, img_anchor)
            ssim = calculate_ssim(img_ref, img_anchor)
            edge_l1 = calculate_edge_distance(img_ref, img_anchor)
            
            t_anchor = cv2_to_tensor(img_anchor, DEVICE)
            with torch.no_grad():
                d_feat = calc_feature_distance(feat_model, t_ref, t_anchor)
            
            all_data.append({
                "Filename": filename,
                "Anchor": anchor_type,
                "L1": l1,
                "D_FeatMap": d_feat,
                "PSNR": psnr,
                "SSIM": ssim,
                "Edge_L1": edge_l1
            })

    if not all_data:
        print("[ERROR] No metric data calculated.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(all_data)
    cols = ["Filename", "Anchor", "L1", "Edge_L1", "D_FeatMap", "PSNR", "SSIM"]
    df = df[cols]
    df.to_csv(OUT_DETAIL_CSV, index=False)
    print(f"\n[SAVED] Detailed metrics: {OUT_DETAIL_CSV}")

    summary = df.groupby("Anchor")[["L1", "Edge_L1", "D_FeatMap", "PSNR", "SSIM"]].mean()
    summary = summary.reindex(ANCHOR_TYPES)
    summary.to_csv(OUT_SUMMARY_CSV)
    print(f"[SAVED] Summary metrics:  {OUT_SUMMARY_CSV}")

    # ================= Console =================
    print("\n" + "="*60)
    print("FINAL EVALUATION REPORT (Average)")
    print("="*60)
    
    metrics_config = [
        ("L1", False),       # (Metric Name, Higher is Better?)
        ("Edge_L1", False),
        ("D_FeatMap", False),
        ("PSNR", True),
        ("SSIM", True)
    ]

    for metric, higher_is_better in metrics_config:
        print(f"\n[{metric}] {'(↑ Higher is Better)' if higher_is_better else '(↓ Lower is Better)'}")
        
        vals = summary[metric].dropna()
        
        sorted_vals = vals.sort_values(ascending=not higher_is_better)
        
        rank = 1
        for anchor, score in sorted_vals.items():
            if metric == "D_FeatMap":
                val_str = f"{score:.6f}"
            else:
                val_str = f"{score:.4f}"
            
            mark = "OURS" if anchor.lower() == "gac" else ""
            print(f"   {rank}. {anchor:<10} : {val_str} {mark}")
            rank += 1

    print("\n" + "="*60)
    print("Done.")

if __name__ == "__main__":
    process_full_metrics()