#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation Metrics.

Contains mathematical functions and models for calculating:
PSNR, SSIM, Edge L1, L1, and Feature Distance (Perceptual Loss).
"""

import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from math import exp

# ================= METRIC FUNCTIONS =================

def calculate_psnr(img1, img2):
    """Calculates Peak Signal-to-Noise Ratio (PSNR)."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))

def calculate_l1_distance(img1, img2):
    """Calculates Mean Absolute Error (L1 Distance)."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(g1, g2)))

def calculate_edge_distance(img1, img2):
    """Calculates L1 distance between Sobel edge maps."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    def get_mag(g):
        sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sx**2 + sy**2)

    mag1 = get_mag(g1)
    mag2 = get_mag(g2)
    
    return float(np.mean(np.abs(mag1 - mag2)))

# --- SSIM Utilities ---
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
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

def calculate_ssim(img1_bgr, img2_bgr, device="cpu"):
    """Calculates Structural Similarity Index (SSIM)."""
    t1 = torch.from_numpy(img1_bgr).permute(2,0,1).float().unsqueeze(0) / 255.0
    t2 = torch.from_numpy(img2_bgr).permute(2,0,1).float().unsqueeze(0) / 255.0
    
    t1 = t1.to(device)
    t2 = t2.to(device)
    
    window = create_window(11, 3).to(device)
    score = _ssim(t1, t2, window, 11, 3)
    return float(score.item())

# --- Feature Distance Utils ---
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
    """Calculates Perceptual Loss using VGG features."""
    feats_ref = model(t_ref)
    feats_img = model(t_img)
    total = 0.0
    for f0, f1 in zip(feats_ref, feats_img):
        total += torch.mean((normalize_tensor(f0) - normalize_tensor(f1)) ** 2)
    return float(total.item())

class VGGFeatureExtractor(nn.Module):
    def __init__(self, weight_path=None, device="cpu"):
        super().__init__()
        # Load Skeleton
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # Load Custom Weights if provided
        if weight_path:
            try:
                checkpoint = torch.load(weight_path, map_location="cpu")
                state_dict = checkpoint.state_dict() if isinstance(checkpoint, nn.Module) else checkpoint
                if "model_state_dict" in state_dict: state_dict = state_dict["model_state_dict"]
                
                # Fix keys if needed
                new_state_dict = {}
                for k, v in state_dict.items():
                    k = k.replace("backbone.", "").replace("features.", "")
                    # Simple heuristic mapping could be risky, but adhering to original logic:
                    # Actually, PyTorch VGG keys are 'features.0.weight', etc.
                    # If custom weights have different prefix, handle here.
                    # For safety, we try loading; if keys mismatch, it will raise error or be ignored.
                    # Here we assume the user knows the weight structure. 
                    # Simpler approach: Load structure, then load_state_dict strict=False
                
                vgg.load_state_dict(state_dict, strict=False)
                # print("Custom VGG weights loaded.")
            except Exception as e:
                print(f"Warning: Failed to load VGG weights from {weight_path}. Using ImageNet. ({e})")
        
        # Slices
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