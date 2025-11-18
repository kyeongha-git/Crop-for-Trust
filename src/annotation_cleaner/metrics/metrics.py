#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
metrics.py
-----------
A collection of image quality evaluation metrics for comparing
restored or generated images against their reference originals.

Available metrics:
- L1 Distance: Mean absolute pixel difference
- SSIM: Structural Similarity Index
- Edge IoU: Edge overlap ratio using Canny detection
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ============================================================
# Metric Functions
# ============================================================


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the mean absolute pixel difference (L1 distance) between two images.

    Args:
        a (np.ndarray): First image.
        b (np.ndarray): Second image.

    Returns:
        float: Average absolute pixel difference.
    """
    if a.shape != b.shape:
        raise ValueError(f"L1 Error: Image size mismatch {a.shape} vs {b.shape}")
    return np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    Args:
        a (np.ndarray): First image.
        b (np.ndarray): Second image.

    Returns:
        float: SSIM score (1.0 = identical, 0.0 = dissimilar).
    """
    if a.shape != b.shape:
        raise ValueError(f"SSIM Error: Image size mismatch {a.shape} vs {b.shape}")
    g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    data_range = float(g1.max() - g1.min()) or 255.0
    return ssim(g1, g2, data_range=data_range)


def edge_iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Edge IoU (Intersection over Union) using Canny edge detection.

    This metric measures how well the edge structures of two images overlap.

    Args:
        a (np.ndarray): First image.
        b (np.ndarray): Second image.

    Returns:
        float: Edge IoU ratio between 0.0 and 1.0.
    """
    if a.shape != b.shape:
        raise ValueError(f"Edge IoU Error: Image size mismatch {a.shape} vs {b.shape}")
    g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    e1, e2 = cv2.Canny(g1, 100, 200), cv2.Canny(g2, 100, 200)
    inter = np.logical_and(e1 > 0, e2 > 0).sum()
    union = np.logical_or(e1 > 0, e2 > 0).sum()
    return float(inter) / union if union > 0 else 0.0


# ============================================================
# Wrapper
# ============================================================


def compute_all_metrics(img1: np.ndarray, img2: np.ndarray) -> dict:
    """
    Compute all supported quality metrics at once.

    Args:
        img1 (np.ndarray): Reference or original image.
        img2 (np.ndarray): Generated or restored image.

    Returns:
        dict: Dictionary containing all metric results.
    """
    return {
        "L1": l1_distance(img1, img2),
        "SSIM": ssim_score(img1, img2),
        "Edge_IoU": edge_iou(img1, img2),
    }
