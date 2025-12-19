#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
metrics_parser.py

Unified parser for YOLO detection performance metrics.
Extracts Precision, Recall, and mAP@0.5 from evaluation logs of various
YOLO versions (Darknet, YOLOv5, YOLOv8).
"""

import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Union

# Adjust path to import utils
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger

logger = get_logger("metrics_parser")


def parse_darknet_eval_log(log_path: str) -> Dict[str, Optional[float]]:
    """
    Parses Darknet-style evaluation logs (YOLOv2/v3/v4).

    Extracts metrics using regex patterns specific to Darknet's C output.

    Args:
        log_path (str): Path to the log file.

    Returns:
        Dict[str, Optional[float]]: Dictionary containing precision, recall, and mAP@0.5.
    """
    log_text = Path(log_path).read_text(errors="ignore")

    # ----------------------------
    # Extract mAP@0.5
    # ----------------------------
    m = re.search(
        r"mean average precision.*?=\s*([0-9]*\.?[0-9]+)\s*%?", log_text, re.I
    )
    mAP = float(m.group(1)) if m else None
    
    # Normalize mAP to percentage if strictly less than 1.0 (heuristic)
    mAP_pct = (
        mAP if (mAP is not None and mAP > 1) else (None if mAP is None else mAP * 100)
    )

    # ----------------------------
    # Extract Precision / Recall
    # ----------------------------
    # Pattern 1: Explicit precision/recall line
    pr = re.search(
        r"for\s+conf_thresh\s*=?\s*[0-9]*\.?[0-9]+\s*[, ]+precision\s*[:=]\s*([0-9]*\.?[0-9]+)\s*[, ]+recall\s*[:=]\s*([0-9]*\.?[0-9]+)",
        log_text,
        re.I,
    )

    if pr:
        precision = float(pr.group(1))
        recall = float(pr.group(2))
    else:
        # Pattern 2: Calculate from TP/FP/FN counts
        tpfpfn = re.search(
            r"TP\s*=\s*(\d+).*?FP\s*=\s*(\d+).*?FN\s*=\s*(\d+)", log_text, re.I | re.S
        )
        if tpfpfn:
            TP, FP, FN = map(int, tpfpfn.groups())
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        else:
            precision = recall = None

    return {
        "precision": precision,
        "recall": recall,
        "mAP@0.5": mAP_pct,
    }


def print_darknet_eval_summary(metrics: Dict[str, Optional[float]]) -> None:
    """
    Logs the parsed Darknet evaluation metrics.

    Args:
        metrics (Dict[str, Optional[float]]): The dictionary returned by the parser.
    """
    logger.info("=== Overall Evaluation (IoU=0.50, 101-point, conf_thresh=0.25) ===")
    
    if metrics["precision"] is not None:
        logger.info(f"Precision: {metrics['precision']:.4f}")
    if metrics["recall"] is not None:
        logger.info(f"Recall     : {metrics['recall']:.4f}")
    if metrics["mAP@0.5"] is not None:
        logger.info(f"mAP@0.50   : {metrics['mAP@0.5']:.2f}%")


def parse_yolov5_eval_log(log_path: str) -> Dict[str, Optional[float]]:
    """
    Parses YOLOv5 validation logs (standard `val.py` console output).

    Args:
        log_path (str): Path to the log file.

    Returns:
        Dict[str, Optional[float]]: Standardized metrics dictionary.
    """
    text = Path(log_path).read_text(errors="ignore")
    
    # Regex to find the 'all' class summary line
    # Columns: Class | Images | Labels | P | R | mAP@.5 | mAP@.5:.95
    match = re.search(
        r"all\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", text
    )

    if not match:
        logger.warning("Could not find 'all' metrics line in YOLOv5 log.")
        return {"precision": None, "recall": None, "mAP@0.5": None}

    precision = float(match.group(1))
    recall = float(match.group(2))
    map50 = float(match.group(3))

    return {
        "precision": precision,
        "recall": recall,
        "mAP@0.5": map50 * 100,  # Convert to percentage
    }


def parse_yolov8_results(json_path: str) -> Dict[str, float]:
    """
    Parses YOLOv8 `results.json` output.

    Args:
        json_path (str): Path to the results JSON file.

    Returns:
        Dict[str, float]: Standardized metrics dictionary.
    """
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"results.json not found: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Some versions nest metrics under a "metrics" key
    metrics = data.get("metrics", data)

    precision = float(metrics.get("precision", 0))
    recall = float(metrics.get("recall", 0))
    map50 = float(metrics.get("map50", metrics.get("mAP@0.5", 0)))

    return {"precision": precision, "recall": recall, "mAP@0.5": map50}


def get_metrics_parser(model_name: str) -> Callable[[str], Dict[str, Union[float, None]]]:
    """
    Factory function to retrieve the appropriate parser based on model name.

    Args:
        model_name (str): Identifier for the model (e.g., 'yolov5', 'darknet').

    Returns:
        Callable: The parsing function corresponding to the model type.

    Raises:
        ValueError: If the model architecture is unsupported.
    """
    model_name = model_name.lower()
    
    if "darknet" in model_name or "yolov2" in model_name or "yolov4" in model_name:
        return parse_darknet_eval_log
    elif "yolov5" in model_name:
        return parse_yolov5_eval_log
    elif "yolov8" in model_name:
        return parse_yolov8_results
    else:
        raise ValueError(f"Unsupported model_name for metrics parsing: {model_name}")