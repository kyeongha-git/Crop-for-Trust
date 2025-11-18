import json
import re
from pathlib import Path


"""
metrics_parser.py
-----------------
This module reads and interprets YOLO or Darknet evaluation logs 
to extract key performance metrics such as Precision, Recall, 
and mAP@0.5.

It supports multiple model formats (Darknet, YOLOv5, YOLOv8) 
and automatically selects the right parser based on model name.

In short, it provides a unified interface to analyze detection 
performance results across different YOLO versions.
"""


def parse_darknet_eval_log(log_path: str):
    """
    Parse a Darknet-style evaluation log and extract:
      - Precision
      - Recall
      - mAP@0.50

    Works with Darknet, YOLOv2, and YOLOv4 output logs.
    """
    log_text = Path(log_path).read_text(errors="ignore")

    # ----------------------------
    # mean Average Precision (mAP@0.5)
    # ----------------------------
    m = re.search(
        r"mean average precision.*?=\s*([0-9]*\.?[0-9]+)\s*%?", log_text, re.I
    )
    mAP = float(m.group(1)) if m else None
    mAP_pct = (
        mAP if (mAP is not None and mAP > 1) else (None if mAP is None else mAP * 100)
    )

    # ----------------------------
    # Precision / Recall
    # ----------------------------
    pr = re.search(
        r"for\s+conf_thresh\s*=?\s*[0-9]*\.?[0-9]+\s*[, ]+precision\s*[:=]\s*([0-9]*\.?[0-9]+)\s*[, ]+recall\s*[:=]\s*([0-9]*\.?[0-9]+)",
        log_text,
        re.I,
    )

    if pr:
        precision = float(pr.group(1))
        recall = float(pr.group(2))
    else:
        # Fallback: TP/FP/FN 방식
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


def print_darknet_eval_summary(metrics: dict):
    """
    Display Darknet evaluation results in a readable format.
    """
    print("=== Overall Evaluation (IoU=0.50, 101-point, conf_thresh=0.25) ===")
    if metrics["precision"] is not None:
        print(f"Precision: {metrics['precision']:.4f}")
    if metrics["recall"] is not None:
        print(f"Recall   : {metrics['recall']:.4f}")
    if metrics["mAP@0.5"] is not None:
        print(f"mAP@0.50 : {metrics['mAP@0.5']:.2f}%")


def parse_yolov5_eval_log(log_path: str):
    """
    Parse YOLOv5 validation logs to extract overall metrics:
      - Precision
      - Recall
      - mAP@0.5

    Compatible with standard `val.py` logs for local datasets.
    """
    text = Path(log_path).read_text(errors="ignore")
    match = re.search(
        r"all\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", text
    )

    if not match:
        print("Could not find 'all' metrics line in YOLOv5 log.")
        return {"precision": None, "recall": None, "mAP@0.5": None}

    precision = float(match.group(1))
    recall = float(match.group(2))
    map50 = float(match.group(3))

    return {
        "precision": precision,
        "recall": recall,
        "mAP@0.5": map50 * 100,  # convert to percentage
    }


def parse_yolov8_results(json_path: str):
    """
    Parse YOLOv8 `results.json` (from Ultralytics val.py output).

    Extracts main performance indicators:
      - Precision
      - Recall
      - mAP@0.5
    """
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"results.json not found: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", data)  # some exports nest under "metrics"

    precision = float(metrics.get("precision", 0))
    recall = float(metrics.get("recall", 0))
    map50 = float(metrics.get("map50", metrics.get("mAP@0.5", 0)))

    return {"precision": precision, "recall": recall, "mAP@0.5": map50}


def get_metrics_parser(model_name: str):
    """
    Automatically select the correct parsing function
    based on the YOLO model version name.

    Example:
        parser = get_metrics_parser("yolov5")
        metrics = parser("path/to/eval_log.txt")
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
