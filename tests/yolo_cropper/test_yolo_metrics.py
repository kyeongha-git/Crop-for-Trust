import json
import pytest

# Import Parser Functions
from src.yolo_cropper.metrics.metrics_parser import (
    get_metrics_parser,
    parse_darknet_eval_log,
    parse_yolov5_eval_log,
    parse_yolov8_results,
)

@pytest.fixture
def mock_logs(tmp_path):
    """
    Creates temporary log files mimicking real model outputs.
    """
    # 1. Darknet Log (Pattern 1: Explicit P/R line)
    darknet_log_1 = tmp_path / "darknet_explicit.log"
    darknet_log_1.write_text(
        "calculation mAP...\n"
        "mean average precision (mAP@0.50) = 0.753376, or 75.34 %\n"
        "for conf_thresh = 0.25, precision = 0.82, recall = 0.79, F1-score = 0.80\n",
        encoding="utf-8"
    )

    # 2. Darknet Log (Pattern 2: TP/FP/FN counts)
    darknet_log_2 = tmp_path / "darknet_counts.log"
    darknet_log_2.write_text(
        "mean average precision (mAP@0.50) = 0.500000\n"
        "TP = 80, FP = 20, FN = 20\n"  # P=80/100=0.8, R=80/100=0.8
        "IoU threshold = 50 %\n",
        encoding="utf-8"
    )

    # 3. YOLOv5 Log (Console Table)
    yolov5_log = tmp_path / "yolov5_val.txt"
    yolov5_log.write_text(
        "val: data=data.yaml, weights=yolov5s.pt, batch_size=32\n"
        "               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95\n"
        "                 all        128        500      0.952      0.885      0.915      0.650\n"
        "              person        128        500      0.952      0.885      0.915      0.650\n",
        encoding="utf-8"
    )

    # 4. YOLOv8 Result (JSON)
    yolov8_json = tmp_path / "results.json"
    data = {
        "metrics": {
            "precision": 0.912,
            "recall": 0.875,
            "mAP@0.5": 92.5
        }
    }
    with open(yolov8_json, "w") as f:
        json.dump(data, f)

    return {
        "darknet_1": str(darknet_log_1),
        "darknet_2": str(darknet_log_2),
        "yolov5": str(yolov5_log),
        "yolov8": str(yolov8_json)
    }

def test_parse_darknet_explicit(mock_logs):
    """Test parsing Darknet logs with explicit Precision/Recall lines."""
    metrics = parse_darknet_eval_log(mock_logs["darknet_1"])
    
    assert metrics["mAP@0.5"] == pytest.approx(75.3376)
    assert metrics["precision"] == pytest.approx(0.82)
    assert metrics["recall"] == pytest.approx(0.79)

def test_parse_darknet_counts(mock_logs):
    """Test parsing Darknet logs with TP/FP/FN counts."""
    metrics = parse_darknet_eval_log(mock_logs["darknet_2"])
    
    # Expected: mAP=0.5 -> 50.0%, P=80/(80+20)=0.8, R=80/(80+20)=0.8
    assert metrics["mAP@0.5"] == 50.0
    assert metrics["precision"] == 0.8
    assert metrics["recall"] == 0.8

def test_parse_yolov5(mock_logs):
    """Test parsing YOLOv5 console output table."""
    metrics = parse_yolov5_eval_log(mock_logs["yolov5"])
    
    # Expected: P=0.952, R=0.885, mAP=0.915 -> 91.5%
    assert metrics["precision"] == 0.952
    assert metrics["recall"] == 0.885
    assert pytest.approx(metrics["mAP@0.5"], 0.01) == 91.5

def test_parse_yolov8(mock_logs):
    """Test parsing YOLOv8 JSON output."""
    metrics = parse_yolov8_results(mock_logs["yolov8"])
    
    assert metrics["precision"] == 0.912
    assert metrics["recall"] == 0.875
    assert metrics["mAP@0.5"] == 92.5

def test_parser_factory():
    """Verify get_metrics_parser returns the correct function."""
    assert get_metrics_parser("darknet") == parse_darknet_eval_log
    assert get_metrics_parser("yolov4-tiny") == parse_darknet_eval_log
    assert get_metrics_parser("yolov5") == parse_yolov5_eval_log
    assert get_metrics_parser("yolov8") == parse_yolov8_results

    with pytest.raises(ValueError):
        get_metrics_parser("unknown_model")