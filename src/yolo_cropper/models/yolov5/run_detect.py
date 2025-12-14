# src/yolo_cropper/models/yolov5/run_detect.py
import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[4]
YOLO_DIR = ROOT / "third_party" / "yolov5"

if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

try:
    from detect import run
except ImportError:
    sys.path.append(str(YOLO_DIR))
    from detect import run

if __name__ == "__main__":
    print("YOLOv5 detect module imported successfully!")