#!/bin/bash
set -e

echo "========================================================"
echo "   Crop for Trust: Reliability-Aware Pipeline Demo      "
echo "          (Docker Environment - YOLOv5)                 "
echo "========================================================"

# 1. API Key Check
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY is missing!"
    exit 1
fi

# 2. Check YOLOv5 Existence (Sanity Check)
if [ ! -d "third_party/yolov5" ]; then
    echo "Error: YOLOv5 source code is missing in /app/third_party/yolov5"
    exit 1
else
    echo "YOLOv5 Source Code detected."
fi

# 3. Run Pipeline
echo -e "\n[Pipeline] Starting src/main.py..."
python src/main.py --config utils/config_docker.yaml

echo -e "\n--------------------------------------------------------"
echo "[DEBUG] Post-Run Verification"

echo "1. Saved Model Check:"
ls -lh /app/saved_model/yolo_cropper/yolov5.pt || echo "Model file (yolov5.pt) not found!"

echo "2. Input Images Check:"
ls -1 /app/data/sample/original | head -n 3 || echo "Input dir empty!"

echo "3. Output Crops Check:"
ls -1 /app/data/sample/original_crop/yolov5 | head -n 3 || echo "No crops found in yolov5 folder!"
echo "--------------------------------------------------------"

echo "========================================================"
echo "   🎉 Demo Pipeline Completed Successfully!             "
echo "========================================================"