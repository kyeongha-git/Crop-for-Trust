#!/bin/bash
set -e

# Default setting
MODEL_TYPE="yolov5"

# Parse arguments to determine which model is being used
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --yolo_model) MODEL_TYPE="$2"; shift ;;
        *) ;; # Ignore other arguments, let them pass or handled by fixed config
    esac
    shift
done

echo "========================================================"
echo "   Crop for Trust: Reliability-Aware Pipeline Demo      "
echo "          (Docker Environment - $MODEL_TYPE)            "
echo "========================================================"

# 1. API Key Check
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY is missing!"
    exit 1
fi

# 2. Environment Sanity Check (Based on Model Type)
echo "[Init] Checking environment for $MODEL_TYPE..."

if [[ "$MODEL_TYPE" == "yolov5" ]]; then
    # Check if YOLOv5 source code exists (cloned in Dockerfile)
    if [ ! -d "third_party/yolov5" ]; then
        echo "Error: YOLOv5 source code is missing in /app/third_party/yolov5"
        exit 1
    else
        echo " -> YOLOv5 Source Code detected."
    fi

elif [[ "$MODEL_TYPE" == "yolov2" || "$MODEL_TYPE" == "yolov4" ]]; then
    # Check if Darknet binary exists (compiled in Dockerfile)
    if [ ! -f "third_party/darknet/darknet" ]; then
        echo "Error: Darknet executable not found! Compilation failed in Dockerfile."
        exit 1
    else
        echo " -> Darknet executable detected."
    fi

elif [[ "$MODEL_TYPE" == "yolov8" ]]; then
    # YOLOv8 uses the 'ultralytics' pip package (installed in Dockerfile)
    echo " -> Using Ultralytics (YOLOv8) library."
fi

# 3. Run Pipeline
# Passing the detected --yolo_model argument to Python script
echo -e "\n[Pipeline] Starting src/main.py..."
python src/main.py --yolo_model "$MODEL_TYPE" --config utils/config_docker.yaml

echo -e "\n--------------------------------------------------------"
echo "[DEBUG] Post-Run Verification"

echo "1. Saved Model Check (Downloaded by Python if needed):"
# Note: File extension might differ (pt vs weights), checking loosely based on dir
if [ -d "/app/saved_model/yolo_cropper" ]; then
    ls -lh /app/saved_model/yolo_cropper/ | grep "$MODEL_TYPE" || echo "Warning: No model file found for $MODEL_TYPE!"
else
    echo "Warning: saved_model directory not found!"
fi

echo "2. Input Images Check:"
ls -1 /app/data/sample/original | head -n 3 || echo "Input dir empty!"

echo "3. Output Crops Check:"
# Output folder name changes dynamically based on the model
if [ -d "/app/data/sample/original_crop/$MODEL_TYPE" ]; then
    ls -1 "/app/data/sample/original_crop/$MODEL_TYPE" | head -n 3 || echo "No crops found in $MODEL_TYPE folder!"
else
    echo "Error: Output directory for $MODEL_TYPE was not created."
fi

echo "--------------------------------------------------------"

echo "========================================================"
echo "   🎉 Demo Pipeline Completed Successfully!             "
echo "========================================================"