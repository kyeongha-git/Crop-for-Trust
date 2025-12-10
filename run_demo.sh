#!/bin/bash
set -e

echo "========================================================"
echo "   Crop for Trust: Reliability-Aware Pipeline Demo      "
echo "   (Docker Environment - YOLOv8s Fine-tuned Only)       "
echo "========================================================"

# 1. API Key Check
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ Error: GEMINI_API_KEY is missing!"
    exit 1
fi

# 2. Lazy Download: Fine-tuned YOLOv8s Only
MODEL_DIR="/app/saved_model/yolo_cropper"
MODEL_PATH="$MODEL_DIR/yolov8s.pt"
FILE_ID="1eNZNze7uYNEXsdsn14lrUZ4dehwYbCWA" 

echo -e "\n[Setup] Checking Model Weights..."
if [ ! -f "$MODEL_PATH" ]; then
    echo "   - Downloading YOLOv8s from Google Drive..."
    gdown --id "$FILE_ID" -O "$MODEL_PATH" || { echo "Download failed!"; exit 1; }
else
    echo "   Found existing weights."
fi

# 3. Run Pipeline
echo -e "\n[Pipeline] Starting src/main.py..."
python src/main.py

echo "========================================================"
echo "   🎉 Demo Pipeline Completed Successfully!             "
echo "========================================================"