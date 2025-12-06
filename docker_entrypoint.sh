#!/bin/bash
set -e

echo "Starting Crop-Conquer"

# 1️⃣ Move to project root
cd /app

# 2️⃣ Check API KEY
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set!"
    exit 1
fi

# 3️⃣ Darknet weights
DARKNET_DIR="third_party/darknet"
mkdir -p "$DARKNET_DIR"

if [ ! -f "$DARKNET_DIR/yolov2.weights" ]; then
    echo "Downloading YOLOv2 weights..."
    wget -O "$DARKNET_DIR/yolov2.weights" \
        "https://github.com/hank-ai/darknet/releases/download/v2.0/yolov2.weights"
fi

if [ ! -f "$DARKNET_DIR/yolov4.conv.137" ]; then
    echo "Downloading YOLOv4 pretrained weights..."
    wget -O "$DARKNET_DIR/yolov4.conv.137" \
        "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137"
fi

# 4️⃣ YOLOv8 weights
MODEL_DIR="saved_model/yolo_cropper"
mkdir -p "$MODEL_DIR"
YOLO_PATH="$MODEL_DIR/yolov8s.pt"

if ! command -v gdown &> /dev/null; then 
    pip install gdown 
fi

if [ ! -f "$YOLO_PATH" ]; then
    echo "Downloading YOLOv8s weights..."
    gdown --id "1eNZNze7uYNEXsdsn14lrUZ4dehwYbCWA" -O "$YOLO_PATH"
fi

# 5️⃣ run
python src/main.py "$@"

echo "Pipeline Done!"
