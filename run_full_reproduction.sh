#!/bin/bash
set -e

echo "========================================================"
echo "   Crop for Trust: Full Reproduction Setup              "
echo "   (Linux Host - Legacy Models & Darknet Compilation)   "
echo "========================================================"

# 1. Environment Check
echo -e "\n[1/5] Checking Environment..."
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  Warning: GEMINI_API_KEY is not set. GAC module might fail."
fi

# 2. Install Python Dependencies
echo -e "\n[2/5] Installing Python Dependencies..."
pip install -r requirements.txt
pip install gdown

# 3. Third-party Repositories (Darknet/YOLOv5)
echo -e "\n[3/5] Setting up Third-party Repositories..."
mkdir -p third_party

# --- YOLOv5 ---
if [ ! -d "third_party/yolov5" ]; then
    git clone https://github.com/ultralytics/yolov5.git third_party/yolov5
fi

# --- Darknet (Compile Required) ---
if [ ! -d "third_party/darknet" ]; then
    git clone https://github.com/AlexeyAB/darknet.git third_party/darknet
fi

# 4. Download ALL Weights
echo -e "\n[4/5] Downloading All Model Weights..."
DARKNET_DIR="third_party/darknet"
MODEL_DIR="saved_model/yolo_cropper"
mkdir -p "$MODEL_DIR"

# YOLOv2
if [ ! -f "$DARKNET_DIR/yolov2.weights" ]; then
    wget -q --show-progress -O "$DARKNET_DIR/yolov2.weights" \
        "https://github.com/hank-ai/darknet/releases/download/v2.0/yolov2.weights"
fi
# YOLOv4
if [ ! -f "$DARKNET_DIR/yolov4.conv.137" ]; then
    wget -q --show-progress -O "$DARKNET_DIR/yolov4.conv.137" \
        "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137"
fi

# 5. Run Main Pipeline
echo -e "\n[5/5] Running Main Pipeline (Default: YOLOv8s)..."

python src/main.py --config utils/config.yaml --test on

echo "========================================================"
echo "   Full Reproduction Environment Ready & Ran Successfully! "
echo "   To try legacy models, run:"
echo "   python src/main.py --yolo_model yolov4"
echo "========================================================"