#!/bin/bash
set -e

echo "========================================================"
echo "   Crop for Trust: Full Reproduction Setup"
echo "   (Linux Host - Legacy Models & Darknet Compilation)"
echo "========================================================"

# ---------------------------------------------------------
# 1. Environment Check
# ---------------------------------------------------------
echo -e "\n[1/4] Checking Environment..."
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  Warning: GEMINI_API_KEY is NOT set."
    echo "    GAC module will not work until you run:"
    echo "    export GEMINI_API_KEY=your_key_here"
else
    echo "GEMINI_API_KEY detected."
fi

# ---------------------------------------------------------
# 2. Install Python Dependencies
# ---------------------------------------------------------
echo -e "\n[2/4] Installing Python Dependencies..."
pip install --upgrade pip

# Important: do NOT include torch/vision installs here manually.
# Your requirements.txt ALREADY includes the correct versions.
pip install -r requirements.txt

# ---------------------------------------------------------
# 3. Setup Third-party Repositories
# ---------------------------------------------------------
echo -e "\n[3/4] Setting up Third-party Repositories..."
mkdir -p third_party

# --- YOLOv5 ---
if [ ! -d "third_party/yolov5" ]; then
    echo " - Cloning YOLOv5..."
    git clone https://github.com/ultralytics/yolov5.git third_party/yolov5
else
    echo " - YOLOv5 already exists. Skipping."
fi

# --- Darknet ---
if [ ! -d "third_party/darknet" ]; then
    echo " - Cloning Darknet..."
    git clone https://github.com/AlexeyAB/darknet.git third_party/darknet
else
    echo " - Darknet already exists. Skipping."
fi

# ---------------------------------------------------------
# 4. Download Pretrained Weights
# ---------------------------------------------------------
echo -e "\n[4/4] Downloading Darknet Pretrained Weights..."
DARKNET_DIR="third_party/darknet"

# YOLOv2
if [ ! -f "$DARKNET_DIR/yolov2.weights" ]; then
    echo " - Downloading yolov2.weights..."
    wget -q --show-progress -O "$DARKNET_DIR/yolov2.weights" \
        "https://github.com/hank-ai/darknet/releases/download/v2.0/yolov2.weights"
else
    echo " - yolov2.weights already exists."
fi

# YOLOv4
if [ ! -f "$DARKNET_DIR/yolov4.conv.137" ]; then
    echo " - Downloading yolov4.conv.137..."
    wget -q --show-progress -O "$DARKNET_DIR/yolov4.conv.137" \
        "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137"
else
    echo " - yolov4.conv.137 already exists."
fi

echo ""
echo "Darknet Weights Installed:"
echo " - $DARKNET_DIR/yolov2.weights"
echo " - $DARKNET_DIR/yolov4.conv.137"

echo ""
echo "========================================================"
echo " SETUP COMPLETE!"
echo " You can now run the full pipeline manually:"
echo ""
echo "   export GEMINI_API_KEY=your_key_here"
echo "   python src/main.py --config utils/config.yaml"
echo ""
echo "========================================================"
