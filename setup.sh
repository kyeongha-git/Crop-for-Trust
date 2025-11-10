#!/bin/bash
# ===============================================
# 🔧 Project Setup Script
# Automatically sets up environment for the pipeline
# ===============================================

echo "🚀 Setting up project environment..."

# 1️⃣ Conda Environment Setup
if command -v conda &> /dev/null
then
    echo "🟢 Creating Conda environment: tf_env"
    conda create -n tf_env python=3.10 -y
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate tf_env
else
    echo "⚠️ Conda not found. Using system Python environment."
fi

# 2️⃣ Install Python Dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3️⃣ Create third_party directory if not exist
mkdir -p third_party

# 4️⃣ Clone YOLOv5 Repository
echo "📂 Setting up YOLOv5..."
if [ ! -d "third_party/yolov5" ]; then
    git clone https://github.com/ultralytics/yolov5.git third_party/yolov5
else
    echo "✅ YOLOv5 already exists. Skipping clone."
fi

# 5️⃣ Clone Darknet (YOLOv2 / YOLOv4)
echo "🧱 Setting up Darknet..."
if [ ! -d "third_party/darknet" ]; then
    git clone https://github.com/AlexeyAB/darknet.git third_party/darknet
    echo "⚙️ Darknet cloned successfully. Build will be managed by makemanager.py."
else
    echo "✅ Darknet already exists. Skipping clone."
fi

# 6️⃣ Download YOLO Weights (YOLOv2 / YOLOv4)
echo "🎯 Downloading YOLO pretrained weights..."
DARKNET_DIR="third_party/darknet"
mkdir -p "$DARKNET_DIR"

# YOLOv2 weights
if [ ! -f "$DARKNET_DIR/yolov2.weights" ]; then
    echo "⬇️ Downloading YOLOv2 weights..."
    wget -O "$DARKNET_DIR/yolov2.weights" "https://github.com/hank-ai/darknet/releases/download/v2.0/yolov2.weights"
else
    echo "✅ YOLOv2 weights already exist. Skipping download."
fi

# YOLOv4 pretrained weights
if [ ! -f "$DARKNET_DIR/yolov4.conv.137" ]; then
    echo "⬇️ Downloading YOLOv4 pretrained weights..."
    wget -O "$DARKNET_DIR/yolov4.conv.137" "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137"
else
    echo "✅ YOLOv4 pretrained weights already exist. Skipping download."
fi

echo "🎉 Setup complete!"
echo "➡️ To activate environment, run:"
echo "   conda activate tf_env"
