#!/bin/bash
set -e

echo "Docker Setting up environment..."

pip install --upgrade pip
pip install -r requirements.txt

mkdir -p third_party

if [ ! -d "third_party/yolov5" ]; then
    git clone https://github.com/ultralytics/yolov5.git third_party/yolov5
fi

if [ ! -d "third_party/darknet" ]; then
    git clone https://github.com/AlexeyAB/darknet.git third_party/darknet
fi

echo "Docker Build setup completed"
