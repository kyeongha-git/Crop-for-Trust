#!/bin/bash
set -e

echo "========================================================"
echo "   Crop for Trust: Reliability-Aware Pipeline Demo      "
echo "   (Docker Environment - YOLOv8s Fine-tuned Only)       "
echo "========================================================"

# 1. API Key Check
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY is missing!"
    exit 1
fi

# 2. Run Pipeline
echo -e "\n[Pipeline] Starting src/main.py..."
python src/main.py --config utils/config_docker.yaml

echo "========================================================"
echo "   🎉 Demo Pipeline Completed Successfully!             "
echo "========================================================"