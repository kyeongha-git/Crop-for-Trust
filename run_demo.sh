#!/bin/bash
set -e

echo "========================================================"
echo "   Crop for Trust: Reliability-Aware Pipeline Demo      "
echo "          (Docker Environment)                          "
echo "========================================================"


# 1. API Key Check
if [ -z "$GEMINI_API_KEY" ]; then
    echo "[ERROR] GEMINI_API_KEY is missing!"
    echo "        Please run with: -e GEMINI_API_KEY=YOUR_KEY"
    exit 1
fi

echo "[INFO] GEMINI_API_KEY detected."


# 2. Basic Sanity Check (Framework-level only)
if [ ! -d "src" ]; then
    echo "[ERROR] src/ directory not found!"
    exit 1
fi

if [ ! -f "utils/config_docker.yaml" ]; then
    echo "[ERROR] utils/config_docker.yaml not found!"
    exit 1
fi

echo "[INFO] Project structure sanity check passed."

# 3. Run Pipeline (CLI arguments are forwarded as-is)
echo -e "\n[Pipeline] Launching src/main.py ..."
echo "[INFO] Forwarded CLI arguments: $@"
echo "--------------------------------------------------------"

exec python src/main.py --config utils/config_docker.yaml "$@"