#!/bin/bash
set -e

# 0. Default Configuration
MODEL_TYPE="yolov5"
REPAIR_IMG="img_01.png"   # [Default] 자네 파일명에 맞춤
REPLACE_IMG="img_01.jpg"

# 1. Argument Parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --yolo_model) MODEL_TYPE="$2"; shift ;;
        --repair_image) REPAIR_IMG="$2"; shift ;;
        --replace_image) REPLACE_IMG="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================================"
echo "   Crop for Trust: Reliability-Aware Pipeline Demo      "
echo "          (Optimization: Partial GenAI Execution)       "
echo "========================================================"

if [ -z "$GEMINI_API_KEY" ]; then
    echo "[Error] GEMINI_API_KEY is missing!"
    exit 1
fi

# =================================================================
# 3. Sandbox Setup (Physical Data Preparation)
# =================================================================
SOURCE_ROOT="/app/data/sample"
SANDBOX_ROOT="/app/data/demo_run"

rm -rf "$SANDBOX_ROOT"
mkdir -p "$SANDBOX_ROOT"

mkdir -p "$SANDBOX_ROOT/original/repair"
mkdir -p "$SANDBOX_ROOT/original/replace"
mkdir -p "$SANDBOX_ROOT/annotation_cleaner/only_annotation_image/repair"
mkdir -p "$SANDBOX_ROOT/annotation_cleaner/only_annotation_image/replace"

echo "[Shell] 📦 Copying files with Optimization Strategy..."

# (1) [REPAIR] 
if [ -f "$SOURCE_ROOT/original/repair/$REPAIR_IMG" ]; then
    cp "$SOURCE_ROOT/original/repair/$REPAIR_IMG" \
       "$SANDBOX_ROOT/original/repair/$REPAIR_IMG"
       
else
    echo "[Error] Repair image not found: $REPAIR_IMG"
    exit 1
fi

# (2) [REPLACE]
if [ -f "$SOURCE_ROOT/original/replace/$REPLACE_IMG" ]; then
    cp "$SOURCE_ROOT/original/replace/$REPLACE_IMG" \
       "$SANDBOX_ROOT/original/replace/$REPLACE_IMG"

    if [ -f "$SOURCE_ROOT/annotation_cleaner/only_annotation_image/replace/$REPLACE_IMG" ]; then
        cp "$SOURCE_ROOT/annotation_cleaner/only_annotation_image/replace/$REPLACE_IMG" \
           "$SANDBOX_ROOT/annotation_cleaner/only_annotation_image/replace/$REPLACE_IMG"
    else
        cp "$SOURCE_ROOT/original/replace/$REPLACE_IMG" \
           "$SANDBOX_ROOT/annotation_cleaner/only_annotation_image/replace/$REPLACE_IMG"
    fi
else
    echo "[Error] Replace image not found: $REPLACE_IMG"
    exit 1
fi

# =================================================================
# 4. Run Pipeline
# =================================================================
echo -e "\n[Pipeline] Starting src/main.py..."

python src/main.py \
    --yolo_model "$MODEL_TYPE" \
    --config utils/config.yaml \
    --data_dir "$SANDBOX_ROOT"

echo "[Shell] Pipeline Finished."