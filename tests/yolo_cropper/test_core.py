#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_core.py
-------------
Unit and integration tests for the YOLO Cropper module.

Modules tested:
- YOLOConverter: Converts YOLO detection results into JSON format.
- YOLOCropper: Crops images based on detection results and copies originals if needed.

Test Focus:
- Validating dataset and detection folder structures.
- Ensuring correct parsing and JSON aggregation.
- Handling missing folders or files gracefully with proper exceptions.
- Verifying cropping and copying logic for detection and non-detection cases.
"""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.yolo_cropper.core.converter import (YOLOConverter,
                                             infer_class_from_folder)
from src.yolo_cropper.core.cropper import YOLOCropper

# ==============================================================
# Fixtures
# ==============================================================


@pytest.fixture
def tmp_dataset(tmp_path):
    """
    Create a temporary dataset directory for testing.

    Structure:
        tmp_path/
            data/
                original/
                    repair/
                        img0.png
                        img1.png
                    replace/
                        img0.png
                        img1.png

    Returns:
        Path: Root directory of the temporary dataset.
    """
    data_root = tmp_path / "data" / "original"
    (data_root / "repair").mkdir(parents=True)
    (data_root / "replace").mkdir(parents=True)

    # Create dummy grayscale images (100x100)
    import cv2

    for cls in ["repair", "replace"]:
        for i in range(2):
            img = np.full(
                (100, 100, 3), 127 if cls == "repair" else 200, dtype=np.uint8
            )
            cv2.imwrite(str(data_root / cls / f"img{i}.png"), img)

    return data_root


@pytest.fixture
def tmp_detect_root(tmp_path):
    """
    Create a mock YOLO detection output directory for testing.

    Structure:
        tmp_path/
            runs/
                detect/
                    yolov5_repair/
                        labels/
                            img0.txt
                            img1.txt
                    yolov5_replace/
                        labels/
                            img0.txt
    """
    detect_root = tmp_path / "runs" / "detect"
    (detect_root / "yolov5_repair" / "labels").mkdir(parents=True)
    (detect_root / "yolov5_replace" / "labels").mkdir(parents=True)

    # repair labels
    (detect_root / "yolov5_repair" / "labels" / "img0.txt").write_text(
        "0 0.5 0.5 0.2 0.2 0.9\n"
    )
    (detect_root / "yolov5_repair" / "labels" / "img1.txt").write_text(
        "0 0.4 0.4 0.3 0.3 0.8\n"
    )

    # replace labels
    (detect_root / "yolov5_replace" / "labels" / "img0.txt").write_text(
        "0 0.6 0.6 0.2 0.2 0.7\n"
    )

    # Add mock YOLO output images
    for cls in ["yolov5_repair", "yolov5_replace"]:
        for i in range(2):
            (detect_root / cls / f"img{i}.jpg").write_text("mock_image_data")

    return detect_root


@pytest.fixture
def config_template(tmp_dataset, tmp_detect_root, tmp_path):
    """
    Provide a minimal YOLO Cropper configuration dictionary mimicking config.yaml.

    Returns:
        dict: Minimal configuration for YOLOConverter and YOLOCropper.
    """
    return {
        "yolo_cropper": {
            "main": {"model_name": "yolov5", "input_dir": str(tmp_dataset)},
            "dataset": {
                "detect_dir": str(tmp_detect_root),
                "results_dir": str(tmp_path / "outputs" / "json_results"),
            },
        }
    }


@pytest.fixture
def tmp_results_json(tmp_path, tmp_dataset):
    """
    Create mock `result.json` and `predict.txt` files for YOLOCropper testing.

    Returns:
        dict: Paths to generated JSON, predict.txt, and root output directory.
    """
    results_dir = tmp_path / "outputs" / "json_results" / "yolov5"
    results_dir.mkdir(parents=True, exist_ok=True)

    result_data = [
        {
            "filename": str(tmp_dataset / "repair" / "img0.png"),
            "objects": [
                {
                    "relative_coordinates": {
                        "center_x": 0.5,
                        "center_y": 0.5,
                        "width": 0.4,
                        "height": 0.4,
                    },
                    "confidence": 0.9,
                    "class_id": 0,
                    "name": "repair",
                }
            ],
        },
        {
            "filename": str(tmp_dataset / "replace" / "img1.png"),
            "objects": [],
        },
    ]
    json_path = results_dir / "result.json"
    json_path.write_text(
        json.dumps(result_data, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    predict_txt = results_dir.parent / "predict.txt"
    all_imgs = sorted(str(p) for p in tmp_dataset.rglob("*.png"))
    predict_txt.write_text("\n".join(all_imgs), encoding="utf-8")

    return {"json": json_path, "predict": predict_txt, "root": results_dir.parent}


# ==============================================================
# Unit Tests for YOLOConverter
# ==============================================================


def test_infer_class_from_folder():
    """Test that infer_class_from_folder() correctly identifies class names from paths."""
    repair_path = Path("/some/path/repair/images")
    replace_path = Path("/another/replace/set")
    unknown_path = Path("/no/class/here")

    assert infer_class_from_folder(repair_path)["name"] == "repair"
    assert infer_class_from_folder(replace_path)["name"] == "replace"
    assert infer_class_from_folder(unknown_path)["name"] == "unknown"


def test_parse_detect_folder_parses_valid_labels(
    tmp_dataset, tmp_detect_root, config_template
):
    """Ensure _parse_detect_folder() correctly parses YOLO label files and returns valid JSON objects."""
    converter = YOLOConverter(config_template)
    detect_dir = tmp_detect_root / "yolov5_repair"

    results, next_frame = converter._parse_detect_folder(detect_dir)
    assert isinstance(results, list)
    assert len(results) == 2
    assert next_frame == 3
    assert all("objects" in r for r in results)
    assert "filename" in results[0]
    assert "relative_coordinates" in results[0]["objects"][0]


def test_run_creates_result_json(tmp_dataset, tmp_detect_root, config_template):
    """Test that YOLOConverter.run() aggregates results and creates a valid result.json file."""
    converter = YOLOConverter(config_template)
    converter.run()

    output_json = converter.output_json
    assert output_json.exists(), "result.json should be created"
    data = json.loads(output_json.read_text(encoding="utf-8"))

    assert isinstance(data, list)
    assert len(data) > 0
    first_item = data[0]
    assert "filename" in first_item
    assert "objects" in first_item
    assert isinstance(first_item["objects"], list)
    assert "relative_coordinates" in first_item["objects"][0]


def test_run_raises_if_no_detect_folder(tmp_path, config_template):
    """
    Verify that YOLOConverter.run() raises FileNotFoundError
    when no valid detection folders are found.
    """
    cfg = copy.deepcopy(config_template)

    empty_detect = tmp_path / "runs" / "detect_empty"
    empty_detect.mkdir(parents=True, exist_ok=True)
    cfg["yolo_cropper"]["dataset"]["detect_dir"] = str(empty_detect)

    converter = YOLOConverter(cfg)

    assert (
        list(converter.detect_root.iterdir()) == []
    ), f"Detect root not empty: {list(converter.detect_root.iterdir())}"

    with pytest.raises(FileNotFoundError) as excinfo:
        converter.run()
        
    assert "No detection folders found" in str(
        excinfo.value
    ), f"Unexpected error message: {excinfo.value}"


# ==============================================================
# Unit Tests for YOLOCropper
# ==============================================================


def test_cropper_creates_crops_and_originals(
    tmp_path, config_template, tmp_results_json
):
    """Test that YOLOCropper crops detected images and copies originals for missing detections."""
    config_template["yolo_cropper"]["dataset"]["results_dir"] = str(
        tmp_results_json["root"]
    )

    cropper = YOLOCropper(config_template)
    cropper.crop_from_json()

    out_dir = cropper.output_dir
    assert out_dir.exists(), "Output directory must exist after cropping."

    repair_dir = out_dir / "repair"
    replace_dir = out_dir / "replace"
    repair_files = list(repair_dir.glob("*.jpg"))
    replace_files = list(replace_dir.glob("*.png"))

    assert any(
        "_1" in f.name for f in repair_files
    ), "Cropped file must contain '_1' suffix"
    assert any(
        "img1.png" in f.name for f in replace_files
    ), "No-detection file must be copied"

    img = cv2.imread(str(repair_files[0]))
    assert img is not None and img.size > 0, "Cropped image should be valid."


def test_cropper_raises_if_json_missing(tmp_path, config_template, tmp_results_json):
    """Test that FileNotFoundError is raised when result.json is missing."""
    config_template["yolo_cropper"]["dataset"]["results_dir"] = str(
        tmp_results_json["root"]
    )
    cropper = YOLOCropper(config_template)
    cropper.json_path.unlink()

    with pytest.raises(FileNotFoundError):
        cropper.crop_from_json()


def test_cropper_raises_if_predict_missing(tmp_path, config_template, tmp_results_json):
    """Test that FileNotFoundError is raised when predict.txt is missing."""
    config_template["yolo_cropper"]["dataset"]["results_dir"] = str(
        tmp_results_json["root"]
    )
    cropper = YOLOCropper(config_template)
    cropper.predict_list.unlink()

    with pytest.raises(FileNotFoundError):
        cropper.crop_from_json()


def test_cropper_copies_missing_images(tmp_path, config_template, tmp_results_json):
    """Test that missing images in result.json are still copied from predict.txt."""
    config_template["yolo_cropper"]["dataset"]["results_dir"] = str(
        tmp_results_json["root"]
    )
    cropper = YOLOCropper(config_template)

    partial_data = json.loads(cropper.json_path.read_text(encoding="utf-8"))[:1]
    cropper.json_path.write_text(
        json.dumps(partial_data, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    cropper.crop_from_json()

    out_dir = cropper.output_dir
    copied = list(out_dir.rglob("*.png")) + list(out_dir.rglob("*.jpg"))
    assert len(copied) > 0, "Missing images should be copied to output folder."
