import json
import pytest
import cv2
import numpy as np
from pathlib import Path

from src.yolo_cropper.core.converter import YOLOConverter
from src.yolo_cropper.core.cropper import YOLOCropper
from src.yolo_cropper.core.make_predict import YOLOPredictListGenerator

@pytest.fixture
def core_env(tmp_path):
    """
    Sets up a shared environment for Core Utils tests.
    Structure:
      root/
        data/
          class_A/ (contains sample.jpg)
        runs/detect/yolov5_exp/labels/ (contains sample.txt)
        output/
    """
    root_dir = tmp_path / "core_test"
    data_dir = root_dir / "data"
    detect_dir = root_dir / "runs" / "detect" / "yolov5_exp"
    output_dir = root_dir / "output"
    
    # Create Dirs
    (data_dir / "class_A").mkdir(parents=True, exist_ok=True)
    (detect_dir / "labels").mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create Dummy Image (100x100)
    img_path = data_dir / "class_A" / "sample.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (60, 60), (255, 255, 255), -1) 
    cv2.imwrite(str(img_path), img)
    
    # 2. Create Dummy Label (YOLO format: class x y w h)
    # Center box: 0.5, 0.5, 0.2, 0.2 -> (50, 50, 20, 20) in pixels
    label_path = detect_dir / "labels" / "sample.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.2 0.95", encoding="utf-8")
    
    # 3. Create Config
    config = {
        "main": {
            "categories": ["class_A", "class_B"],
            "input_dir": str(data_dir)
        },
        "yolo_cropper": {
            "main": {
                "model_name": "yolov5",
                "input_dir": str(data_dir) # Explicitly set for make_predict
            },
            "dataset": {
                "detect_dir": str(root_dir / "runs" / "detect"),
                "results_dir": str(output_dir)
            },
            "cropper": {
                "min_size": 5,
                "pad": 0
            }
        }
    }
    
    return config, root_dir, img_path

def test_predict_list_generator(core_env):
    """
    Verify YOLOPredictListGenerator creates a correct predict.txt file.
    """
    config, root_dir, img_path = core_env
    
    # Initialize Generator
    generator = YOLOPredictListGenerator(config)
    generator.run()
    
    # Check Output File
    output_path = Path(config["yolo_cropper"]["dataset"]["results_dir"]) / "yolov5" / "predict.txt"
    assert output_path.exists(), "predict.txt was not generated"
    
    # Check Content
    content = output_path.read_text(encoding="utf-8").strip()
    expected_path = str(img_path.resolve())
    
    # It should contain the absolute path to the image
    assert content == expected_path

def test_converter_logic(core_env):
    """
    Verify YOLOConverter correctly parses label files into JSON.
    """
    config, root_dir, img_path = core_env
    
    converter = YOLOConverter(config)
    converter.run()
    
    # Check JSON Output
    json_path = Path(config["yolo_cropper"]["dataset"]["results_dir"]) / "yolov5" / "result.json"
    assert json_path.exists(), "Result JSON not created"
    
    with open(json_path) as f:
        data = json.load(f)
        
    assert len(data) == 1
    item = data[0]
    assert item["filename"] == str(img_path.resolve())
    assert len(item["objects"]) == 1
    
    obj = item["objects"][0]
    assert obj["class_id"] == 0
    assert obj["relative_coordinates"]["center_x"] == 0.5
    assert obj["confidence"] == 0.95

def test_cropper_logic(core_env):
    """
    Verify YOLOCropper correctly crops images based on JSON coordinates.
    """
    config, root_dir, img_path = core_env
    
    # Prepare JSON & Predict list (Manual setup to isolate test)
    json_dir = Path(config["yolo_cropper"]["dataset"]["results_dir"]) / "yolov5"
    json_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = json_dir / "result.json"
    predict_path = json_dir / "predict.txt"
    
    mock_data = [{
        "filename": str(img_path),
        "objects": [{
            "class_id": 0,
            "relative_coordinates": {"center_x": 0.5, "center_y": 0.5, "width": 0.2, "height": 0.2}
        }]
    }]
    
    with open(json_path, "w") as f:
        json.dump(mock_data, f)
        
    predict_path.write_text(str(img_path), encoding="utf-8")
    
    # Run Cropper (Redirect output)
    crop_out = root_dir / "crop_output"
    config["yolo_cropper"]["main"]["output_dir"] = str(crop_out)
    
    cropper = YOLOCropper(config)
    cropper.run()
    
    # Check Output
    expected_crop = crop_out / "class_A" / "sample_1.jpg"
    assert expected_crop.exists(), "Cropped image not found"
    
    # Check Size (20x20) & Content (White center)
    crop_img = cv2.imread(str(expected_crop))
    h, w = crop_img.shape[:2]
    
    assert abs(h - 20) <= 1
    assert abs(w - 20) <= 1
    assert np.all(crop_img[10, 10] == 255)

def test_cropper_no_detection_copy(core_env):
    """
    Verify that images with no detections are copied as-is.
    """
    config, root_dir, img_path = core_env
    
    json_dir = Path(config["yolo_cropper"]["dataset"]["results_dir"]) / "yolov5"
    json_dir.mkdir(parents=True, exist_ok=True)
    
    # Empty result list, but image exists in predict.txt
    json_path = json_dir / "result.json"
    predict_path = json_dir / "predict.txt"
    
    with open(json_path, "w") as f:
        json.dump([], f)
        
    predict_path.write_text(str(img_path), encoding="utf-8")
    
    # Run
    crop_out = root_dir / "crop_output_empty"
    config["yolo_cropper"]["main"]["output_dir"] = str(crop_out)
    
    cropper = YOLOCropper(config)
    cropper.run()
    
    # Check Copy
    expected_copy = crop_out / "class_A" / "sample.jpg"
    assert expected_copy.exists()
    
    copy_img = cv2.imread(str(expected_copy))
    assert copy_img.shape[:2] == (100, 100)