import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.yolo_cropper.models.darknet.train import DarknetTrainer
from src.yolo_cropper.models.yolov5.train import YOLOv5Trainer
from src.yolo_cropper.models.yolov8.train import YOLOv8Trainer

@pytest.fixture
def mock_config(tmp_path):
    """
    Creates a master configuration covering all model types.
    """
    return {
        "main": {
            "model_name": "yolov5", # Default
            "input_dir": str(tmp_path / "data/original")
        },
        "yolo_cropper": {
            "main": {"model_name": "yolov5"},
            "yolov5": {
                "yolov5_dir": str(tmp_path / "yolov5_repo"),
                "data_yaml": str(tmp_path / "data.yaml")
            },
            "darknet": {
                "darknet_dir": str(tmp_path / "darknet_repo"),
            },
            "yolov8": {
                "data_yaml": str(tmp_path / "yolov8_data.yaml")
            },
            "train": {
                "epochs": 100,
                "batch_size": 16,
                "batch": 16, # for yolov8
                "imgsz": 416,
                "device": "cpu"
            },
            "dataset": {
                "checkpoints_dir": str(tmp_path / "checkpoints"),
                "saved_model_dir": str(tmp_path / "saved_models"),
                "train_output_dir": str(tmp_path / "runs/train")
            }
        }
    }

@pytest.fixture
def env_setup(mock_config):
    """Creates necessary dummy directories to pass __init__ validation."""
    for path_key in [
        ("yolov5", "yolov5_dir"), ("darknet", "darknet_dir"), 
        ("dataset", "checkpoints_dir"), ("dataset", "saved_model_dir")
    ]:
        section, key = path_key
        p = Path(mock_config["yolo_cropper"][section][key])
        p.mkdir(parents=True, exist_ok=True)
        
    yolov5_yaml = Path(mock_config["yolo_cropper"]["yolov5"]["data_yaml"])
    yolov5_yaml.parent.mkdir(parents=True, exist_ok=True)
    yolov5_yaml.write_text("train: ./train\nval: ./val", encoding="utf-8")

    # YOLOv8 Data YAML
    yolov8_yaml = Path(mock_config["yolo_cropper"]["yolov8"]["data_yaml"])
    yolov8_yaml.parent.mkdir(parents=True, exist_ok=True)
    yolov8_yaml.write_text("train: ./train\nval: ./val", encoding="utf-8")

# 1. Darknet Tests (Subprocess Verification)

@patch("subprocess.run")
@patch("pathlib.Path.glob")
def test_darknet_yolov4_command(mock_glob, mock_subprocess, mock_config, env_setup):
    """Test YOLOv4 training command construction."""
    # Setup
    mock_config["yolo_cropper"]["main"]["model_name"] = "yolov4"
    mock_subprocess.return_value.returncode = 0
    mock_glob.return_value = iter([]) # No best weights found

    # Execute
    trainer = DarknetTrainer(mock_config)
    trainer.run()

    # Assert
    args, _ = mock_subprocess.call_args
    cmd = args[0][2] # bash -lc 'COMMAND' -> index 2 is the actual command
    
    assert "./darknet detector train" in cmd
    assert "cfg/yolov4-obj.cfg" in cmd
    assert "yolov4.conv.137" in cmd # Correct weights for v4
    assert "-clear" not in cmd # YOLOv4 should NOT have clear flag

@patch("subprocess.run")
@patch("pathlib.Path.glob")
def test_darknet_yolov2_command(mock_glob, mock_subprocess, mock_config, env_setup):
    """Test YOLOv2 training command construction (Check for -clear flag)."""
    # Setup
    mock_config["yolo_cropper"]["main"]["model_name"] = "yolov2"
    mock_subprocess.return_value.returncode = 0
    mock_glob.return_value = iter([])

    # Execute
    trainer = DarknetTrainer(mock_config)
    trainer.run()

    # Assert
    args, _ = mock_subprocess.call_args
    cmd = args[0][2]
    
    assert "cfg/yolov2-obj.cfg" in cmd
    assert "yolov2.weights" in cmd
    assert "-clear" in cmd #  YOLOv2 specific


# 2. YOLOv5 Tests (Subprocess Verification)

@patch("subprocess.run")
@patch("pathlib.Path.rglob") # for best.pt search
def test_yolov5_train_command(mock_rglob, mock_subprocess, mock_config, env_setup):
    """Test YOLOv5 training command list construction."""
    mock_config["yolo_cropper"]["main"]["model_name"] = "yolov5"
    mock_subprocess.return_value.returncode = 0
    mock_rglob.return_value = iter([])

    trainer = YOLOv5Trainer(mock_config)
    trainer.run()

    args, kwargs = mock_subprocess.call_args
    cmd_list = args[0]

    assert cmd_list[0] == "python"
    assert cmd_list[1] == "train.py"
    assert "--epochs" in cmd_list
    assert str(cmd_list[cmd_list.index("--epochs")+1]) == "100"
    assert kwargs["cwd"] == Path(mock_config["yolo_cropper"]["yolov5"]["yolov5_dir"])


# 3. YOLOv8 Tests (API Call Verification)

@patch("src.yolo_cropper.models.yolov8.train.YOLO") # Mock the Ultralytics class
@patch("shutil.copy2") # Prevent file copy
def test_yolov8_train_api_call(mock_copy, mock_yolo_class, mock_config, env_setup):
    """
    Test YOLOv8Trainer calls the ultralytics API with correct arguments.
    """
    # Setup
    mock_config["yolo_cropper"]["main"]["model_name"] = "yolov8s"
    
    # Mock the model instance and the train return value
    mock_model_instance = mock_yolo_class.return_value
    mock_results = MagicMock()
    mock_results.save_dir = Path("runs/train/exp") # Fake output dir
    mock_model_instance.train.return_value = mock_results

    # Execute
    trainer = YOLOv8Trainer(mock_config)
    trainer.run()

    # Assertions
    # 1. Was YOLO('yolov8s.pt') initialized?
    mock_yolo_class.assert_called_once_with("yolov8s.pt")
    
    # 2. Was model.train() called with config values?
    mock_model_instance.train.assert_called_once()
    call_kwargs = mock_model_instance.train.call_args.kwargs
    
    assert call_kwargs["epochs"] == 100
    assert call_kwargs["batch"] == 16
    assert call_kwargs["imgsz"] == 416
    assert call_kwargs["project"] == str(Path(mock_config["yolo_cropper"]["dataset"]["train_output_dir"]).resolve())
    assert call_kwargs["data"] == str(Path(mock_config["yolo_cropper"]["yolov8"]["data_yaml"]).resolve())