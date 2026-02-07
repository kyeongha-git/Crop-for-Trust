import sys
import pytest
from unittest.mock import patch

from src.main import main

@pytest.fixture
def mock_modules():
    with patch("src.main.AnnotationCleaner") as MockCleaner, \
         patch("src.main.YOLOCropperController") as MockYOLO, \
         patch("src.main.DataAugmentor") as MockAugmentor, \
         patch("src.main.Classifier") as MockClassifier, \
         patch("src.main.ConfigManager") as MockConfigMgr, \
         patch("src.main.setup_logging"), \
         patch("src.main.get_logger"):

        mock_cfg_instance = MockConfigMgr.return_value

        default_config = {
            "main": {
                "annot_clean": True,
                "yolo_crop": True,
                "yolo_model": "yolov5",
                "classify_model": "vgg16"
            },
            "yolo_cropper": {
                "dataset": {"results_dir": "dummy/results"}
            }
        }
        mock_cfg_instance.update_paths.return_value = default_config

        yield {
            "Cleaner": MockCleaner,
            "YOLO": MockYOLO,
            "Augmentor": MockAugmentor,
            "Classifier": MockClassifier,
            "ConfigMgr": MockConfigMgr,
            "defaults": default_config
        }

def test_pipeline_full_execution(mock_modules):
    """
    Scenario: Run without flags (Default).
    Expectation: Steps 1, 2, 3, 4 are ALL executed.
    """
    test_args = ["main.py", "--config", "config.yaml"]
    
    with patch.object(sys, 'argv', test_args):
        main()
        
    # Assertions
    mock_modules["Cleaner"].return_value.run.assert_called_once()  # Step 1
    mock_modules["YOLO"].return_value.run.assert_called_once()     # Step 2
    mock_modules["Augmentor"].return_value.run.assert_called_once()# Step 3
    mock_modules["Classifier"].return_value.run.assert_called_once()# Step 4

def test_pipeline_skip_annotation(mock_modules):
    """
    Scenario: Run with --no_annot_clean.
    Expectation: Step 1 (Cleaner) is skipped, others run.
    """
    test_args = ["main.py", "--no_annot_clean"]
    
    # Update ConfigManager return value to simulate flag effect
    config = mock_modules["defaults"].copy()
    config["main"]["annot_clean"] = False
    mock_modules["ConfigMgr"].return_value.update_paths.return_value = config
    
    with patch.object(sys, 'argv', test_args):
        main()
        
    # Assertions
    mock_modules["Cleaner"].return_value.run.assert_not_called()   # SKIPPED
    mock_modules["YOLO"].return_value.run.assert_called()          # RUN
    mock_modules["Augmentor"].return_value.run.assert_called()     # RUN
    mock_modules["Classifier"].return_value.run.assert_called()    # RUN

def test_pipeline_skip_yolo(mock_modules):
    """
    Scenario: Run with --no_yolo_crop.
    Expectation: Step 2 (YOLOCropper) is skipped, others run.
    """
    test_args = ["main.py", "--no_yolo_crop"]
    
    # Update ConfigManager return value
    config = mock_modules["defaults"].copy()
    config["main"]["yolo_crop"] = False
    mock_modules["ConfigMgr"].return_value.update_paths.return_value = config
    
    with patch.object(sys, 'argv', test_args):
        main()
    
    # Assertions
    mock_modules["YOLO"].return_value.run.assert_not_called()      # SKIPPED
    mock_modules["Cleaner"].return_value.run.assert_called()       # RUN
    mock_modules["Augmentor"].return_value.run.assert_called()     # RUN
    mock_modules["Classifier"].return_value.run.assert_called()    # RUN

def test_argument_passing(mock_modules):
    """
    Scenario: Pass specific model names via CLI.
    Expectation: ConfigManager receives these arguments correctly.
    """
    test_args = ["main.py", "--yolo_model", "yolov8x"]
    
    with patch.object(sys, 'argv', test_args):
        main()
        
    # Verify ConfigManager update_paths was called with correct args
    mock_modules["ConfigMgr"].return_value.update_paths.assert_called_once()
    call_kwargs = mock_modules["ConfigMgr"].return_value.update_paths.call_args.kwargs
    
    assert call_kwargs["yolo_model"] == "yolov8x"