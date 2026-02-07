import pytest
from unittest.mock import MagicMock, patch

from src.yolo_cropper.yolo_cropper import YOLOCropperController

@pytest.fixture
def mock_config():
    """Returns a basic configuration structure."""
    return {
        "yolo_cropper": {
            "main": {
                "model_name": "yolov5" # Default
            }
        }
    }

@patch("src.yolo_cropper.yolo_cropper.ConfigManager")
@patch("src.yolo_cropper.yolo_cropper.load_yaml_config")
def test_controller_initialization(mock_load, mock_cfg_manager, mock_config, tmp_path):
    """
    Verify that the Controller initializes ConfigManager and updates paths.
    """
    # Setup Mocks
    mock_load.return_value = mock_config
    
    # Run Init
    controller = YOLOCropperController(config_path="dummy_config.yaml")
    
    # Assertions
    # 1. Check ConfigManager usage
    mock_cfg_manager.assert_called_once_with("dummy_config.yaml")
    mock_cfg_manager.return_value.update_paths.assert_called_once()
    mock_cfg_manager.return_value.save.assert_called_once()
    
    # 2. Check internal state
    assert controller.model_name == "yolov5"

@pytest.mark.parametrize("model_name, expected_module, expected_class", [
    ("yolov5", "src.yolo_cropper.models.yolov5.yolov5", "YOLOv5Pipeline"),
    ("yolov5s", "src.yolo_cropper.models.yolov5.yolov5", "YOLOv5Pipeline"),
    ("yolov8n", "src.yolo_cropper.models.yolov8.yolov8", "YOLOv8Pipeline"),
    ("yolov8x", "src.yolo_cropper.models.yolov8.yolov8", "YOLOv8Pipeline"),
    ("yolov4", "src.yolo_cropper.models.darknet.darknet", "DarknetPipeline"),
    ("yolov2", "src.yolo_cropper.models.darknet.darknet", "DarknetPipeline"),
])
def test_controller_dispatch_logic(
    model_name, expected_module, expected_class, mock_config
):
    """
    Verify that the Controller dynamically imports the correct module and class
    based on the model_name in the config.
    """
    # Update config for this test case
    mock_config["yolo_cropper"]["main"]["model_name"] = model_name

    # We need to patch:
    # 1. ConfigManager & load_yaml_config (Initialization)
    # 2. importlib.import_module (Run logic)
    with patch("src.yolo_cropper.yolo_cropper.ConfigManager"), \
         patch("src.yolo_cropper.yolo_cropper.load_yaml_config", return_value=mock_config), \
         patch("importlib.import_module") as mock_import:

        # Setup Mock Module & Pipeline Class
        mock_module = MagicMock()
        mock_pipeline_class = MagicMock()
        mock_pipeline_instance = MagicMock()
        
        # Connect the chain: import -> module -> class -> instance
        mock_import.return_value = mock_module
        setattr(mock_module, expected_class, mock_pipeline_class)
        mock_pipeline_class.return_value = mock_pipeline_instance

        # Initialize & Run
        controller = YOLOCropperController(config_path="dummy.yaml")
        controller.run()

        # Assertions
        # 1. Was the correct module imported?
        mock_import.assert_called_once_with(expected_module)
        
        # 2. Was the pipeline initialized?
        mock_pipeline_class.assert_called_once_with(config_path="dummy.yaml")
        
        # 3. Was the run method called?
        mock_pipeline_instance.run.assert_called_once_with()

def test_controller_unsupported_model(mock_config):
    """Verify that ValueError is raised for unsupported models."""
    mock_config["yolo_cropper"]["main"]["model_name"] = "resnet50" # Not a YOLO

    with patch("src.yolo_cropper.yolo_cropper.ConfigManager"), \
         patch("src.yolo_cropper.yolo_cropper.load_yaml_config", return_value=mock_config):
        
        controller = YOLOCropperController(config_path="dummy.yaml")
        
        with pytest.raises(ValueError, match="Unsupported model_name"):
            controller.run()

def test_controller_import_error(mock_config):
    """Verify that ImportError is raised if the module is missing."""
    mock_config["yolo_cropper"]["main"]["model_name"] = "yolov5"

    with patch("src.yolo_cropper.yolo_cropper.ConfigManager"), \
         patch("src.yolo_cropper.yolo_cropper.load_yaml_config", return_value=mock_config), \
         patch("importlib.import_module", side_effect=ModuleNotFoundError("No module")):
        
        controller = YOLOCropperController(config_path="dummy.yaml")
        
        with pytest.raises(ImportError, match="Failed to import module"):
            controller.run()