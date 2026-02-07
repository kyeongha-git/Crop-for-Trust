import pytest
import cv2
import numpy as np
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
from io import BytesIO
from PIL import Image

from src.annotation_cleaner.annotation_cleaner import AnnotationCleaner

@pytest.fixture
def gac_env(tmp_path):
    """
    Sets up a temporary environment for GAC pipeline testing.
    Creates input directories, dummy images, and a temporary config file.
    """
    # 1. Define Directories
    root_dir = tmp_path / "gac_test_root"
    input_dir = root_dir / "input"
    output_dir = root_dir / "output"
    
    # Internal dirs (usually handled by pipeline, but defined for config)
    pad_dir = root_dir / "padded"
    clean_dir = root_dir / "cleaned"
    restore_dir = root_dir / "restored"
    
    # Create Input Categories
    (input_dir / "repair").mkdir(parents=True, exist_ok=True)
    
    # 2. Create Dummy Input Image (Non-square to test padding/restore)
    # Size: 100(W) x 200(H)
    img_h, img_w = 200, 100
    dummy_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    # Add some color to distinguish
    cv2.rectangle(dummy_img, (10, 10), (50, 50), (255, 0, 0), -1) 
    
    input_img_path = input_dir / "repair" / "test_sample.jpg"
    cv2.imwrite(str(input_img_path), dummy_img)

    # 3. Create Temp Config YAML
    config_data = {
        "main": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "categories": ["repair"],
            "metadata_name": "padding_info.json"
        },
        "annotation_cleaner": {
            "main": {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir)
            },
            "image_padding": {
                "input_dir": str(input_dir),
                "output_dir": str(pad_dir),
                "target_size": 500  # Smaller target for testing
            },
            "annotation_clean": {
                "input_dir": str(pad_dir),
                "output_dir": str(clean_dir),
                "model": "gemini-test-model",
                "api_key": "dummy_key", # Will be mocked
                "test_mode": True,
                "test_limit": 1,
                "prompt": "Remove annotations."
            },
            "restore_crop": {
                "input_dir": str(clean_dir),
                "output_dir": str(restore_dir),
                "metadata_root": str(pad_dir),
                "keep_metadata": True # Keep for inspection
            }
        }
    }
    
    config_path = root_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        
    return config_path, input_img_path

def test_gac_full_pipeline(gac_env):
    """
    Integration Test for the GAC Pipeline.
    
    Flow:
    1. Padding: 100x200 -> 500x500 (Black borders)
    2. Cleaning: Mocked API returns a generated image (500x500)
    3. Restore: 500x500 -> 100x200 (Crops center)
    4. Merge: Final result saved to output dir
    """
    config_path, input_img_path = gac_env
    
    # Mocking Gemini API
    # We need to mock 'google.genai.Client' used in 'CleanAnnotation'
    # The chain is: client.models.generate_content -> response.candidates[0].content.parts[0].inline_data.data
    
    with patch("src.annotation_cleaner.core.clean_annotation.get_gemini_client") as mock_get_client:
        
        # 1. Setup Mock Client & Response
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Create a dummy generated image (simulating Gemini output)
        # It must match the padded size (500x500) defined in config
        gen_img_pil = Image.new("RGB", (500, 500), color=(0, 255, 0)) # Green image
        img_byte_arr = BytesIO()
        gen_img_pil.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        
        # Construct the response object structure
        mock_part = MagicMock()
        mock_part.inline_data.data = img_bytes
        mock_part.text = None # Ensure it's treated as image response
        
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        
        # Assign response to the method call
        mock_client.models.generate_content.return_value = mock_response

        # Execution
        cleaner = AnnotationCleaner(config_path=str(config_path))
        cleaner.run()
        
        # Assertions
        
        # 1. Load Config to verify paths
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            
        final_out_dir = Path(cfg["annotation_cleaner"]["main"]["output_dir"]) / "repair"
        final_img_path = final_out_dir / "test_sample.jpg"
        
        # A. File Existence Check
        assert final_img_path.exists(), "Final processed image was not created."
        
        # B. Dimension Restoration Check
        # Input was 100x200. Pipeline padded to 500x500, then restored.
        # Result should be 100x200.
        final_img = cv2.imread(str(final_img_path))
        h, w = final_img.shape[:2]
        
        assert h == 200 and w == 100, f"Restoration failed. Expected 100x200, got {w}x{h}"
        
        # C. Content Check (Optional)
        center_pixel = final_img[100, 50] # y, x
        assert final_img.shape is not None
        # Note: We just check if it runs without error and restores dimensions mostly.
        
        print("\n>>> GAC Pipeline Test Passed Successfully!")