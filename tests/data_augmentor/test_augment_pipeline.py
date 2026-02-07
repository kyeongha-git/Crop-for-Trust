import pytest
import cv2
import numpy as np
import yaml

from src.data_augmentor.data_augmentor import DataAugmentor

@pytest.fixture
def augment_env(tmp_path):
    """
    Sets up a temporary environment for Data Augmentor pipeline testing.
    Creates an imbalanced dataset and a temporary config file.
    
    Structure:
    root/
      original/
        class_A/ (10 images) - Majority
        class_B/ (2 images)  - Minority
    """
    root_dir = tmp_path / "aug_test_root"
    input_dir = root_dir / "original"
    
    # Create Class Directories
    class_a_dir = input_dir / "class_A"
    class_b_dir = input_dir / "class_B"
    class_a_dir.mkdir(parents=True, exist_ok=True)
    class_b_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Dummy Images (10 for A, 2 for B)
    # Using small black images for speed
    dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
    
    for i in range(10):
        cv2.imwrite(str(class_a_dir / f"img_a_{i}.jpg"), dummy_img)
        
    for i in range(2):
        cv2.imwrite(str(class_b_dir / f"img_b_{i}.jpg"), dummy_img)
        
    return root_dir, input_dir

def create_temp_config(root_dir, input_dir, output_dir, demo_mode=False):
    """Helper to create a temporary YAML config."""
    config_data = {
        "main": {
            "demo": demo_mode
        },
        "data_augmentor": {
            "data": {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir)
            },
            "split": {
                "train_ratio": 0.8,
                "valid_ratio": 0.1,
                "test_ratio": 0.1
            },
            "augmentation": {
                "enable": True,
                # Simple aug params
                "random_hflip_p": 0.5 
            }
        }
    }
    config_path = root_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path

def test_augmentor_full_pipeline(augment_env):
    """
    Test the full pipeline: Split -> Augment -> Cleanup (Production Mode)
    Expectation:
    1. Splits created (train/valid/test).
    2. Class B in train set augmented to match Class A count.
    3. Original source folders (class_A, class_B) removed from input_dir.
    """
    root_dir, input_dir = augment_env
    output_dir = root_dir / "dataset" # Production output usually same as input or separate
    
    # Config for Production Mode (demo=False)
    config_path = create_temp_config(root_dir, input_dir, output_dir, demo_mode=False)
    
    # Run Pipeline
    augmentor = DataAugmentor(config_path=str(config_path))
    augmentor.run()
    
    # Assertions
    
    # 1. Check Split Directories Existence
    for split in ["train", "valid", "test"]:
        assert (output_dir / split).exists(), f"{split} directory missing"
        assert (output_dir / split / "class_A").exists()
        assert (output_dir / split / "class_B").exists()
        
    # 2. Check Class Balancing (Augmentation)
    # Train ratio 0.8 -> Class A: 8 images, Class B: ~1-2 images (initially)
    # After balancing, Class B train count should equal Class A train count (8)
    train_a_count = len(list((output_dir / "train" / "class_A").glob("*.jpg")))
    train_b_count = len(list((output_dir / "train" / "class_B").glob("*.jpg")))
    
    # Note: Split is randomized, but usually A gets 8 images (10 * 0.8).
    # Balancing target is max(class_counts), so B should be boosted to match A.
    assert train_a_count == train_b_count, \
        f"Class balancing failed. A: {train_a_count}, B: {train_b_count}"
        
    # 3. Check Cleanup (Production Mode)
    assert not (input_dir / "class_A").exists(), "Cleanup failed: class_A still exists"
    assert not (input_dir / "class_B").exists(), "Cleanup failed: class_B still exists"

def test_augmentor_demo_mode(augment_env):
    """
    Test Demo Mode: Split -> Augment -> No Cleanup
    Expectation:
    1. Output is inside a subdirectory (e.g., input_dir/dataset).
    2. Original source folders are PRESERVED.
    """
    root_dir, input_dir = augment_env
    # In demo mode logic provided: output_dir is derived inside the class if not explicit,
    # but here we provide one via config to be safe/clear.
    # The code says: if demo_mode: output_dir = input_dir / demo_subdir
    # Let's verify that behavior by NOT setting output_dir in data_cfg if possible, 
    # OR just follow the controller logic.
    # Controller logic:
    # if demo_mode: output_dir = input_dir / demo_subdir
    
    # Let's create a specific config for demo
    config_data = {
        "main": {
            "demo": True
        },
        "data_augmentor": {
            "data": {
                "input_dir": str(input_dir),
                "demo_subdir": "demo_output"
            },
            "split": {
                "train_ratio": 0.8,
                "valid_ratio": 0.1,
                "test_ratio": 0.1
            },
            "augmentation": {
                "enable": False # Disable aug to just test split/cleanup logic
            }
        }
    }
    config_path = root_dir / "demo_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        
    # Run Pipeline
    augmentor = DataAugmentor(config_path=str(config_path))
    augmentor.run()
    
    # Assertions
    # 1. Check Output Location
    expected_out = input_dir / "demo_output"
    assert expected_out.exists(), "Demo output directory not created"
    assert (expected_out / "train").exists()
    
    # 2. Check Cleanup Skipped
    assert (input_dir / "class_A").exists(), "Source class_A removed in demo mode!"
    assert (input_dir / "class_B").exists(), "Source class_B removed in demo mode!"