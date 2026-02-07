import pytest
import cv2
import numpy as np
import yaml
from unittest.mock import patch

from src.classifier.classifier import Classifier

@pytest.fixture
def clf_env(tmp_path):
    """
    Sets up a temporary environment for Classifier testing.
    """
    root_dir = tmp_path / "clf_test_root"
    data_dir = root_dir / "data"
    save_dir = root_dir / "checkpoints"
    metric_dir = root_dir / "metrics"
    check_dir = root_dir / "checkpoints_epoch"
    
    # Create Dirs
    for d in [save_dir, metric_dir, check_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Create Dummy Data (2 Classes: A, B)
    splits = ["train", "valid", "test"]
    classes = ["class_A", "class_B"]
    
    for split in splits:
        for cls in classes:
            cls_dir = data_dir / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            
            # Create 2 small images per class
            for i in range(2):
                img = np.zeros((32, 32, 3), dtype=np.uint8)
                cv2.imwrite(str(cls_dir / f"img_{i}.jpg"), img)
    
    # Create Config
    config_data = {
        "main": {
            "categories": classes,
            "classify_model": "resnet"
        },
        "classifier": {
            "data": {
                "input_dir": str(data_dir)
            },
            "train": {
                "epochs": 1, 
                "batch_size": 2,
                "save_dir": str(save_dir),
                "metric_dir": str(metric_dir),
                "check_dir": str(check_dir),
                "lr": 0.001
            },
            "wandb": {
                "enabled": False,
                "wandb_dir" : "wandb"
            },
            "criterion": "ce"
        },
        "weights": {
            "resnet": "http://dummy_url" 
        }
    }
    
    config_path = root_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        
    return config_path, save_dir, metric_dir

def test_classifier_full_run(clf_env):
    """
    Test End-to-End Classifier Pipeline:
    1. Initialize Classifier with config
    2. Run (Train -> Evaluate)
    3. Verify Outputs (Weights, Metrics)
    """
    config_path, save_dir, metric_dir = clf_env
    
    # Define a lightweight Mock Model
    import torch.nn as nn
    import torch
    
    class TinyModel(nn.Module):
        def __init__(self, model_name=None, num_classes=2, **kwargs):
            super().__init__()
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(3, num_classes) # Input 3 channels
            
        def forward(self, x):
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    with patch("src.classifier.models.factory.get_model", side_effect=TinyModel):
        # Initialize
        clf = Classifier(config_path=str(config_path))
        
        # Run Pipeline
        best_acc, test_acc, test_f1 = clf.run()
        
        # Assertions
        
        # 1. Check Return Values
        assert 0.0 <= best_acc <= 1.0
        assert 0.0 <= test_acc <= 1.0
        
        # 2. Check Saved Weights
        assert (save_dir / "resnet.pt").exists(), "Final model weights not saved"
        
        # 3. Check Metrics Artifacts
        res_metric_dir = metric_dir / "resnet"
        assert (res_metric_dir / "metrics.json").exists(), "Evaluation metrics JSON not saved"
        assert (res_metric_dir / "confusion_matrix.png").exists(), "Confusion Matrix not saved"