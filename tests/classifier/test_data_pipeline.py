import pytest
import numpy as np
import cv2
import torch
from pathlib import Path
from torchvision import transforms

from src.classifier.data.cnn_data_loader import ClassificationDataset, create_dataloader
from src.classifier.data.data_preprocessor import DataPreprocessor

@pytest.fixture
def data_env(tmp_path):
    """
    Sets up a temporary dataset structure.
    root/
      train/
        class_A/ (5 images)
        class_B/ (5 images)
      valid/
        class_A/ (2 images)
        class_B/ (2 images)
    """
    root_dir = tmp_path / "clf_data"
    for split in ["train", "valid"]:
        for cls in ["class_A", "class_B"]:
            d = root_dir / split / cls
            d.mkdir(parents=True, exist_ok=True)
            
            # Create dummy images
            count = 5 if split == "train" else 2
            for i in range(count):
                img = np.zeros((64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(d / f"img_{i}.jpg"), img)
                
    return root_dir

def test_dataset_loading(data_env):
    """Verify ClassificationDataset loads images and labels correctly."""
    dataset = ClassificationDataset(input_dir=str(data_env), split="train", verbose=False)
    
    assert len(dataset) == 10  # 5 A + 5 B
    assert len(dataset.label_to_idx) == 2
    
    # Check item retrieval
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, int)
    assert img.shape == (3, 64, 64) # Without transform, ToTensor only changes HWC->CHW

def test_dataloader_batching(data_env):
    """Verify DataLoader yields correct batches."""
    batch_size = 4
    loader = create_dataloader(
        input_dir=str(data_env), 
        split="train", 
        batch_size=batch_size, 
        shuffle=False
    )
    
    batch = next(iter(loader))
    images, labels = batch
    
    assert images.shape == (batch_size, 3, 64, 64)
    assert labels.shape == (batch_size,)

def test_data_preprocessor_logic():
    """Verify DataPreprocessor normalizations and transforms."""
    preprocessor = DataPreprocessor(img_size=(100, 100))
    
    # 1. Check Normalization Map
    assert "resnet" in preprocessor._NORMALIZATION_MAP
    assert "mobilenet_v2" in preprocessor._NORMALIZATION_MAP
    
    # 2. Check Transform Creation
    # Training mode (Should have RandomAffine)
    train_tf = preprocessor.get_transform("resnet", mode="train")
    has_affine = any(isinstance(t, transforms.RandomAffine) for t in train_tf.transforms)
    assert has_affine, "Training transform should include augmentation"
    
    # Eval mode (Should NOT have RandomAffine)
    eval_tf = preprocessor.get_transform("resnet", mode="eval")
    has_affine = any(isinstance(t, transforms.RandomAffine) for t in eval_tf.transforms)
    assert not has_affine, "Eval transform should not include augmentation"
    
    # 3. Check Output Shape
    from PIL import Image
    dummy_pil = Image.new("RGB", (200, 200))
    out_tensor = eval_tf(dummy_pil)
    assert out_tensor.shape == (3, 100, 100)