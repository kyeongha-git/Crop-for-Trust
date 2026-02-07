import pytest
import numpy as np
import torch
from pathlib import Path

from src.annotation_cleaner.evaluation import metrics as M

@pytest.fixture
def sample_data():
    np.random.seed(42)
    
    img_a = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    img_identical = img_a.copy()
    
    img_diff = np.zeros_like(img_a)
    
    return img_a, img_identical, img_diff

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def test_calculate_l1_distance(sample_data):
    img_a, img_identical, img_diff = sample_data
    
    # Case 1: Same image -> expected L1 = 0
    l1_same = M.calculate_l1_distance(img_a, img_identical)
    assert l1_same == 0.0, f"Identical images should have L1=0, got {l1_same}"
    
    # Case 2: Different image -> expected L1 > 0
    l1_diff = M.calculate_l1_distance(img_a, img_diff)
    assert l1_diff > 0.0, "Different images should have L1 > 0"

def test_calculate_psnr(sample_data):
    img_a, img_identical, img_diff = sample_data
    
    # Case 1: Same image -> expected PSNR = 100.0
    psnr_same = M.calculate_psnr(img_a, img_identical)
    assert psnr_same == 100.0, f"Identical images PSNR should be 100.0, got {psnr_same}"
    
    # Case 2: Different image -> expected PSNR < 0
    psnr_diff = M.calculate_psnr(img_a, img_diff)
    assert psnr_diff < 100.0
    assert psnr_diff > 0.0

def test_calculate_ssim(sample_data, device):
    img_a, img_identical, img_diff = sample_data
    
    # Case 1: Same image -> expected SSIM = 1.0
    ssim_same = M.calculate_ssim(img_a, img_identical, device=device)
    
    assert np.isclose(ssim_same, 1.0, atol=1e-4), f"Identical images SSIM should be 1.0, got {ssim_same}"
    
    # Case 2: Different image -> expected SSIM < 1.0
    ssim_diff = M.calculate_ssim(img_a, img_diff, device=device)
    assert ssim_diff < 1.0, f"Different images SSIM should be < 1.0, got {ssim_diff}"

def test_calculate_edge_distance(sample_data):
    img_a, img_identical, img_diff = sample_data
    
    # Case 1: Same image -> expected Edge L1 = 0
    edge_same = M.calculate_edge_distance(img_a, img_identical)
    assert edge_same == 0.0
    
    # Case 2: Different image -> expected Edge L1 > 0.0
    edge_diff = M.calculate_edge_distance(img_a, img_diff)
    assert edge_diff > 0.0

def test_vgg_feature_extractor_init(device):
    try:
        model = M.VGGFeatureExtractor(device=device)
        assert model is not None
        assert not model.training
    except Exception as e:
        pytest.fail(f"VGG Model initialization failed: {e}")

def test_calc_feature_distance(sample_data, device):
    img_a, img_identical, img_diff = sample_data
    
    model = M.VGGFeatureExtractor(device=device)
    
    t_a = M.cv2_to_tensor(img_a, device)
    t_identical = M.cv2_to_tensor(img_identical, device)
    t_diff = M.cv2_to_tensor(img_diff, device)
    
    # Case 1: Same image -> expected Edge D_FeatMap = 0
    with torch.no_grad():
        dist_same = M.calc_feature_distance(model, t_a, t_identical)
    assert np.isclose(dist_same, 0.0, atol=1e-5), f"Feature distance for same images should be 0, got {dist_same}"
    
    # Case 2: Different image -> expected D_FeatMap > 0.0
    with torch.no_grad():
        dist_diff = M.calc_feature_distance(model, t_a, t_diff)
    assert dist_diff > 0.0, "Feature distance for different images should be positive"