import pytest
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from src.annotation_cleaner.evaluation.generate_anchor import run_anchor_generation
from src.annotation_cleaner.evaluation.evaluate import Evaluator

@pytest.fixture
def mock_env(tmp_path):
    """
    Sets up a temporary directory structure with dummy data and configuration.
    """
    # 1. Define Directories
    root_dir = tmp_path / "gac_eval"
    orig_dir = root_dir / "original"
    label_dir = root_dir / "label"
    gac_dir = root_dir / "gac_full"
    
    # Create dirs
    for d in [orig_dir, label_dir, gac_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Create Dummy Data (Image & Label)
    img_name = "test_sample"
    
    # Create dummy image (100x100, black)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(orig_dir / f"{img_name}.jpg"), dummy_img)
    cv2.imwrite(str(gac_dir / f"{img_name}.jpg"), dummy_img) # Assume GAC output exists

    # Create dummy YOLO label (One object in the center)
    # Format: class x_center y_center width height (normalized)
    with open(label_dir / f"{img_name}.txt", "w") as f:
        f.write("0 0.5 0.5 0.5 0.5")

    # 3. Construct Config
    config = {
        "annotation_cleaner": {
            "evaluate": {
                "dirs": {
                    "root": str(root_dir),
                    "original": str(orig_dir),
                    "label": str(label_dir),
                    "gac_full": str(gac_dir),
                    # Outputs
                    "baseline_crop": str(root_dir / "baseline"),
                    "anchor_root": str(root_dir / "anchors"),
                    "results": str(root_dir / "results"),
                    "marked_full": str(root_dir / "marked_full"),
                },
                "params": {
                    "seed": 42,
                    "marking": {"thickness_min": 1, "thickness_max": 2},
                    "mix": {"use_mixed_clone": False},
                    "benign": {"blur_sigma": 0.5, "jpeg_quality": 80}
                },
                "weights": {
                    "vgg": None # Use ImageNet weights
                }
            }
        }
    }
    
    return config, root_dir

def test_pipeline_execution(mock_env):
    """
    Tests the full pipeline execution flow:
    Generation -> Evaluation -> Visualization
    """
    config, root_dir = mock_env
    
    # Step 1: Data Preparation (Anchor Gen)
    try:
        run_anchor_generation(config)
    except Exception as e:
        pytest.fail(f"Anchor Generation failed: {e}")

    # Check if files were generated
    anchor_root = Path(config["annotation_cleaner"]["evaluate"]["dirs"]["anchor_root"])
    assert (anchor_root / "marked").exists(), "Marked anchors missing"
    assert (anchor_root / "blur").exists(), "Blur anchors missing"
    assert (anchor_root / "mix").exists(), "Mix anchors missing"
    
    # Check if crops exist (Baseline & GAC)
    baseline_dir = Path(config["annotation_cleaner"]["evaluate"]["dirs"]["baseline_crop"])
    assert list(baseline_dir.glob("*.jpg")), "Baseline crop not created"

    # Step 2: Evaluation & Visualization
    try:
        # Evaluator runs metric calc AND visualization
        evaluator = Evaluator(config)
        evaluator.run()
    except Exception as e:
        pytest.fail(f"Evaluation process failed: {e}")

    # Check Results
    results_dir = Path(config["annotation_cleaner"]["evaluate"]["dirs"]["results"])
    
    # 1. CSV files
    assert (results_dir / "metrics_details.csv").exists(), "Details CSV missing"
    assert (results_dir / "metrics_summary.csv").exists(), "Summary CSV missing"
    
    # 2. PDF Figure
    assert list(results_dir.glob("*.pdf")), "Figure PDF missing"

    # 3. Verify CSV Content
    df = pd.read_csv(results_dir / "metrics_summary.csv")
    assert not df.empty, "Summary CSV is empty"
    assert "SSIM" in df.columns, "SSIM column missing in summary"