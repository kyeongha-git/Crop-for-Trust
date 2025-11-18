#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_models.py
--------------
Unit and integration tests for all classification models.

Covers:
- VGG16 / ResNet152 / MobileNetV2 / MobileNetV3
- Verifies forward pass, parameter freezing, dropout behavior,
  and end-to-end data → transform → model pipeline.
"""


import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.classifier.data.cnn_data_loader import ClassificationDataset
from src.classifier.data.data_preprocessing import DataPreprocessor
from src.classifier.models.factory import get_model


# ==============================================================
# Helper Functions
# ==============================================================
def run_forward_pass(
    model_name: str, num_classes: int = 1, input_size=(1, 3, 360, 360)
):
    """
    Initialize a model and perform a dummy forward pass.

    """
    model = get_model(model_name, num_classes=num_classes)
    model.eval()
    x = torch.randn(*input_size)
    with torch.no_grad():
        y = model(x)
    return model, y


def compute_loss(output: torch.Tensor):
    """
    Compute BCEWithLogitsLoss to verify output validity (no NaNs).
    """
    criterion = nn.BCEWithLogitsLoss()
    target = torch.ones_like(output)
    loss = criterion(output, target)
    assert not torch.isnan(loss), "NaN detected during loss computation"
    return loss.item()


def count_parameters(model: nn.Module):
    """Count total and trainable parameters of a given model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ==============================================================
# Test Model Structure & Forward
# ==============================================================
@pytest.mark.parametrize(
    "model_name, expect_dropout",
    [
        ("vgg16", True),
        ("resnet152", False),  # ResNet has no Dropout
        ("mobilenet_v2", True),
        ("mobilenet_v3", True),
    ],
)
def test_model_forward_and_structure(model_name, expect_dropout):
    """Verify model forward pass, output shape, and dropout presence."""
    print(f"\n[TEST] {model_name.upper()} forward pass & structure validation")

    model, output = run_forward_pass(model_name)

    # Validate output dimension
    assert (
        output.ndim == 2 and output.shape[1] == 1
    ), f"{model_name} output shape invalid: {output.shape}"

    # Validate dropout presence
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    assert has_dropout == expect_dropout, (
        f"{model_name}: Dropout mismatch "
        f"(expected={expect_dropout}, found={has_dropout})"
    )

    # Compute sample loss
    loss_val = compute_loss(output)
    total_params, trainable_params = count_parameters(model)
    print(f" - BCE Loss: {loss_val:.4f}")
    print(f" - Params: total={total_params:,}, trainable={trainable_params:,}")
    print(f"{model_name.upper()} structure & forward test passed")


# ==============================================================
# Test Backbone Freezing
# ==============================================================
@pytest.mark.parametrize("model_name", ["resnet152", "mobilenet_v2", "mobilenet_v3"])
def test_freeze_backbone_option(model_name):
    """Verify freeze_backbone option correctly sets parameter gradients."""
    model_frozen = get_model(model_name, freeze_backbone=True)
    model_trainable = get_model(model_name, freeze_backbone=False)

    frozen_params = [p.requires_grad for p in model_frozen.parameters()]
    trainable_params = [p.requires_grad for p in model_trainable.parameters()]

    assert any(
        trainable_params
    ), f"{model_name}: freeze_backbone=False but all params frozen"
    assert not all(
        frozen_params
    ), f"{model_name}: freeze_backbone=True but params remain trainable"


# ==============================================================
# Test Data → Transform → Model Pipeline
# ==============================================================
@pytest.mark.parametrize(
    "model_name",
    ["vgg16", "resnet152", "mobilenet_v2", "mobilenet_v3"],
)
def test_real_end_to_end_pipeline(tmp_path, model_name):
    """
    Full pipeline test: Dataset → Transform → Model forward.

    Steps:
        1️⃣ Create dummy dataset.
        2️⃣ Load transform for the target model.
        3️⃣ Perform forward pass through model.
    """
    print(f"\n{model_name.upper()} full data pipeline test")

    # Create dummy dataset
    data_dir = tmp_path / "data" / "original_crop" / "yolov2" / "train" / "repair"
    data_dir.mkdir(parents=True, exist_ok=True)
    dummy_path = data_dir / "dummy.jpg"

    img = Image.fromarray(
        (torch.rand(3, 360, 360).permute(1, 2, 0).numpy() * 255).astype("uint8")
    )
    img.save(dummy_path)

    # Load dataset + transform
    dp = DataPreprocessor(img_size=(360, 360))
    transform = dp.get_transform(model_name=model_name, mode="train")

    dataset = ClassificationDataset(
        input_dir=str(tmp_path / "data" / "original_crop" / "yolov2"),
        split="train",
        transform=transform,
        verbose=True,
    )

    # Verify sample shape
    img_tensor, label = dataset[0]
    assert isinstance(img_tensor, torch.Tensor), "Transform did not return a Tensor"
    assert img_tensor.shape == (
        3,
        360,
        360,
    ), f"Image shape mismatch: {img_tensor.shape}"

    x = img_tensor.unsqueeze(0)

    # Forward pass
    model = get_model(model_name, num_classes=1)
    model.eval()
    with torch.no_grad():
        y = model(x)

    # Validate results
    assert (
        y.ndim == 2 and y.shape[1] == 1
    ), f"{model_name} output shape invalid: {y.shape}"
    loss_val = compute_loss(y)

    print(f"{model_name.upper()} end-to-end pipeline passed (loss={loss_val:.4f})")
