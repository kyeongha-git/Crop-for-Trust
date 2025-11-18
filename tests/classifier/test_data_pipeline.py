#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_classifier_data_pipeline.py
--------------------------------
Unit tests for the classification data pipeline.

Covers:
1️⃣ Data loader utilities (list_image_paths, label mapping, dataset initialization)
2️⃣ Data preprocessor transformations for training and evaluation modes
"""

import pytest
import torch
from PIL import Image

from src.classifier.data.cnn_data_loader import (ClassificationDataset,
                                                 build_label_mappings,
                                                 list_image_paths)
from src.classifier.data.data_preprocessing import DataPreprocessor


# ==========================================================
# Data Loader Unit Tests
# ==========================================================
def test_list_image_paths(tmp_path):
    """list_image_paths(): should traverse directory and detect classes."""
    (tmp_path / "repair").mkdir()
    (tmp_path / "replace").mkdir()

    # Create 2 images per class
    for cls in ["repair", "replace"]:
        for i in range(2):
            Image.new("RGB", (10, 10)).save(tmp_path / cls / f"img_{i}.jpg")

    pairs = list_image_paths(tmp_path)
    labels = [p[1] for p in pairs]

    from collections import Counter

    label_counts = Counter(labels)

    assert set(labels) == {"repair", "replace"}
    assert all(
        count >= 2 for count in label_counts.values()
    ), f"Each class should have ≥2 samples: {label_counts}"


def test_build_label_mappings():
    """build_label_mappings(): should produce bidirectional label-index maps."""
    labels = ["repair", "replace", "repair"]
    label_to_idx, idx_to_label = build_label_mappings(labels)

    assert label_to_idx == {"repair": 0, "replace": 1}
    assert idx_to_label == {0: "repair", 1: "replace"}
    assert label_to_idx[idx_to_label[0]] == 0


def test_classification_dataset_init(tmp_path):
    """ClassificationDataset: should load images and build label mapping."""
    data_dir = tmp_path / "data" / "original" / "train" / "repair"
    data_dir.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(data_dir / "x.jpg")

    dataset = ClassificationDataset(
        input_dir=str(tmp_path / "data" / "original"),
        split="train",
        verbose=False,
    )

    assert len(dataset) == 1
    assert isinstance(dataset.image_paths[0], str)
    assert dataset.labels[0] == "repair"
    assert dataset.label_to_idx == {"repair": 0}


def test_classification_dataset_getitem(tmp_path):
    """ClassificationDataset: __getitem__() should return a tensor and label index."""
    data_dir = tmp_path / "data" / "original" / "train" / "replace"
    data_dir.mkdir(parents=True)
    img_path = data_dir / "img.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    dataset = ClassificationDataset(
        input_dir=str(tmp_path / "data" / "original"),
        split="train",
        verbose=False,
    )

    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 3
    assert isinstance(label, int)


# ==========================================================
# Data Preprocessor Unit Tests
# ==========================================================
@pytest.fixture(scope="module")
def preprocessor():
    """Fixture providing a shared DataPreprocessor instance."""
    return DataPreprocessor(img_size=(224, 224))


@pytest.mark.parametrize(
    "model_name, expected_mean",
    [
        ("vgg", [0.485, 0.456, 0.406]),
        ("resnet", [0.485, 0.456, 0.406]),
        ("mobilenet_v1", [0.0, 0.0, 0.0]),
        ("mobilenet_v2", [0.5, 0.5, 0.5]),
        ("mobilenet_v3", [0.5, 0.5, 0.5]),
    ],
)
def test_get_transform_eval(preprocessor, model_name, expected_mean):
    """get_transform(): should return deterministic evaluation transform."""
    transform = preprocessor.get_transform(model_name, mode="eval")
    img = Image.new("RGB", (100, 100))
    tensor = transform(img)

    assert tensor.shape[1:] == (224, 224)
    assert tensor.shape[0] == 3
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32

    op_names = [t.__class__.__name__ for t in transform.transforms]
    assert "RandomAffine" not in op_names


@pytest.mark.parametrize("model_name", ["vgg", "resnet", "mobilenet_v2"])
def test_get_transform_train_randomness(preprocessor, model_name):
    """get_transform(): should include stochastic augmentations in training mode."""
    transform = preprocessor.get_transform(model_name, mode="train")

    import numpy as np

    arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    out1 = transform(img)
    out2 = transform(img)

    diff_ratio = torch.mean(torch.abs(out1 - out2)).item()
    assert diff_ratio > 1e-4, f"{model_name}: augmentation seems inactive"


def test_invalid_model_raises(preprocessor):
    """get_transform(): should raise ValueError for unsupported model names."""
    with pytest.raises(ValueError):
        preprocessor.get_transform("invalid_model")


def test_internal_compose(preprocessor):
    """_compose(): should include/exclude augmentations correctly."""
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    t_train = preprocessor._compose(mean, std, augment=True)
    t_eval = preprocessor._compose(mean, std, augment=False)

    op_train = [t.__class__.__name__ for t in t_train.transforms]
    assert "RandomAffine" in op_train

    op_eval = [t.__class__.__name__ for t in t_eval.transforms]
    assert "RandomAffine" not in op_eval
