#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_train_pipeline.py
------------------------
Unit and integration tests for the classifier training pipeline.

Covers:
1️⃣ Config loading
2️⃣ WandB initialization
3️⃣ train_one_epoch() and validate()
4️⃣ train_model() saving behavior
5️⃣ Classifier.run() integration
6️⃣ Exception handling
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.classifier.classifier import Classifier
from src.classifier.train import train_model, train_one_epoch, validate


# ======================================================
# Config Loading
# ======================================================
def test_load_config_contains_sections(tmp_path):
    """Verify that config.yaml contains required sections."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
classifier:
  data:
    input_dir: "data/original"
  train:
    model_name: "mobilenet_v2"
    batch_size: 2
    lr: 0.001
    weight_decay: 0.01
    epochs: 1
    save_dir: "./saved_model/classifier"
    check_dir: "./checkpoints/classifier"
  wandb:
    enabled: false
"""
    )

    clf = Classifier(str(cfg_file))
    cfg = clf.cfg
    assert "data" in cfg and "train" in cfg and "wandb" in cfg
    print("_load_config() passed")


# ======================================================
# WandB Initialization
# ======================================================
def test_init_wandb_disabled(tmp_path):
    """_init_wandb(): should return None when disabled."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
classifier:
  data:
    input_dir: "data/original"
  train:
    model_name: "mobilenet_v2"
  wandb:
    enabled: false
"""
    )

    clf = Classifier(str(cfg_file))
    result = clf._init_wandb()
    assert result is None
    print("_init_wandb() passed — disabled mode")


@patch("wandb.init")
def test_init_wandb_enabled(mock_init, tmp_path):
    """_init_wandb(): should call wandb.init() when enabled."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
classifier:
  data:
    input_dir: "data/original_crop/yolov2"
  train:
    model_name: "vgg16"
  wandb:
    enabled: true
    project: "TestProj"
    entity: "TestEntity"
    run_name_pattern: "{model}_{input_dir}"
"""
    )

    clf = Classifier(str(cfg_file))
    wandb_obj = clf._init_wandb()
    mock_init.assert_called_once()
    assert wandb_obj is not None
    print("_init_wandb() passed — enabled mode")


# ======================================================
# train_one_epoch() / validate() Tests
# ======================================================
def make_dummy_dataloader(batch_size=4, num_samples=8):
    """Create a small dummy TensorDataset for quick testing."""
    x = torch.rand(num_samples, 3, 360, 360)
    y = (torch.rand(num_samples, 1) > 0.5).float()
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)


def test_train_one_epoch():
    """train_one_epoch(): verifies loss and accuracy computation."""
    dataloader = make_dummy_dataloader()
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 360 * 360, 1))
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cpu")

    train_loss, train_acc = train_one_epoch(
        model, dataloader, criterion, optimizer, device
    )
    assert train_loss >= 0
    assert 0 <= train_acc <= 1
    print(f"train_one_epoch() OK — loss={train_loss:.4f}, acc={train_acc:.4f}")


def test_validate():
    """validate(): verifies evaluation loss and accuracy computation."""
    dataloader = make_dummy_dataloader()
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 360 * 360, 1))
    criterion = torch.nn.BCEWithLogitsLoss()
    device = torch.device("cpu")

    val_loss, val_acc = validate(model, dataloader, criterion, device)
    assert val_loss >= 0
    assert 0 <= val_acc <= 1
    print(f"validate() OK — loss={val_loss:.4f}, acc={val_acc:.4f}")


# ======================================================
# train_model() Tests
# ======================================================
def test_train_model_saves_best(tmp_path):
    """train_model(): should save best and last checkpoint files."""
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 360 * 360, 1))
    dataloader = make_dummy_dataloader()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cpu")

    save_path = tmp_path / "best_model.pt"
    check_path = tmp_path / "checkpoints" / "last.pt"
    os.makedirs(check_path.parent, exist_ok=True)

    best_acc = train_model(
        model,
        dataloader,
        dataloader,
        criterion,
        optimizer,
        device,
        epochs=2,
        save_path=str(save_path),
        check_path=str(check_path),
        wandb_run=None,
    )

    assert save_path.exists()
    assert check_path.exists()
    print(f"train_model() OK — best_acc={best_acc:.4f}")


# ======================================================
# Classifier.run() Integration Test
# ======================================================
@patch("src.classifier.train.train_model", return_value=0.95)
@patch("src.classifier.models.factory.get_model")
@patch("src.classifier.evaluate.Evaluator.run", return_value=(0.90, 0.88))
def test_classifier_run(mock_eval_run, mock_get_model, mock_train_model, tmp_path):
    """Classifier.run(): full pipeline test with mocks."""
    data_dir = tmp_path / "data" / "original_crop" / "yolov2"
    train_dir = data_dir / "train" / "repair"
    valid_dir = data_dir / "valid" / "repair"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10)).save(train_dir / "x.jpg")
    Image.new("RGB", (10, 10)).save(valid_dir / "x.jpg")

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        f"""
classifier:
  data:
    input_dir: "{data_dir}"
  train:
    model_name: "mobilenet_v2"
    batch_size: 2
    epochs: 1
    lr: 0.001
    weight_decay: 0.00001
    save_dir: "{tmp_path}/saved_model/classifier"
    metric_dir: "{tmp_path}/metrics/classifier"
    check_dir: "{tmp_path}/checkpoints/classifier"
  wandb:
    enabled: false
"""
    )

    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
    mock_model.to.return_value = mock_model
    mock_get_model.return_value = mock_model

    clf = Classifier(str(cfg_file))
    best_acc, acc, f1 = clf.run()

    assert isinstance(best_acc, float)
    assert isinstance(acc, float)
    assert isinstance(f1, float)
    print(
        f"Classifier.run() OK — best_acc={best_acc:.4f}, acc={acc:.4f}, f1={f1:.4f}"
    )


# ======================================================
# Exception Handling
# ======================================================
def test_load_config_file_not_found():
    """Ensure Classifier raises FileNotFoundError when config missing."""
    with pytest.raises(FileNotFoundError):
        Classifier("./non_existent.yaml")
    print("Exception handling OK — FileNotFoundError")
