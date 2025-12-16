#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate.py
-----------
Performs end-to-end evaluation of trained classification models.

This module loads a saved model, runs inference on the test dataset,
computes key performance metrics (Accuracy, F1-score), and exports results
including a confusion matrix visualization.
Supports both local evaluation and standalone execution.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging
from src.classifier.data.cnn_data_loader import ClassificationDataset
from src.classifier.data.data_preprocessor import DataPreprocessor
from src.classifier.models.factory import get_model


class Evaluator:
    def __init__(
        self,
        input_dir: str,
        model: str,
        save_dir: str,
        metric_dir: str,
        wandb_run=None,
    ):
        setup_logging("logs/classifier_eval")
        self.logger = get_logger("Evaluator")

        self.input_dir = Path(input_dir)
        self.model_name = model.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb_run = wandb_run

        self.save_root = Path(save_dir)
        self.metric_root = Path(metric_dir)

        self.logger.info(f"Device     : {self.device}")
        self.logger.info(f"Dataset    : {self.input_dir}")
        self.logger.info(f"Model      : {self.model_name}")
        self.logger.info(f"Save Dir   : {self.save_root}")
        self.logger.info(f"Metric Dir : {self.metric_root}")


    # ======================================================
    # Transform
    # ======================================================
    def _get_transform(self):
        return DataPreprocessor().get_transform(self.model_name, mode="eval")

    # ======================================================
    # Load Model
    # ======================================================
    def _load_model(self):
        model = get_model(self.model_name, num_classes=1)
        model_path = self.save_root / f"{self.model_name}.pt"

        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        model.to(self.device).eval()

        self.logger.info(f"Model loaded successfully → {model_path}")
        return model

    # ======================================================
    # Load Data
    # ======================================================
    def _load_data(self, transform):
        test_dataset = ClassificationDataset(
            input_dir=self.input_dir,
            split="test",
            transform=transform,
            verbose=False,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False
        )
        self.logger.info(f"Test dataset loaded — {len(test_dataset)} samples")
        return test_loader

    # ======================================================
    # Run Evaluation
    # ======================================================
    def run(self):
        transform = self._get_transform()
        test_loader = self._load_data(transform)
        model = self._load_model()

        y_true, y_pred = [], []

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                outputs = torch.sigmoid(model(imgs))
                preds = (outputs > 0.5).long().cpu().numpy().flatten()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        self.logger.info(f"Test Accuracy: {acc:.4f}")
        self.logger.info(f"Test F1-score: {f1:.4f}")

        self._save_results(y_true, y_pred, acc, f1)

        if self.wandb_run is not None:
            self.wandb_run.log({"test_accuracy": acc, "test_f1": f1})

        return acc, f1

    # ======================================================
    # Save Results
    # ======================================================
    def _save_results(self, y_true, y_pred, acc, f1):
        save_dir = self.metric_root / self.model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = save_dir / "metrics.json"
        cm_path = save_dir / "cm.png"

        metrics_data = {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "num_samples": len(y_true),
        }

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=4)

        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(
            cmap="Blues", values_format="d"
        )
        plt.savefig(cm_path, dpi=200, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Metrics saved → {metrics_path}")
        self.logger.info(f"Confusion matrix saved → {cm_path}")