#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate.py

Performs end-to-end evaluation of trained classification models.
Computes performance metrics (Accuracy, F1-Score) and generates visualization
artifacts (Confusion Matrix) for both binary and multi-class tasks.
"""

import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger, setup_logging
from src.classifier.data.cnn_data_loader import ClassificationDataset
from src.classifier.data.data_preprocessor import DataPreprocessor
from src.classifier.models.factory import get_model


class Evaluator:
    """
    Manages the evaluation process for classification models.
    """

    def __init__(
        self,
        model: str,
        input_dir: Union[str, Path],
        save_dir: Union[str, Path],
        metric_dir: Union[str, Path],
        num_classes: int,
        wandb_run: Optional[Any] = None,
    ) -> None:
        """
        Initializes the evaluator with model and dataset configurations.

        Args:
            model (str): Name of the model architecture (e.g., 'resnet50').
            input_dir (Union[str, Path]): Path to the test dataset.
            save_dir (Union[str, Path]): Directory where model weights are stored.
            metric_dir (Union[str, Path]): Directory to save evaluation results.
            num_classes (int): Number of target classes.
            wandb_run (Optional[Any]): Active wandb run object for logging.
        """
        setup_logging("logs/classifier_eval")
        self.logger = get_logger("classifier.Evaluator")

        self.input_dir = Path(input_dir)
        self.model_name = model.lower()
        self.num_classes = int(num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb_run = wandb_run

        self.save_root = Path(save_dir)
        self.metric_root = Path(metric_dir)

        self.logger.info(f"Device      : {self.device}")
        self.logger.info(f"Dataset     : {self.input_dir}")
        self.logger.info(f"Model       : {self.model_name}")
        self.logger.info(f"Num classes : {self.num_classes}")
        self.logger.info(f"Save Dir    : {self.save_root}")
        self.logger.info(f"Metric Dir  : {self.metric_root}")

    # --------------------------------------------------
    # Resource Loading
    # --------------------------------------------------

    def _get_transform(self) -> Any:
        """Retrieves the data transformation pipeline for evaluation."""
        return DataPreprocessor().get_transform(self.model_name, mode="eval")

    def _load_model(self) -> nn.Module:
        """
        Loads the trained model architecture and weights.

        Returns:
            nn.Module: The model in evaluation mode.
        """
        model = get_model(self.model_name, num_classes=self.num_classes)
        model_path = self.save_root / f"{self.model_name}.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load weights
        model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        model.to(self.device).eval()

        self.logger.info(f"Model loaded → {model_path}")
        return model

    def _load_data(self, transform: Any) -> DataLoader:
        """
        Prepares the test data loader.

        Args:
            transform (Any): The preprocessing transform to apply.

        Returns:
            DataLoader: Iterator for the test dataset.
        """
        test_dataset = ClassificationDataset(
            input_dir=self.input_dir,
            split="test",
            transform=transform,
            verbose=False,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False
        )
        self.logger.info(f"Test samples: {len(test_dataset)}")
        return test_loader

    # --------------------------------------------------
    # Execution Logic
    # --------------------------------------------------

    def run(self) -> Tuple[float, float]:
        """
        Executes the evaluation loop on the test set.

        Computes predictions, calculates metrics, saves results, and optionally logs to wandb.

        Returns:
            Tuple[float, float]: (accuracy, f1_score)
        """
        transform = self._get_transform()
        test_loader = self._load_data(transform)
        model = self._load_model()

        y_true = []
        y_pred = []

        # Inference Loop
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                logits = model(imgs)

                # Binary Classification: Sigmoid + Threshold
                if self.num_classes == 1:
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).long().squeeze(1)

                # Multi-class Classification: Softmax + Argmax
                else:
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Metric Calculation
        acc = accuracy_score(y_true, y_pred)

        if self.num_classes == 1:
            f1 = f1_score(y_true, y_pred, average="binary")
        else:
            f1 = f1_score(y_true, y_pred, average="macro")

        self.logger.info(f"Test Accuracy: {acc:.4f}")
        self.logger.info(f"Test F1-score: {f1:.4f}")

        # Persistence & Logging
        self._save_results(y_true, y_pred, acc, f1)

        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "test_accuracy": acc,
                    "test_f1": f1,
                }
            )

        return acc, f1

    def _save_results(
        self, y_true: List[int], y_pred: List[int], acc: float, f1: float
    ) -> None:
        """
        Saves evaluation metrics to JSON and generates a confusion matrix plot.

        Args:
            y_true (List[int]): Ground truth labels.
            y_pred (List[int]): Predicted labels.
            acc (float): Accuracy score.
            f1 (float): F1 score.
        """
        save_dir = self.metric_root / self.model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = save_dir / "metrics.json"
        cm_path = save_dir / "confusion_matrix.png"

        # Save Metrics JSON
        metrics_data = {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "num_samples": len(y_true),
        }

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=4)

        # Generate & Save Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(
            cmap="Blues", values_format="d"
        )
        plt.savefig(cm_path, dpi=200, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Metrics saved → {metrics_path}")
        self.logger.info(f"Confusion matrix saved → {cm_path}")