#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
classifier.py

Main entry point for the training and evaluation pipeline.
This module orchestrates data loading, model initialization, training loops,
and evaluation procedures based on the YAML configuration.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.classifier.data.cnn_data_loader import ClassificationDataset
from src.classifier.data.data_preprocessor import DataPreprocessor
from src.classifier.evaluate import Evaluator
from src.classifier.models.factory import get_model
from src.classifier.train import train_model
from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging


class Classifier:
    """
    Unified pipeline manager for training and evaluating classification models.

    Handles:
        - Configuration loading and path resolution
        - Data loading and preprocessing
        - Model construction and optimization setup
        - Execution of training and evaluation loops
        - Experiment tracking via Weights & Biases (wandb)
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the classifier pipeline.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        setup_logging("logs/classifier")
        self.logger = get_logger("classifier")

        # Load configuration
        self.config_path = Path(config_path)
        self.cfg_all = load_yaml_config(self.config_path)

        if "classifier" not in self.cfg_all:
            raise KeyError("Missing 'Classifier' section in config.yaml.")
        
        self.global_main_cfg = self.cfg_all.get("main", {})
        self.cfg = self.cfg_all["classifier"]

        # Config Sub-sections
        self.data_cfg = self.cfg.get("data", {})
        self.train_cfg = self.cfg.get("train", {})
        self.wandb_cfg = self.cfg.get("wandb", {})

        # Training Parameters
        self.model_name = self.global_main_cfg.get("classify_model", "mobilenet_v2").lower()
        self.use_wandb = self.wandb_cfg.get("enabled", True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = self.train_cfg.get("epochs", 80)

        # Paths
        self.input_dir = Path(self.data_cfg.get("input_dir", "data/original"))
        self.save_dir = self.train_cfg["save_dir"]
        self.metric_dir = self.train_cfg["metric_dir"]
        self.check_dir = self.train_cfg["check_dir"]
        self.weights_save_path = os.path.join(self.save_dir, f"{self.model_name}.pt")
        self.check_path = os.path.join(self.check_dir, f"{self.model_name}_last.pt")

        # Dataset Metadata
        self.categories = self.global_main_cfg.get("categories", [])
        self.num_classes = len(self.categories)

        # Loss Criterion Setup
        self.criterion_name = self.cfg.get("criterion", "auto")

        if self.criterion_name == "auto":
            self.criterion_name = "bce" if self.num_classes == 2 else "ce"

        if self.criterion_name == "bce":
            self.output_dim = 1
        else:
            self.output_dim = self.num_classes

        self.logger.info(f"Device    : {self.device}")
        self.logger.info(f"Model     : {self.model_name}")
        self.logger.info(f"Input Dir : {self.input_dir}")

    # ==========================================================
    # System & Tool Initialization
    # ==========================================================

    def _init_wandb(self) -> Optional[Any]:
        """
        Initializes a wandb run session for experiment tracking.

        Returns:
            Optional[Any]: The wandb run object if enabled, else None.
        """
        if not self.use_wandb:
            self.logger.warning("wandb logging disabled.")
            return None

        project_root = ROOT_DIR
        wandb_dir = project_root / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)

        # Generate a distinct run name based on input data path
        input_dir_rel = str(Path(self.input_dir).as_posix())
        data_subpath = (
            "/".join(Path(input_dir_rel).parts[-2:])
            if len(Path(input_dir_rel).parts) >= 2
            else Path(input_dir_rel).name
        )
        data_subpath = data_subpath.replace("data/", "").replace("/", "_")

        run_name_pattern = self.wandb_cfg.get("run_name_pattern", "{model}_{input_dir}")
        run_name = run_name_pattern.format(
            model=self.model_name, input_dir=data_subpath
        )

        wandb_run = wandb.init(
            project=self.wandb_cfg.get("project", "default-project"),
            entity=self.wandb_cfg.get("entity", None),
            name=run_name,
            config=self.cfg,
            dir=str(wandb_dir),
            notes=self.wandb_cfg.get("notes", None),
        )

        self.logger.info(f"wandb initialized: {run_name}")
        return wandb_run

    def _load_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Loads train and validation datasets.

        Returns:
            Tuple[DataLoader, DataLoader]: (train_loader, valid_loader)
        """
        dp = DataPreprocessor()
        train_tf = dp.get_transform(self.model_name, "train")
        eval_tf = dp.get_transform(self.model_name, "eval")

        base_dir = self.input_dir
        bs = self.train_cfg.get("batch_size", 32)

        train_dataset = ClassificationDataset(
            base_dir, split="train", transform=train_tf, verbose=False
        )
        valid_dataset = ClassificationDataset(
            base_dir, split="valid", transform=eval_tf, verbose=False
        )

        train_loader = DataLoader(
            train_dataset, batch_size=bs, shuffle=True, num_workers=0
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=bs, shuffle=False, num_workers=0
        )

        self.logger.info(
            f"Data Loaded: train={len(train_dataset)}, valid={len(valid_dataset)}"
        )
        return train_loader, valid_loader

    def _build_model(self) -> Tuple[nn.Module, nn.Module, Any]:
        """
        Builds the model, loss function, and optimizer.

        Returns:
            Tuple[nn.Module, nn.Module, Any]: (model, criterion, optimizer)
        """
        # Configure model output dimension
        if self.criterion_name == "bce":
            model = get_model(self.model_name, num_classes=1).to(self.device)
            criterion = nn.BCEWithLogitsLoss()
        elif self.criterion_name == "ce":
            model = get_model(self.model_name, num_classes=self.num_classes).to(self.device)
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion: {self.criterion_name}")

        optimizer = AdamW(
            model.parameters(),
            lr=self.train_cfg.get("lr", 0.001),
            weight_decay=self.train_cfg.get("weight_decay", 1e-5),
        )

        self.logger.info(
            f"Classifier setup | num_classes={self.num_classes}, criterion={self.criterion_name.upper()}"
        )

        return model, criterion, optimizer

    # ==========================================================
    # Step 1. Train
    # ==========================================================

    def step_train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Any,
        device: torch.device,
        epochs: int,
        save_path: str,
        check_path: str,
        num_classes: int,
        wandb_run: Optional[Any],
    ) -> float:
        """
        Executes the training loop.

        Returns:
            float: Best validation accuracy achieved during training.
        """
        best_acc = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            save_path=save_path,
            check_path=check_path,
            num_classes=num_classes,
            wandb_run=wandb_run,
        )
        return best_acc

    # ==========================================================
    # Step 2. Evaluate
    # ==========================================================

    def step_evaluate(
        self,
        model: Union[str, nn.Module],
        input_dir: Path,
        save_dir: str,
        metric_dir: str,
        num_classes: int,
        wandb_run: Optional[Any],
    ) -> Tuple[float, float]:
        """
        Runs the evaluation protocol.

        Returns:
            Tuple[float, float]: (test_accuracy, test_f1_score)
        """
        evaluator = Evaluator(
            model=model,
            input_dir=input_dir,
            save_dir=save_dir,
            metric_dir=metric_dir,
            num_classes=num_classes,
            wandb_run=wandb_run,
        )
        acc, f1 = evaluator.run()
        return acc, f1

    # ==========================================================
    # Main Execution Entrypoint
    # ==========================================================

    def run(self) -> Tuple[float, float, float]:
        """
        Runs the complete classifier pipeline (Train -> Evaluate).

        Returns:
            Tuple[float, float, float]: (best_val_acc, test_acc, test_f1)
        """
        self.logger.info(
            f"Start Training {self.model_name.upper()} on {self.input_dir}"
        )
        train_loader, valid_loader = self._load_data()
        model, criterion, optimizer = self._build_model()
        wandb_run = self._init_wandb()

        # Phase 1: Training
        best_acc = self.step_train(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            epochs=self.epochs,
            save_path=self.weights_save_path,
            check_path=self.check_path,
            num_classes=self.output_dim,
            wandb_run=wandb_run,
        )

        # Phase 2: Evaluation
        acc, f1 = self.step_evaluate(
            model=self.model_name,
            input_dir=self.input_dir,
            save_dir=self.save_dir,
            metric_dir=self.metric_dir,
            num_classes=self.output_dim,
            wandb_run=wandb_run,
        )

        # Finalize
        if wandb_run:
            wandb_run.log(
                {
                    "final_best_acc": float(best_acc),
                    "test_acc": float(acc),
                    "test_f1": float(f1),
                }
            )
            wandb_run.finish()

        self.logger.info("Classifier Pipeline Finished Successfully")
        return best_acc, acc, f1


# ======================================================
# CLI Entry Point
# ======================================================

def main() -> None:
    """
    Parses arguments and runs the standalone classifier pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Standalone Classifier Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="utils/config.yaml",
        help="Path to config.yaml",
    )

    args = parser.parse_args()

    setup_logging("logs/classifier")
    logger = get_logger("classifier.main")

    logger.info("Starting standalone Classifier execution")
    logger.info(f"Using config: {args.config}")

    try:
        classifier = Classifier(config_path=args.config)
        best_acc, test_acc, test_f1 = classifier.run()

        logger.info(
            f"Classifier finished successfully | "
            f"Best Val Acc: {best_acc:.4f}, "
            f"Test Acc: {test_acc:.4f}, "
            f"Test F1: {test_f1:.4f}"
        )

    except Exception:
        logger.exception("Classifier execution failed")
        raise


if __name__ == "__main__":
    main()