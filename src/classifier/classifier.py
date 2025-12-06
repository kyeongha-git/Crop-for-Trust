#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
classifier.py
-------------
Main entry point for training and evaluating classification models.

This script builds an end-to-end training–evaluation pipeline driven by
`config.yaml`. It automatically configures data loading, model setup,
optimizer initialization, checkpoint management, and optional Weights & Biases (wandb)
logging for experiment tracking.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.classifier.data.cnn_data_loader import ClassificationDataset
from src.classifier.data.data_preprocessing import DataPreprocessor
from src.classifier.evaluate import Evaluator
from src.classifier.models.factory import get_model
from src.classifier.train import train_model
from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging


class Classifier:
    """
    Unified pipeline manager for training and evaluating classification models.

    Features:
        - Auto configuration via `config.yaml`
        - Model and data directory resolution
        - wandb integration for tracking
        - Full train → validation → test flow
    """

    def __init__(self, config_path: str):
        """
        Initialize classifier with configuration and logging setup.

        Args:
            config_path (str): Path to YAML configuration file.
        """
        setup_logging("logs/classifier")
        self.logger = get_logger("classifier")

        # Load configuration file
        self.config_path = Path(config_path)
        self.cfg_all = load_yaml_config(self.config_path)

        if "classifier" not in self.cfg_all:
            raise KeyError("Missing 'Classifier' section in config.yaml.")
        self.cfg = self.cfg_all["classifier"]

        # Split sub-sections
        self.data_cfg = self.cfg.get("data", {})
        self.train_cfg = self.cfg.get("train", {})
        self.wandb_cfg = self.cfg.get("wandb", {})

        # Extract values
        self.input_dir = Path(self.data_cfg.get("input_dir", "data/original"))
        self.model_name = self.train_cfg.get("model_name", "mobilenet_v2").lower()
        self.use_wandb = self.wandb_cfg.get("enabled", True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Automatically resolve output paths
        self._resolve_output_paths()

        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Input Dir: {self.input_dir}")

    # ==========================================================
    # Path Resolution
    # ==========================================================
    def _resolve_output_paths(self):
        """
        Dynamically expand save, metric, and checkpoint directories
        according to the input dataset path specified in config.
        """
        input_dir = str(self.input_dir).replace("\\", "/")
        if input_dir.startswith("data/"):
            data_subpath = input_dir[len("data/") :]  # e.g., "original_crop/yolov2"
        else:
            data_subpath = Path(input_dir).name  # e.g., "original_"

        # Base directories
        base_save = self.train_cfg.get("save_dir", "./saved_model/classifier")
        base_metric = self.train_cfg.get("metric_dir", "./metrics/classifier")
        base_check = self.train_cfg.get("check_dir", "./checkpoints/classifier")

        # Resolved directories
        resolved_save = os.path.join(base_save, data_subpath)
        resolved_metric = os.path.join(base_metric, data_subpath)
        resolved_check = os.path.join(base_check, data_subpath)

        # Update config with resolved paths
        self.train_cfg["save_dir"] = resolved_save
        self.train_cfg["metric_dir"] = resolved_metric
        self.train_cfg["check_dir"] = resolved_check

        os.makedirs(resolved_save, exist_ok=True)
        os.makedirs(resolved_metric, exist_ok=True)
        os.makedirs(resolved_check, exist_ok=True)

        self.logger.info(f"Resolved Save Dir: {resolved_save}")
        self.logger.info(f"Resolved Metric Dir: {resolved_metric}")
        self.logger.info(f"Resolved Check Dir: {resolved_check}")

    # ==========================================================
    # wandb Initialization
    # ==========================================================
    def _init_wandb(self):
        """
        Initialize a wandb run session for experiment tracking.
        Returns None if disabled.
        """
        if not self.use_wandb:
            self.logger.warning("wandb logging disabled.")
            return None

        project_root = ROOT_DIR
        wandb_dir = project_root / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)

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

    # ==========================================================
    # Data Loading
    # ==========================================================
    def _load_data(self):
        """Load train and validation datasets based on config paths."""
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
            train_dataset, batch_size=bs, shuffle=True, num_workers=2
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=bs, shuffle=False, num_workers=2
        )

        self.logger.info(
            f"Data Loaded: train={len(train_dataset)}, valid={len(valid_dataset)}"
        )
        return train_loader, valid_loader

    # ==========================================================
    # Model & Optimizer Setup
    # ==========================================================
    def _build_model(self):
        """Initialize model, loss function, and optimizer."""
        model = get_model(self.model_name, num_classes=1).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(
            model.parameters(),
            lr=self.train_cfg.get("lr", 0.001),
            weight_decay=self.train_cfg.get("weight_decay", 1e-5),
        )
        return model, criterion, optimizer

    # ==========================================================
    # Training Wrapper
    # ==========================================================
    def _train_model(
        self, model, criterion, optimizer, train_loader, valid_loader, wandb_run
    ):
        """
        Wrapper function for training loop.

        Saves checkpoints and returns best validation accuracy.
        """
        save_dir = self.train_cfg["save_dir"]
        check_dir = self.train_cfg["check_dir"]

        save_path = os.path.join(save_dir, f"{self.model_name}.pt")
        check_path = os.path.join(check_dir, f"{self.model_name}_last.pt")

        self.logger.info(f"Save Path: {save_path}")
        self.logger.info(f"Checkpoint Dir: {check_dir}")

        best_acc = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            epochs=self.train_cfg.get("epochs", 80),
            save_path=save_path,
            check_path=check_path,
            wandb_run=wandb_run,
        )

        self.logger.info(f"Training Complete — Best Val Acc: {best_acc:.4f}")
        return best_acc

    # ==========================================================
    # Full Pipeline (Train + Evaluate)
    # ==========================================================
    def run(self):
        """
        Execute the complete training and evaluation pipeline.

        Steps:
            1. Load datasets
            2. Build model and optimizer
            3. Train model and track via wandb
            4. Evaluate final test performance
            5. Save metrics and finish session
        """
        self.logger.info(
            f"Start Training {self.model_name.upper()} on {self.input_dir}"
        )
        train_loader, valid_loader = self._load_data()
        model, criterion, optimizer = self._build_model()
        wandb_run = self._init_wandb()

        # Training phase
        best_acc = self._train_model(
            model, criterion, optimizer, train_loader, valid_loader, wandb_run
        )

        # Evaluation phase
        evaluator = Evaluator(
            input_dir=self.input_dir,
            model=self.model_name,
            cfg=self.cfg,
            wandb_run=wandb_run,
        )
        acc, f1 = evaluator.run()

        # Log and finalize
        if wandb_run:
            wandb_run.log(
                {
                    "final_best_acc": float(best_acc),
                    "test_acc": float(acc),
                    "test_f1": float(f1),
                }
            )
            wandb_run.finish()

        self.logger.info("🎉 Classifier Pipeline Finished Successfully")
        return best_acc, acc, f1


# ==========================================================
# CLI Entry Point
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifier Training Entry Point")
    parser.add_argument(
        "--config_path",
        type=str,
        default="utils/config.yaml",
        help="Path to configuration YAML file (default: utils/config.yaml)",
    )
    args = parser.parse_args()

    clf = Classifier(config_path=args.config_path)
    clf.run()
