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
from src.classifier.data.data_preprocessor import DataPreprocessor
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
        self.global_main_cfg = self.cfg_all.get("main", {})
        self.cfg = self.cfg_all["classifier"]

        # Split sub-sections
        self.data_cfg = self.cfg.get("data", {})
        self.train_cfg = self.cfg.get("train", {})
        self.wandb_cfg = self.cfg.get("wandb", {})

        # Extract values
        self.model_name = self.global_main_cfg.get("classify_model", "mobilenet_v2").lower()
        self.use_wandb = self.wandb_cfg.get("enabled", True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = self.train_cfg.get("epochs", 80)

        # paths
        self.input_dir = Path(self.data_cfg.get("input_dir", "data/original"))
        self.save_dir = self.train_cfg["save_dir"]
        self.metric_dir = self.train_cfg["metric_dir"]
        self.check_dir = self.train_cfg["check_dir"]
        self.weights_save_path = os.path.join(self.save_dir, f"{self.model_name}.pt")
        self.check_path = os.path.join(self.check_dir, f"{self.model_name}_last.pt")

        self.categories = self.global_main_cfg.get("categories", [])
        self.num_classes = len(self.categories)

        self.criterion_name = self.cfg.get("criterion", "auto")

        if self.criterion_name == "auto":
            self.criterion_name = "bce" if self.num_classes == 2 else "ce"

        if self.criterion_name == "bce":
            self.output_dim = 1
        else:
            self.output_dim = self.num_classes

        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Input Dir: {self.input_dir}")


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

    # Data Loading
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
            train_dataset, batch_size=bs, shuffle=True, num_workers=0
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=bs, shuffle=False, num_workers=0
        )

        self.logger.info(
            f"Data Loaded: train={len(train_dataset)}, valid={len(valid_dataset)}"
        )
        return train_loader, valid_loader

    # Model & Optimizer Setup
    def _build_model(self):
        # Model output dimension
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
        self, model, train_loader, valid_loader, criterion, optimizer, device, epochs, save_path, check_path, num_classes, wandb_run
    ):
        """
        Wrapper function for training loop.

        Saves checkpoints and returns best validation accuracy.
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
            num_classes = num_classes,
            wandb_run=wandb_run,
        )
        return best_acc
    
    # ==========================================================
    # Step 2. Evaluate
    # ==========================================================
    def step_evaluate(self, model, input_dir, save_dir, metric_dir, num_classes, wandb_run):
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
    # Entrypoint
    # ==========================================================
    def run(self):
        self.logger.info(
            f"Start Training {self.model_name.upper()} on {self.input_dir}"
        )
        train_loader, valid_loader = self._load_data()
        model, criterion, optimizer = self._build_model()
        wandb_run = self._init_wandb()

        # Training phase
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
            wandb_run=wandb_run
        )

        # Evaluation phase
        acc, f1 = self.step_evaluate(
        model=self.model_name,
        input_dir=self.input_dir,
        save_dir=self.save_dir,
        metric_dir=self.metric_dir,
        num_classes=self.output_dim,
        wandb_run=wandb_run
        )

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

        self.logger.info("Classifier Pipeline Finished Successfully")
        return best_acc, acc, f1


# ======================================================
# Standalone Entrypoint
# ======================================================
def main():
    """
    Standalone entrypoint for Classifier pipeline.

    Example:
        python src/classifier/classifier.py --config utils/config.yaml
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