#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_augmentor.py
-----------------
End-to-end data augmentation pipeline script.

Features:
- Automates dataset splitting and augmentation based on `config.yaml`
- Links `split_dataset` and `augment_dataset` modules for full workflow

Demo mode behavior (main.demo == 'on'):
- input_dir: e.g., data/sample/original
- output_dir: input_dir / "dataset" (e.g., data/sample/original/dataset)
- Original images in input_dir are preserved, and splits are created under `dataset/`.
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.data_augmentor.core.augment_dataset import balance_augmentation
from src.data_augmentor.core.split_dataset import split_dataset
from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging


class DataAugmentor:
    """
    Integrated class for dataset splitting and augmentation.

    This class automates:
        1. Dataset splitting (train/valid/test)
        2. Class balancing via data augmentation
    """

    def __init__(self, config_path: str):
        """
        Initialize DataAugmentor with configuration and logging.

        Args:
            config_path (str): Path to YAML configuration file.
        """
        setup_logging("logs/data_augmentor")
        self.logger = get_logger("DataAugmentor")

        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)

        # ----- Load configuration sections -----
        self.main_cfg = self.cfg.get("main", {})
        augmentor_cfg = self.cfg.get("data_augmentor", {})
        self.data_cfg = augmentor_cfg.get("data", {})
        self.split_cfg = augmentor_cfg.get("split", {})
        self.aug_cfg = augmentor_cfg.get("augmentation", {})
        self.demo_mode = self.main_cfg.get("demo", False)

        self.input_dir = Path(self.data_cfg.get("input_dir", "data/original"))

        if self.demo_mode:
            demo_subdir = self.data_cfg.get("demo_subdir", "dataset")
            self.output_dir = self.input_dir / demo_subdir
        else:
            self.output_dir = Path(
                self.data_cfg.get("output_dir", str(self.input_dir))
            )

        self.logger.info(f"Initialized Data Augmentor Pipeline")
        self.logger.info(f" - Demo Mode  : {self.demo_mode}")
        self.logger.info(f" - Config path: {self.config_path}")
        self.logger.info(f" - Input dir  : {self.input_dir}")
        self.logger.info(f" - Output dir : {self.output_dir}")

    # ============================================================
    # Split Stage
    # ============================================================
    def _run_split(self):
        """Execute dataset splitting into train/valid/test subsets."""
        self.logger.info("\n[1/2] Running Split stage...")
        split_dataset(
            data_dir=self.input_dir,
            output_dir=self.output_dir,
            split_cfg=self.split_cfg,
        )
        self.logger.info("Split completed!")

    # ============================================================
    # 🔹 Augmentation Stage
    # ============================================================
    def _run_augment(self):
        """Run augmentation if enabled in the configuration."""
        if not self.aug_cfg.get("enable", False):
            self.logger.info(
                "\n[2/2] Augmentation disabled (skipped per config.yaml)"
            )
            return

        self.logger.info("\n[2/2] Running class imbalance augmentation...")
        balance_augmentation(self.output_dir, self.aug_cfg)
        self.logger.info("Augmentation completed!")

    # ============================================================
    # 🔹 Cleanup Stage
    # ============================================================
    def _cleanup_original_folders(self):
        """Remove original class folders in full mode (demo=off)."""
        if self.demo_mode:
            self.logger.info("Demo mode: original folders preserved.")
            return
        
        # full mode only cleaning
        self.logger.info("Full mode: removing original class folders...")
        protected = {"train", "valid", "test"}

        for item in self.input_dir.iterdir():
            if not item.is_dir():
                continue
            if item.name in protected:
                continue
            shutil.rmtree(item)
            self.logger.info(f"Removed original folder: {item}")    

    # ============================================================
    # 🔹 Full Execution
    # ============================================================
    def run(self):
        """
        Execute the full pipeline:
        1. Split dataset
        2. Perform augmentation (optional)
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Input data directory not found: {self.input_dir}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("\n[DataAugmentor] Starting pipeline")
        self.logger.info(f" - Split ratios: {self.split_cfg}")
        self.logger.info(
            f" - Augmentation: {'Enabled' if self.aug_cfg.get('enable', False) else 'Disabled'}"
        )

        self._run_split()
        self._run_augment()
        self._cleanup_original_folders()

        self.logger.info("\n Augmentor pipeline completed successfully!")