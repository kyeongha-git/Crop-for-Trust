#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_augmentor.py

Orchestrates the data splitting and augmentation pipeline.

This module manages the transition from raw data to a prepared dataset by:
1. Splitting data into train/validation/test sets.
2. Applying augmentation for class balancing.
3. Handling directory cleanup based on the execution mode (Demo vs Production).
"""

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
    Controller for the dataset splitting and augmentation pipeline.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the augmentor with configuration settings.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        setup_logging("logs/data_augmentor")
        self.logger = get_logger("DataAugmentor")

        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)

        # Load configuration sections
        self.main_cfg = self.cfg.get("main", {})
        augmentor_cfg = self.cfg.get("data_augmentor", {})
        self.data_cfg = augmentor_cfg.get("data", {})
        self.split_cfg = augmentor_cfg.get("split", {})
        self.aug_cfg = augmentor_cfg.get("augmentation", {})
        self.demo_mode = self.main_cfg.get("demo", False)

        self.input_dir = Path(self.data_cfg.get("input_dir", "data/original"))

        # Configure output directory based on mode
        if self.demo_mode:
            demo_subdir = self.data_cfg.get("demo_subdir", "dataset")
            self.output_dir = self.input_dir / demo_subdir
        else:
            self.output_dir = Path(
                self.data_cfg.get("output_dir", str(self.input_dir))
            )

        self.logger.info("Initialized Data Augmentor Pipeline")
        self.logger.info(f" - Config    : {self.config_path}")
        self.logger.info(f" - Input Dir : {self.input_dir}")
        self.logger.info(f" - Output Dir: {self.output_dir}")
        self.logger.info(f" - Demo Mode : {self.demo_mode}")

    # ============================================================
    # Step 1. Data Split
    # ============================================================
    def step_split(self) -> None:
        """
        Splits the raw dataset into train, validation, and test subsets.
        """
        self.logger.info("[STEP 1] Starting Data Split...")
        split_dataset(
            data_dir=self.input_dir,
            output_dir=self.output_dir,
            split_cfg=self.split_cfg,
        )

    # ============================================================
    # Step 2. Data Augmentation
    # ============================================================
    def step_augment(self) -> None:
        """
        Applies data augmentation to balance class distributions.
        Skips if augmentation is disabled in the configuration.
        """
        if not self.aug_cfg.get("enable", False):
            self.logger.info("[STEP 2] Augmentation disabled in config. Skipping.")
            return

        self.logger.info("[STEP 2] Starting Data Augmentation...")
        balance_augmentation(self.output_dir, self.aug_cfg)

    # ============================================================
    # Step 3. Cleanup
    # ============================================================
    def step_cleanup(self) -> None:
        """
        Removes original source directories to maintain a clean dataset structure.
        Cleanup is skipped in 'demo' mode to preserve original data.
        """
        if self.demo_mode:
            self.logger.info("[STEP 3] Demo mode enabled. Skipping cleanup.")
            return
        
        self.logger.info("[STEP 3] Starting Cleanup...")
        protected = {"train", "valid", "test"}

        # Remove all directories except the split results
        for item in self.input_dir.iterdir():
            if not item.is_dir():
                continue
            if item.name in protected:
                continue
            
            try:
                shutil.rmtree(item)
                self.logger.info(f"Removed source directory: {item.name}")
            except Exception as e:
                self.logger.warning(f"Failed to remove {item.name}: {e}")

    # ============================================================
    # Pipeline Execution
    # ============================================================
    def run(self) -> None:
        """
        Executes the full augmentation pipeline.
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Input data directory not found: {self.input_dir}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Pipeline execution started.")
        self.step_split()
        self.step_augment()
        self.step_cleanup()
        self.logger.info("Pipeline completed successfully.")