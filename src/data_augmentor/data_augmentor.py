#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_augmentor.py
-----------------
End-to-end data augmentation pipeline script.

Features:
- Automates dataset splitting and augmentation based on `config.yaml`
- Links `split_dataset` and `augment_dataset` modules for full workflow
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

        # Load configuration sections
        augmentor_cfg = self.cfg.get("data_augmentor", {})
        self.data_cfg = augmentor_cfg.get("data", {})
        self.split_cfg = augmentor_cfg.get("split", {})
        self.aug_cfg = augmentor_cfg.get("augmentation", {})

        # Resolve input/output directories
        self.input_dir = Path(self.data_cfg.get("input_dir", "data/original"))
        self.output_dir = Path(self.data_cfg.get("output_dir", "data/original"))

        self.logger.info(f"Config loaded from: {self.config_path}")
        self.logger.info(f"Input dir : {self.input_dir}")
        self.logger.info(f"Output dir: {self.output_dir}")

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

    def _cleanup_original_classes(self):
        """Remove original class folders after splitting."""
        self.logger.info("\n[Cleanup] Removing original class directories...")
        for cls in ["repair", "replace"]:
            target = self.output_dir / cls
            if target.exists():
                try:
                    shutil.rmtree(target)
                    self.logger.info(f"Deleted {target}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete {target}: {e}")
        self.logger.info("Cleanup completed!")

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
    # 🔹 Full Execution
    # ============================================================
    def run(self):
        """
        Execute the full pipeline:
        1. Split dataset
        2. Clean up original folders
        3. Perform augmentation (optional)
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
        self._cleanup_original_classes()
        self._run_augment()

        self.logger.info("\n🎉 Augmentor pipeline completed successfully!")


# ============================================================
# ✅ CLI Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DataAugmentor pipeline")
    parser.add_argument("--config", default="./utils/config.yaml")
    args = parser.parse_args()

    augmentor = DataAugmentor(config_path=args.config)
    augmentor.run()
