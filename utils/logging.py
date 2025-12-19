#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Centralized Logging Utility.

Configures a unified logging system that streams output to both
the console and timestamped log files.
"""

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> None:
    """
    Initializes the root logger with FileHandler and StreamHandler.

    Args:
        log_dir (str): Directory where timestamped log files will be saved.
        log_level (int): Threshold for logging (default: logging.INFO).
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=fmt,
        datefmt=datefmt,
        force=True,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    logging.getLogger().info(f"Logging initialized. Output: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance with the specified name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)