#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
logging.py
----------
Unified Logging Utility for All Modules

This module provides a standardized logging setup across the entire
pipeline. It creates timestamped log files in a given directory and
streams logs to both console and file outputs.

Example:
    ```python
    from utils.logging import setup_logging, get_logger

    setup_logging("logs/yolo_cropper")
    logger = get_logger(__name__)
    logger.info("This is an info message")
    ```
"""

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO):
    """
    Initialize the logging system for the entire project.

    This function creates a timestamped log file under the specified
    directory, sets a unified format for log messages, and enables
    simultaneous output to both the console and the file.

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

    logging.getLogger().info(f"Logging initialized. Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a named logger instance.

    This function provides a consistent way to get a logger for each
    module, ensuring all loggers share the same global configuration
    set by `setup_logging()`.

    """
    return logging.getLogger(name)
