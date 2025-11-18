#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
load_config.py
--------------
Centralized YAML Configuration Loader

This module provides a single utility function to safely load and parse
YAML configuration files across the entire project. It ensures consistent
error handling and validation of YAML structures.

Example:
    ```python
    from utils.load_config import load_yaml_config

    cfg = load_yaml_config("src/yolo_cropper/config.yaml")
    print(cfg["main"]["input_dir"])
    ```
"""

from pathlib import Path

import yaml


def load_yaml_config(config_path: str | Path) -> dict:
    """
    Load and validate a YAML configuration file.

    This function standardizes how configuration files are read throughout
    the pipeline. It validates file existence, ensures proper YAML structure,
    and returns the parsed content as a dictionary.

    Args:
        config_path (str | Path): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary containing all key–value pairs
        defined in the YAML file.

    """
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"[Config] File not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"[Config] YAML parsing error: {e}")

    if not isinstance(config, dict):
        raise ValueError(
            f"[Config] Invalid YAML structure (expected dict): {config_path}"
        )

    print(f"Loaded configuration from: {config_path}")
    return config
