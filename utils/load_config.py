#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YAML Configuration Loader.

Provides a centralized utility to safe-load and validate YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Loads and validates a YAML configuration file.

    Args:
        config_path (str | Path): Path to the configuration file.

    Returns:
        Dict[str, Any]: The parsed configuration dictionary.
    """
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML parsing error: {e}")

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML structure (expected dict): {config_path}")

    print(f"Loaded configuration: {config_path}")
    return config