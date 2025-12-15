#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cfg_manager.py
--------------
This module manages the generation of Darknet configuration files (.cfg)
for YOLOv2 and YOLOv4 models. It reads a YAML configuration provided by
the main controller, copies a base cfg file, applies user-defined overrides,
and automatically updates class and filter values.
"""

import shutil
import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[5]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class CfgManager:
    """
    Generates and customizes Darknet cfg files for YOLOv2 or YOLOv4.

    The class receives a complete YAML configuration and creates a new
    Darknet configuration file by copying the base cfg and applying
    all user-defined overrides automatically.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the configuration manager.

        Args:
            config (Dict[str, Any]): Full YAML configuration dictionary from the main controller.

        """
        self.logger = get_logger("yolo_cropper.CfgManager")

        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.darknet_cfg = self.yolo_cropper_cfg.get("darknet", {})
        self.cfg_overrides = self.darknet_cfg.get("cfg_overrides", {})
        self.model_name = self.main_cfg.get("model_name", "yolov2").lower().strip()

        if self.model_name not in ("yolov2", "yolov4"):
            raise ValueError("model_name must be either 'yolov2' or 'yolov4'")

        darknet_root = Path(
            self.darknet_cfg.get("darknet_dir", "third_party/darknet")
        ).resolve()

        self.base_cfg = darknet_root / "cfg" / f"{self.model_name}.cfg"
        self.target_cfg = darknet_root / "cfg" / f"{self.model_name}-obj.cfg"

        if not self.base_cfg.exists():
            raise FileNotFoundError(f"Base cfg not found: {self.base_cfg}")

        self.target_cfg.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized CfgManager for {self.model_name.upper()}")
        self.logger.debug(f"Base cfg   : {self.base_cfg}")
        self.logger.debug(f"Target cfg : {self.target_cfg}")

    def _copy_base_cfg(self) -> None:
        """
        Copy the base Darknet cfg file to create a new training cfg file.
        """
        shutil.copy(self.base_cfg, self.target_cfg)
        self.logger.info(f"Copied base {self.model_name}.cfg → {self.target_cfg}")

    def _apply_cfg_overrides(self) -> None:
        """
        Apply user-defined parameter overrides from the YAML configuration.

        This method updates values such as batch size, image dimensions,
        and other parameters defined under the [net] section of the cfg file.
        """
        overrides = self.cfg_overrides
        if not overrides:
            self.logger.warning("No 'cfg_overrides' found in config.")
            return

        with open(self.target_cfg, "r", encoding="utf-8") as f:
            lines = f.readlines()

        def replace_line(lines, key, val):
            for i, line in enumerate(lines):
                if line.strip().replace(" ", "").startswith(f"{key}="):
                    lines[i] = f"{key}={val}\n"
            return lines

        in_net = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[net]"):
                in_net = True
                continue
            elif stripped.startswith("[") and not stripped.startswith("[net]"):
                in_net = False
            if in_net:
                for k, v in overrides.items():
                    if stripped.replace(" ", "").startswith(f"{k}="):
                        lines[i] = f"{k}={v}\n"

        for k, v in overrides.items():
            lines = replace_line(lines, k, v)

        with open(self.target_cfg, "w", encoding="utf-8") as f:
            f.writelines(lines)

        self.logger.info("Applied cfg_overrides from config.yaml")

    def _update_filters_and_classes(self) -> None:
        """
        Update the 'filters' and 'classes' values in the cfg file.

        The number of filters is computed automatically based on the number
        of classes and anchor boxes used by YOLOv2 or YOLOv4.
        """
        overrides = self.cfg_overrides
        num_classes = int(overrides.get("classes", 2))

        if self.model_name == "yolov2":
            filters = (num_classes + 5) * 5
            marker = "[region]"
        else:
            filters = (num_classes + 5) * 3
            marker = "[yolo]"

        with open(self.target_cfg, "r", encoding="utf-8") as f:
            lines = f.readlines()

        updated = lines.copy()

        for i, line in enumerate(lines):
            if line.strip().startswith(marker):
                j = i - 1
                while j >= 0 and not lines[j].strip().startswith("[convolutional]"):
                    j -= 1
                if j >= 0:
                    for k in range(j, i):
                        if lines[k].strip().startswith("filters="):
                            updated[k] = f"filters={filters}\n"
                            break
                for k in range(i, len(lines)):
                    if lines[k].strip().startswith("classes="):
                        updated[k] = f"classes={num_classes}\n"
                        break

        with open(self.target_cfg, "w", encoding="utf-8") as f:
            f.writelines(updated)

        self.logger.info(
            f"Updated filters={filters}, classes={num_classes} ({self.model_name.upper()})"
        )

    def run(self) -> str:
        """
        Generate the final Darknet cfg file.

        Steps:
            1. Copy the base cfg file.
            2. Apply all user-defined overrides.
            3. Update filters and class counts automatically.

        Returns:
            str: Path to the generated cfg file.
        """
        self.logger.info(f"Generating cfg for {self.model_name.upper()} ...")
        self._copy_base_cfg()
        self._apply_cfg_overrides()
        self._update_filters_and_classes()
        self.logger.info(f"Config generated → {self.target_cfg}")
        return str(self.target_cfg)
