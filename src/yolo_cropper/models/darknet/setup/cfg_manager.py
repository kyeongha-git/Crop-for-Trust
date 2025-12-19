#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Darknet Configuration Manager.

Generates model-specific configuration files (.cfg) for YOLOv2 and YOLOv4
by applying user-defined overrides and dataset-specific parameters to base templates.
"""

import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[5]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class CfgManager:
    """
    Manages the creation and customization of Darknet configuration files.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = get_logger("yolo_cropper.CfgManager")

        self.cfg = config
        global_main_cfg = self.cfg.get("main", {})
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.darknet_cfg = self.yolo_cropper_cfg.get("darknet", {})
        self.cfg_overrides = self.darknet_cfg.get("cfg_overrides", {})
        self.model_name = self.main_cfg.get("model_name", "yolov2").lower().strip()

        self.categories = global_main_cfg.get("categories", [])
        if not self.categories:
            raise ValueError("Categories must be defined in configuration.")

        self.num_classes = len(self.categories)

        if self.model_name not in ("yolov2", "yolov4"):
            raise ValueError("model_name must be 'yolov2' or 'yolov4'")

        darknet_root = Path(
            self.darknet_cfg.get("darknet_dir", "third_party/darknet")
        )

        self.base_cfg = (darknet_root / "cfg" / f"{self.model_name}.cfg").resolve()
        self.target_cfg = (darknet_root / "cfg" / f"{self.model_name}-obj.cfg").resolve()

        if not self.base_cfg.exists():
            raise FileNotFoundError(f"Base cfg not found: {self.base_cfg}")

        self.target_cfg.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized CfgManager (Model: {self.model_name.upper()})")

    def _copy_base_cfg(self) -> None:
        """
        Copies the base configuration template to the target path.
        """
        shutil.copy(self.base_cfg, self.target_cfg)
        self.logger.debug(f"Copied base cfg to {self.target_cfg}")

    def _apply_cfg_overrides(self) -> None:
        """
        Applies user-defined parameter overrides (e.g., batch size) to the configuration file.
        """
        overrides = self.cfg_overrides
        if not overrides:
            return

        with open(self.target_cfg, "r", encoding="utf-8") as f:
            lines = f.readlines()

        def replace_line(lines: List[str], key: str, val: Any) -> List[str]:
            for i, line in enumerate(lines):
                if line.strip().replace(" ", "").startswith(f"{key}="):
                    lines[i] = f"{key}={val}\n"
            return lines

        # Update [net] section and global overrides
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

        self.logger.info("Applied configuration overrides")

    def _update_filters_and_classes(self) -> None:
        """
        Updates 'filters' and 'classes' parameters based on the number of categories.
        """
        num_classes = self.num_classes

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
                # Backtrack to find the preceding [convolutional] layer
                j = i - 1
                while j >= 0 and not lines[j].strip().startswith("[convolutional]"):
                    j -= 1
                if j >= 0:
                    for k in range(j, i):
                        if lines[k].strip().startswith("filters="):
                            updated[k] = f"filters={filters}\n"
                            break
                # Update classes definition
                for k in range(i, len(lines)):
                    if lines[k].strip().startswith("classes="):
                        updated[k] = f"classes={num_classes}\n"
                        break

        with open(self.target_cfg, "w", encoding="utf-8") as f:
            f.writelines(updated)

        self.logger.info(f"Updated schema: filters={filters}, classes={num_classes}")

    def run(self) -> str:
        """
        Executes the configuration generation pipeline.

        Returns:
            str: Path to the generated configuration file.
        """
        self.logger.info(f"Generating configuration for {self.model_name.upper()}")
        
        self._copy_base_cfg()
        self._apply_cfg_overrides()
        self._update_filters_and_classes()
        
        self.logger.info(f"Configuration generated: {self.target_cfg}")
        return str(self.target_cfg)