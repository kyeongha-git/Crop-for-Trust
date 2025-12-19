#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Darknet Build Manager.

Configures the Makefile based on the selected build mode (CPU/GPU)
and orchestrates the compilation of the Darknet executable.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[5]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class MakeManager:
    """
    Manages the configuration and compilation of the Darknet source code.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = get_logger("yolo_cropper.MakeManager")

        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.darknet_cfg = self.yolo_cropper_cfg.get("darknet", {})

        darknet_root = Path(
            self.darknet_cfg.get("darknet_dir", "third_party/darknet")
        )
        self.darknet_dir = darknet_root.resolve()
        self.makefile = self.darknet_dir / "Makefile"

        if not self.makefile.exists():
            raise FileNotFoundError(f"Makefile not found in {self.darknet_dir}")

        self.build_mode = self.darknet_cfg.get("build_mode", "cpu").lower()
        self.mode_flags = self.darknet_cfg.get("modes", {}).get(self.build_mode)

        if not self.mode_flags:
            valid_modes = list(self.darknet_cfg.get("modes", {}).keys())
            raise ValueError(
                f"Invalid build_mode '{self.build_mode}'. Available: {valid_modes}"
            )

        self.jobs = self.mode_flags.get("MAKE_JOBS", 4)
        self.logger.info(f"Initialized MakeManager (Mode: {self.build_mode.upper()})")

    def configure(self, quiet: bool = True) -> None:
        """
        Updates Makefile flags based on the active build configuration.
        """
        self.logger.info(f"Configuring Makefile for {self.build_mode.upper()}")

        patch_flags = {k: v for k, v in self.mode_flags.items() if k != "MAKE_JOBS"}
        self._patch_makefile(patch_flags, quiet=quiet)

    def rebuild(self, quiet: bool = True) -> None:
        """
        Cleans and recompiles the Darknet binaries.
        """
        self.logger.info("Rebuilding Darknet...")

        def _run(cmd):
            subprocess.run(
                cmd,
                cwd=self.darknet_dir,
                check=True,
                stdout=subprocess.DEVNULL if quiet else None,
                stderr=subprocess.DEVNULL if quiet else None,
            )

        _run(["make", "clean"])
        _run(["make", f"-j{self.jobs}"])
        
        self.logger.info(f"Build completed (Jobs: {self.jobs})")

    def verify_darknet(self, quiet: bool = True) -> None:
        """
        Validates the generated executable.
        """
        darknet_exec = self.darknet_dir / "darknet"
        if not darknet_exec.exists():
            raise FileNotFoundError("Darknet executable not found after build.")

        # Dry run to verify execution permissions and integrity
        subprocess.run(
            ["./darknet"],
            cwd=self.darknet_dir,
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=subprocess.DEVNULL if quiet else None,
            check=True,
        )
        self.logger.info("Darknet executable verified")

    def _patch_makefile(self, flags: Dict[str, Any], quiet: bool = True) -> None:
        """
        Modifies Makefile variables in-place using sed.
        """
        for key, value in flags.items():
            try:
                subprocess.run(
                    ["sed", "-i", f"s/^{key}=.*/{key}={value}/", "Makefile"],
                    cwd=self.darknet_dir,
                    check=True,
                    stdout=subprocess.DEVNULL if quiet else None,
                    stderr=subprocess.DEVNULL if quiet else None,
                )
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to patch Makefile key: {key}")
                raise e

        self.logger.debug(f"Patched Makefile flags: {flags}")

    def run(self) -> None:
        """
        Executes the full build workflow: configure -> rebuild -> verify.
        """
        self.configure()
        self.rebuild()
        self.verify_darknet()