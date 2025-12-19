#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for removing visual annotations using the Gemini API.

This script interacts with a generative model to clean images based on
configured prompts and handles the I/O for processed datasets.
"""

import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from google import genai
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger, setup_logging


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Initializes the Gemini client with API key resolution.

    Resolution Priority:
    1. Direct argument input.
    2. Environment variable reference (format: "${VAR_NAME}").
    3. Default system environment variable "GEMINI_API_KEY".

    Args:
        api_key (Optional[str]): The API key or a reference string.

    Returns:
        genai.Client: Authenticated Gemini client instance.

    Raises:
        EnvironmentError: If the API key is missing.
        RuntimeError: If client initialization fails.
    """
    if api_key and isinstance(api_key, str) and api_key.startswith("${"):
        env_var_name = api_key.strip("${}")
        key = os.getenv(env_var_name)
    elif api_key:
        key = api_key
    else:
        key = os.getenv("GEMINI_API_KEY")

    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Ensure it is set in .env or config.yaml"
        )

    try:
        return genai.Client(api_key=key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini client: {e}")


class CleanAnnotation:
    """
    Controller for removing visual annotations via the Gemini API.
    """

    def __init__(self, config: Dict) -> None:
        """
        Args:
            config (Dict): Loaded configuration dictionary.
        """
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("annotation_cleaner.CleanAnnotation")

        # Configuration setup
        self.cfg = config
        global_main_cfg = self.cfg.get("main", {})
        cleaner_cfg = self.cfg.get("annotation_cleaner", {})
        self.main_cfg = cleaner_cfg.get("main", {})
        self.clean_cfg = cleaner_cfg.get("annotation_clean", {})

        # Path setup
        self.input_root = Path(
            self.clean_cfg.get(
                "input_dir",
                self.main_cfg.get(
                    "input_dir",
                    "data/annotation_cleaner/only_annotation_image_padded",
                ),
            )
        ).resolve()

        self.output_root = Path(
            self.clean_cfg.get(
                "output_dir",
                "data/annotation_cleaner/generated_image_padded",
            )
        ).resolve()

        # Execution parameters
        self.test_mode: bool = bool(self.clean_cfg.get("test_mode", False))
        self.test_limit: int = int(self.clean_cfg.get("test_limit", 3))
        self.categories: List[str] = global_main_cfg.get(
            "categories", ["repair", "replace"]
        )

        # Model initialization
        self.model: str = self.clean_cfg.get(
            "model", "gemini-2.5-flash-image-preview"
        )
        self.prompt: Optional[str] = self.clean_cfg.get("prompt")
        api_key = self.clean_cfg.get("api_key")
        self.client = get_gemini_client(api_key)

        self.logger.info("Initialized CleanAnnotation")
        self.logger.info(f" - Input dir      : {self.input_root}")
        self.logger.info(f" - Output dir     : {self.output_root}")
        self.logger.info(f" - Model          : {self.model}")
        self.logger.info(f" - Test mode      : {self.test_mode}")
        if self.test_mode:
            self.logger.info(
                f" - Test limit : {self.test_limit} images per category"
            )

    def _generate_clean_image(self, image_path: Path, output_path: Path) -> bool:
        """
        Generates a clean image using the model and saves it.

        Args:
            image_path (Path): Path to the source image with annotations.
            output_path (Path): Destination path for the cleaned image.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            image = Image.open(image_path)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[self.prompt, image],
            )

            for part in response.candidates[0].content.parts:
                if getattr(part, "inline_data", None):
                    gen_img = Image.open(BytesIO(part.inline_data.data))
                    gen_img.save(output_path)
                    self.logger.info(f"Saved cleaned image: {output_path.name}")
                    return True
                elif getattr(part, "text", None):
                    self.logger.warning(
                        f"Text response for {image_path.name}: {part.text}"
                    )
                    return False

        except Exception as e:
            self.logger.error(f"Error processing {image_path.name}: {e}")
            return False

        return False

    def run(self) -> None:
        """
        Executes the cleaning pipeline for all configured categories.
        Iterates through categories and processes images sequentially.
        """
        if not self.input_root.exists():
            raise FileNotFoundError(f"Input folder not found: {self.input_root}")

        self.output_root.mkdir(parents=True, exist_ok=True)

        for category in self.categories:
            processed_count = 0

            in_dir = self.input_root / category
            out_dir = self.output_root / category
            out_dir.mkdir(parents=True, exist_ok=True)

            if not in_dir.exists():
                self.logger.warning(f"Missing folder: {in_dir}")
                continue

            image_files = sorted(
                f
                for f in os.listdir(in_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )

            for filename in image_files:
                input_path = in_dir / filename
                output_path = out_dir / filename

                # Skip if already exists
                if output_path.exists():
                    if self.test_mode:
                        self.logger.info(
                            f"[TEST MODE] Skipping existing file: {filename}"
                        )
                    else:
                        self.logger.info(f"Skipping existing file: {filename}")
                    continue

                success = self._generate_clean_image(
                    input_path, output_path
                )
                processed_count += int(success)

                # Handle test mode limit
                if self.test_mode and processed_count >= self.test_limit:
                    self.logger.info(
                        f"[TEST MODE] Category '{category}' reached limit "
                        f"({self.test_limit}). Moving to next category."
                    )
                    break

            self.logger.info(
                f"Category '{category}' completed "
                f"({processed_count} images processed)."
            )

        self.logger.info("Annotation cleaning process finished.")