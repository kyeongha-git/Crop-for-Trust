#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_annotation.py
-------------------
This module automatically removes drawn annotations or visual marks
from car windshield images using the Gemini API.

Purpose:
    - Clean annotated or marked images to create unbiased training data.
    - Generate clear images suitable for machine learning or visual analysis.
    - Support automated processing of multiple image categories.

Usage Example:
    >>> cleaner = CleanAnnotation(
    ...     input_dir="data/original",
    ...     output_dir="data/cleaned",
    ...     model="gemini-1.5-pro",
    ...     prompt="Remove any pen marks or drawn lines from this windshield image."
    ... )
    >>> cleaner.run()
"""

import os
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from google import genai
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger, setup_logging


# ============================================================
# Gemini Client Initialization
# ============================================================
def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Initialize and return a Gemini API client.

    The client is used to send image and text prompts to the Gemini model
    for image cleaning tasks.

    Args:
        api_key (Optional[str]): Gemini API key. If not provided,
                                 retrieves from environment variable `GEMINI_API_KEY`.

    Returns:
        genai.Client: Authenticated Gemini client.

    Raises:
        EnvironmentError: If the API key is missing.
        RuntimeError: If client creation fails.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
    try:
        return genai.Client(api_key=key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini client: {e}")


# ============================================================
# CleanAnnotation Class
# ============================================================
class CleanAnnotation:
    """
    A class that removes visual annotations or guide marks from images using the Gemini API.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model: str,
        prompt: str,
        categories: Optional[List[str]] = None,
        test_mode: bool = False,
        test_limit: int = 3,
        client: Optional[genai.Client] = None,
    ):
        """
        Initialize the cleaning module with paths, model configuration, and logging.

        Args:
            input_dir (str): Folder containing input images grouped by category.
            output_dir (str): Folder where cleaned images will be saved.
            model (str): Gemini model to use for image cleaning.
            prompt (str): Instruction for how to remove annotations.
            categories (Optional[List[str]]): List of category subfolders to process.
            test_mode (bool): If True, limits processing for debugging or preview.
            test_limit (int): Number of images to process in test mode.
            client (Optional[genai.Client]): Existing Gemini client, if already initialized.
        """
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("CleanAnnotation")

        self.input_root = Path(input_dir)
        self.output_root = Path(output_dir)
        self.categories = categories or ["repair", "replace"]
        self.model = model
        self.prompt = prompt
        self.test_mode = test_mode
        self.test_limit = test_limit

        self.client = client or get_gemini_client()

        self.logger.info(f"Input directory: {self.input_root}")
        self.logger.info(f"Output directory: {self.output_root}")
        self.logger.info(f"Model: {self.model}")

    # ============================================================
    # Internal Utility
    # ============================================================
    def _generate_clean_image(self, image_path: Path, output_path: Path) -> bool:
        """
        Send an image to the Gemini model and save the cleaned output.
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

    # ============================================================
    # Public API
    # ============================================================
    def run(self):
        """
        Clean all annotated images under each category folder.

        The method scans each category directory (e.g., 'repair', 'replace'),
        sends images to the Gemini model to remove visual marks,
        and saves the cleaned images to the output directory.
        """
        if not self.input_root.exists():
            raise FileNotFoundError(f"Input folder not found: {self.input_root}")

        self.output_root.mkdir(parents=True, exist_ok=True)
        processed_count = 0

        for category in self.categories:
            in_dir = self.input_root / category
            out_dir = self.output_root / category
            out_dir.mkdir(parents=True, exist_ok=True)

            if not in_dir.exists():
                self.logger.warning(f"Missing folder: {in_dir}")
                continue

            image_files = [
                f
                for f in os.listdir(in_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            for filename in image_files:
                input_path = in_dir / filename
                output_path = out_dir / filename

                if output_path.exists():
                    self.logger.info(f"Skipping existing file: {filename}")
                    continue

                success = self._generate_clean_image(input_path, output_path)
                processed_count += int(success)

                if (
                    self.test_mode
                    and self.test_limit
                    and processed_count >= self.test_limit
                ):
                    self.logger.info(
                        f"Test mode limit reached ({self.test_limit}). Stopping early."
                    )
                    return

        self.logger.info(
            f"🎉 Cleaning complete. Total processed: {processed_count} images."
        )
