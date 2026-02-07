# utils/model_hub.py

import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Optional


def sha256_of(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, save_path: Path):
    """
    Download file using curl, with specific handling for Dropbox links.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if "dropbox.com" in url:
        if "dl=0" in url:
            url = url.replace("dl=0", "dl=1")
        elif "dl=1" not in url:
            if "?" in url:
                url += "&dl=1"
            else:
                url += "?dl=1"

    command = ["curl", "-L", "-f", "-o", str(save_path), url]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download file from {url}. Ensure the link is valid.") from e


def verify_sha256(file_path: Path, expected: str) -> bool:
    actual = sha256_of(file_path)
    if expected.startswith("sha256:"):
        expected = expected.split("sha256:")[1]
    return actual.lower() == expected.lower()


def download_fine_tuned_weights(
    *,
    cfg: Dict,
    model_name: str,
    saved_model_path: Path,
    logger,
):
    """
    Download fine-tuned YOLO weights if not present.

    This function is intentionally stateless and reusable across pipelines.
    """

    if saved_model_path.exists():
        logger.info(f"[WEIGHT] Using existing weight: {saved_model_path}")
        return

    logger.info(f"[WEIGHT] Local YOLO weight missing. Downloading '{model_name}' from external storage...")

    weights_cfg = cfg.get("weights", {})
    sha_cfg = cfg.get("sha256", {})

    if model_name not in weights_cfg:
        raise FileNotFoundError(
            f"No fine-tuned weight URL found for model '{model_name}' in config."
        )

    url = weights_cfg[model_name]
    sha = sha_cfg.get(model_name)

    download_file(url, saved_model_path)

    if sha:
        logger.info(f"[WEIGHT] Verifying SHA256 integrity...")
        if verify_sha256(saved_model_path, sha):
            logger.info("[WEIGHT] SHA256 verified successfully.")
        else:
            saved_model_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256 mismatch for downloaded YOLO weight: {saved_model_path}. \n"
                f"The file might be corrupted during download. Please try again."
            )
    else:
        logger.warning("[WEIGHT] No SHA256 provided — skipping integrity check.")