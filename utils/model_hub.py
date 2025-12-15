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
    save_path.parent.mkdir(parents=True, exist_ok=True)

    command = ["curl", "-L", "-f", "-o", str(save_path), url]
    subprocess.run(command, check=True)


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

    logger.info("[WEIGHT] Local YOLO weight missing → downloading...")

    weights_cfg = cfg.get("weights", {})
    sha_cfg = cfg.get("sha256", {})

    if model_name not in weights_cfg:
        raise FileNotFoundError(
            f"No fine-tuned weight URL found for model '{model_name}'"
        )

    url = weights_cfg[model_name]
    sha = sha_cfg.get(model_name)

    download_file(url, saved_model_path)

    if sha:
        if verify_sha256(saved_model_path, sha):
            logger.info("[WEIGHT] SHA256 verified.")
        else:
            raise RuntimeError(
                f"SHA256 mismatch for downloaded YOLO weight: {saved_model_path}"
            )
    else:
        logger.warning("[WEIGHT] No SHA256 provided — skipping integrity check.")
