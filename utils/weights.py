import hashlib
import subprocess
from pathlib import Path
import os


def sha256_of(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, save_path: Path):
    """Download file using system curl (more robust for Docker/GitHub)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] {url}")
    print(f" → Saving to {save_path}")

    command = ["curl", "-L", "-f", "-o", str(save_path), url]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ Download failed (curl error): {url}\n{e}")


def verify_sha256(file_path: Path, expected: str) -> bool:
    """Verify the SHA256 hash of the file."""
    actual = sha256_of(file_path)

    # Remove "sha256:" prefix if provided
    if expected.startswith("sha256:"):
        expected = expected.split("sha256:")[1]

    return actual.lower() == expected.lower()