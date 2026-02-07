import warnings
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================= CONFIGURATION =================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

ROOT = Path("data/experiments/gac_eval")
ORIGINAL_DIR = ROOT / "original"
GEN_DIR = ROOT / "gac"
LABEL_DIR = ROOT / "label"
SAVE_ORIG_CROP_DIR = ROOT / "baseline"
SAVE_GEN_CROP_DIR = ROOT / "anchors" / "gac"


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def _find_image_by_stem(folder: Path, stem: str) -> Optional[Path]:
    if not folder.exists():
        return None

    cands = []
    for p in folder.glob(f"{stem}.*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            cands.append(p)

    return cands[0] if cands else None


def get_crop_coords(yolo_line: str, img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int]]:
    """
    YOLO label line supports:
      - bbox: cls xc yc w h
      - polygon: cls x1 y1 x2 y2 ...
    coords are normalized [0,1]
    """
    parts = yolo_line.strip().split()
    if len(parts) < 5:
        return None

    try:
        nums = list(map(float, parts))
    except ValueError:
        return None

    coords = nums[1:]  # skip class_id

    # bbox: xc, yc, w, h
    if len(coords) == 4:
        xc, yc, w, h = coords
        box_w = w * img_w
        box_h = h * img_h

        left = (xc * img_w) - (box_w / 2)
        top = (yc * img_h) - (box_h / 2)
        right = left + box_w
        bottom = top + box_h
    else:
        # polygon
        xs = coords[0::2]
        ys = coords[1::2]
        if len(xs) == 0 or len(ys) == 0:
            return None

        left = min(xs) * img_w
        right = max(xs) * img_w
        top = min(ys) * img_h
        bottom = max(ys) * img_h

    left = max(0, int(left))
    top = max(0, int(top))
    right = min(img_w, int(right))
    bottom = min(img_h, int(bottom))

    if right <= left or bottom <= top:
        return None

    return (left, top, right, bottom)


def crop_and_save(
    img_path: Path,
    coords: Tuple[int, int, int, int],
    save_dir: Path,
    filename_stem: str,
    obj_idx: int,
) -> bool:
    if obj_idx != 0:
        return True

    save_dir.mkdir(parents=True, exist_ok=True)

    out_ext = img_path.suffix.lower()  # e.g., ".jpg", ".png"
    save_path = save_dir / f"{filename_stem}{out_ext}"

    try:
        with Image.open(img_path) as img:
            crop_img = img.crop(coords)

            if out_ext in {".jpg", ".jpeg"} and crop_img.mode in ("RGBA", "LA", "P"):
                crop_img = crop_img.convert("RGB")

            crop_img.save(save_path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------
# Main Crop Process
# ---------------------------------------------------------
def run_crop_process() -> None:
    print("Starting Crop Process...")

    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")

    label_files = sorted(LABEL_DIR.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(
            f"No .txt label files found in: {LABEL_DIR}\n"
        )

    SAVE_ORIG_CROP_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_GEN_CROP_DIR.mkdir(parents=True, exist_ok=True)

    for label_path in tqdm(label_files, desc="Cropping"):
        stem = label_path.stem

        orig_img_path = _find_image_by_stem(ORIGINAL_DIR, stem)
        gen_img_path = _find_image_by_stem(GEN_DIR, stem)

        if orig_img_path is None:
            continue

        try:
            with Image.open(orig_img_path) as img0:
                W, H = img0.size
        except Exception:
            continue

        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            lines = label_path.read_text(encoding="cp949").splitlines()

        for obj_idx, line in enumerate(lines):
            if not line.strip():
                continue

            coords = get_crop_coords(line, W, H)
            if coords is None:
                continue

            crop_and_save(orig_img_path, coords, SAVE_ORIG_CROP_DIR, stem, obj_idx)

            if gen_img_path is not None:
                crop_and_save(gen_img_path, coords, SAVE_GEN_CROP_DIR, stem, obj_idx)

    print("\nAll jobs finished.")
    print(f"Original Crops saved to: {SAVE_ORIG_CROP_DIR}")
    print(f"Generation Crops saved to: {SAVE_GEN_CROP_DIR}")


def main() -> None:
    run_crop_process()


if __name__ == "__main__":
    main()
