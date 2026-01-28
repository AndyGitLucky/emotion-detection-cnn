import csv
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


# ==================================================
# Paths
# ==================================================

ARCHIVE_DIR = Path("archive")

FER2013_COMP_DIR = ARCHIVE_DIR / "fer2013_competition"
FER2013_CSV = FER2013_COMP_DIR / "icml_face_data.csv"
FER2013_IMG_DIR = FER2013_COMP_DIR / "images"

FERPLUS_DIR = ARCHIVE_DIR / "ferplus"
FERPLUS_LABEL_CSV = FERPLUS_DIR / "fer2013new.csv"


# ==================================================
# Kaggle Competition
# ==================================================

FER2013_COMPETITION = (
    "challenges-in-representation-learning-facial-expression-recognition-challenge"
)


# ==================================================
# Helpers
# ==================================================

def _images_exist() -> bool:
    return FER2013_IMG_DIR.exists() and any(FER2013_IMG_DIR.glob("fer*.png"))


def _download_competition() -> None:
    print("Downloading FER2013 competition data from Kaggle...")

    FER2013_COMP_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            FER2013_COMPETITION,
            "-p",
            str(FER2013_COMP_DIR),
            "--force",
        ],
        check=True,
    )

    # 1) unzip competition zip
    zip_files = list(FER2013_COMP_DIR.glob("*.zip"))
    if not zip_files:
        raise RuntimeError("Competition ZIP not found after download")

    for zip_path in zip_files:
        subprocess.run(
            ["unzip", "-o", zip_path.name],
            cwd=FER2013_COMP_DIR,
            check=True,
        )
        zip_path.unlink()

    # 2) extract fer2013.tar.gz
    tar_files = list(FER2013_COMP_DIR.glob("*.tar.gz"))
    if not tar_files:
        raise RuntimeError("fer2013.tar.gz not found in competition archive")

    for tar_path in tar_files:
        with tarfile.open(tar_path) as tar:
            tar.extractall(FER2013_COMP_DIR)
        tar_path.unlink()

    if not FER2013_CSV.exists():
        raise RuntimeError("icml_face_data.csv not found after extraction")

    print("FER2013 competition data ready.")


def _build_images_from_csv() -> None:
    print("Building FER2013 images from icml_face_data.csv (one-time step)...")

    FER2013_IMG_DIR.mkdir(parents=True, exist_ok=True)

    with open(FER2013_CSV, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)

        for idx, row in enumerate(reader):
            pixels = row["pixels"]

            pixel_values = np.fromstring(pixels, sep=" ", dtype=np.uint8)
            if pixel_values.size != 48 * 48:
                continue

            img = pixel_values.reshape((48, 48))
            img = Image.fromarray(img, mode="L")

            img_path = FER2013_IMG_DIR / f"fer{idx:07d}.png"
            img.save(img_path)

    print("FER2013 images built successfully.")


# ==================================================
# Public API
# ==================================================

def ensure_fer2013_available() -> None:
    """
    Ensure original FER2013 images exist, built from icml_face_data.csv.

    This is the ONLY valid base for FER+.
    """
    if _images_exist():
        return

    _download_competition()
    _build_images_from_csv()


def ensure_ferplus_available() -> None:
    """
    Ensure FER+ prerequisites are met:
    - original FER2013 images exist
    - FER+ label CSV exists (downloaded separately)
    """
    ensure_fer2013_available()

    if not FERPLUS_LABEL_CSV.exists():
        raise RuntimeError(
            "FER+ labels missing: expected fer2013new.csv at "
            f"{FERPLUS_LABEL_CSV}"
        )
