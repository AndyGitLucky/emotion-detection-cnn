import csv
import subprocess
import tarfile
import urllib.request

from pathlib import Path

import numpy as np
from PIL import Image


# ==================================================
# Constants
# ==================================================

FER2013_COMPETITION = (
    "challenges-in-representation-learning-facial-expression-recognition-challenge"
)
FERPLUS_CSV_URL = (
    "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
)


# ==================================================
# Helpers (internal)
# ==================================================

def _images_exist(image_dir: Path) -> bool:
    return image_dir.exists() and any(image_dir.glob("fer*.png"))


def _download_competition(target_dir: Path) -> None:
    """
    Download and extract FER2013 Kaggle competition data.
    """
    print("Downloading FER2013 competition data from Kaggle...")

    target_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            FER2013_COMPETITION,
            "-p",
            str(target_dir),
            "--force",
        ],
        check=True,
    )

    # unzip competition zip(s)
    zip_files = list(target_dir.glob("*.zip"))
    if not zip_files:
        raise RuntimeError("Competition ZIP not found after download")

    for zip_path in zip_files:
        subprocess.run(
            ["unzip", "-o", zip_path.name],
            cwd=target_dir,
            check=True,
        )
        zip_path.unlink()

    # extract fer2013.tar.gz
    tar_files = list(target_dir.glob("*.tar.gz"))
    if not tar_files:
        raise RuntimeError("fer2013.tar.gz not found in competition archive")

    for tar_path in tar_files:
        with tarfile.open(tar_path) as tar:
            tar.extractall(target_dir)
        tar_path.unlink()

    print("FER2013 competition data ready.")


def _download_ferplus_csv(target_path: Path) -> None:
    print("Downloading FER+ labels (fer2013new.csv)...")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(FERPLUS_CSV_URL, target_path)

    if not target_path.exists():
        raise RuntimeError("FER+ CSV download failed")

    print("FER+ labels downloaded successfully.")


def _build_images_from_csv(csv_path: Path, image_dir: Path) -> None:
    """
    Build FER2013 PNG images from icml_face_data.csv.
    """
    print("Building FER2013 images from CSV (one-time step)...")

    image_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)

        for idx, row in enumerate(reader):
            pixels = row.get("pixels")
            if pixels is None:
                continue

            pixel_values = np.fromstring(pixels, sep=" ", dtype=np.uint8)
            if pixel_values.size != 48 * 48:
                continue

            img = pixel_values.reshape((48, 48))
            img = Image.fromarray(img, mode="L")

            img_path = image_dir / f"fer{idx:07d}.png"
            img.save(img_path)

    print("FER2013 images built successfully.")


# ==================================================
# Public API
# ==================================================

def ensure_fer2013_available(config) -> None:
    """
    Ensure original FER2013 images exist.

    This is the ONLY valid base for FER+.
    """
    dataset_root = config.dataset.root
    comp_dir = dataset_root / "fer2013_competition"
    csv_path = comp_dir / "icml_face_data.csv"
    image_dir = comp_dir / "images"

    if _images_exist(image_dir):
        return

    _download_competition(comp_dir)

    if not csv_path.exists():
        raise RuntimeError(
            "icml_face_data.csv not found after competition extraction"
        )

    _build_images_from_csv(csv_path, image_dir)


def ensure_ferplus_available(config) -> None:
    """
    Ensure FER+ prerequisites are met:
    - original FER2013 images exist
    - FER+ label CSV exists (auto-download if missing)
    """
    ensure_fer2013_available(config)

    ferplus_csv = config.dataset.ferplus.label_csv

    if not ferplus_csv.exists():
        _download_ferplus_csv(ferplus_csv)
