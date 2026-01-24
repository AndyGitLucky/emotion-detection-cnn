import subprocess
import zipfile
from pathlib import Path

import tensorflow as tf

from src.config import (
    TRAIN_DIR, TEST_DIR, DATASET_ROOT,
    IMG_HEIGHT, IMG_WIDTH,
    BATCH_SIZE
)

# Kaggle dataset slug
KAGGLE_DATASET = "ananthu017/emotion-detection-fer"


def _dataset_exists() -> bool:
    """
    Check whether the dataset exists and is non-empty.
    """
    return (
        TRAIN_DIR.exists()
        and TEST_DIR.exists()
        and any(TRAIN_DIR.iterdir())
        and any(TEST_DIR.iterdir())
    )


def _download_from_kaggle() -> None:
    """
    Download and extract the dataset from Kaggle into DATASET_ROOT.
    Fix directory structure if Kaggle nests it.
    """
    print("Dataset not found. Downloading from Kaggle...")

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(DATASET_ROOT),
        "--force"
    ]

    subprocess.run(cmd, check=True)

    # Kaggle names the zip after the dataset slug
    downloaded_zip = next(DATASET_ROOT.glob("*.zip"))

    with zipfile.ZipFile(downloaded_zip, "r") as zip_ref:
        zip_ref.extractall(DATASET_ROOT)

    downloaded_zip.unlink()

    print("Kaggle dataset downloaded and extracted.")

    # ---- Fix nested Kaggle directory structure ----
    # Expected final structure:
    # archive/
    # ├── train/
    # └── test/

    nested_root = DATASET_ROOT / "emotion-detection-fer"
    if nested_root.exists():
        print("Fixing nested Kaggle directory structure...")

        for subdir in ["train", "test"]:
            src = nested_root / subdir
            dst = DATASET_ROOT / subdir

            if src.exists():
                if dst.exists():
                    # Merge contents if destination already exists
                    for item in src.iterdir():
                        item.rename(dst / item.name)
                else:
                    src.rename(dst)

        # Remove empty nested root if possible
        try:
            nested_root.rmdir()
        except OSError:
            pass

    print("Dataset directory structure is ready.")


def load_datasets():
    """
    Load train, validation and test datasets.
    If the dataset is missing, download it automatically from Kaggle.
    """
    if not _dataset_exists():
        _download_from_kaggle()

    """ 
    we have less train data and more test data (0.25), so we split the test data files into test and validate 
    to validate the training with the 'training' split and Test the model at the very end with this  
    small - never seen before - 'validation' split 
    """

    train_data = tf.keras.utils.image_dataset_from_directory(
        directory=str(TRAIN_DIR),
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        seed=42
    )

    val_data = tf.keras.utils.image_dataset_from_directory(
        directory=str(TEST_DIR),
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        validation_split=0.2,
        subset="training",
        seed=42
    )

    test_data = tf.keras.utils.image_dataset_from_directory(
        directory=str(TEST_DIR),
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        validation_split=0.2,
        subset="validation",
        seed=42
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
    val_data   = val_data.cache().prefetch(buffer_size=AUTOTUNE)
    test_data  = test_data.cache().prefetch(buffer_size=AUTOTUNE)

    return train_data, val_data, test_data
