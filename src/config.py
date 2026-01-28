from pathlib import Path
import os
from types import SimpleNamespace

# -------------------------------------------------
# Project paths
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = Path(
    os.environ.get("EMOTION_DATASET_ROOT", PROJECT_ROOT / "archive")
)

# -------------------------------------------------
# Dataset paths
# -------------------------------------------------

TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR  = DATASET_ROOT / "test"

# FER+ paths
FERPLUS_LABEL_CSV = DATASET_ROOT / "ferplus" / "fer2013new.csv"
FERPLUS_IMAGE_DIR = DATASET_ROOT / "ferplus" / "images"

# -------------------------------------------------
# Assets / Runtime
# -------------------------------------------------

CASCADE_PATH = PROJECT_ROOT / "assets" / "haarcascade_frontalface_default.xml"
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "emotion_model"

FORCE_RETRAIN = False

FREEZE_BEST_MODEL = True
PROMOTION_METRIC = "weighted_f1"   # or: "accuracy"

# -------------------------------------------------
# Training hyperparameters (legacy flat access)
# -------------------------------------------------

IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 256
NUM_CLASSES = 7
LEARNING_RATE = 0.0005
EARLY_STOP_PATIENCE = 20
EPOCHS = 50

# -------------------------------------------------
# Dataset selection (legacy)
# -------------------------------------------------

DATASET = "ferplus"  
# Options:
# - "fer2013"
# - "ferplus"

# -------------------------------------------------
# FER+ specific
# -------------------------------------------------

FERPLUS_MIN_AGREEMENT = 0.6
FERPLUS_DROP_CONTEMPT = True

# -------------------------------------------------
# Labels
# -------------------------------------------------

CLASS_LABELS = [
    'angry',
    'disgusted',
    'fearful',
    'happy',
    'neutral',
    'sad',
    'surprised'
]

# =================================================
# NEW STRUCTURED CONFIG (non-breaking)
# =================================================

dataset = SimpleNamespace(
    # identity
    name=DATASET,

    # data geometry
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    num_classes=NUM_CLASSES,

    # batching
    batch_size=BATCH_SIZE,

    # splits (can be tuned later)
    train_split=0.8,
    val_split=0.1,

    # reproducibility
    seed=42,

    # paths
    root=DATASET_ROOT,
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,

    # FER+ specific
    ferplus=SimpleNamespace(
        label_csv=FERPLUS_LABEL_CSV,
        image_dir=FERPLUS_IMAGE_DIR,
        min_agreement=FERPLUS_MIN_AGREEMENT,
        drop_contempt=FERPLUS_DROP_CONTEMPT,
    )
)

training = SimpleNamespace(
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    early_stop_patience=EARLY_STOP_PATIENCE,
)
