from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(
    os.environ.get("EMOTION_DATASET_ROOT", PROJECT_ROOT / "archive")
)

TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR  = DATASET_ROOT / "test"

CASCADE_PATH = PROJECT_ROOT / "assets" / "haarcascade_frontalface_default.xml"
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "emotion_model"

# Define constants
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 256
NUM_CLASSES = 7
LEARNING_RATE = 0.0005
EARLY_STOP_PATIENCE = 20
EPOCHS = 50

# Define class labels
CLASS_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']