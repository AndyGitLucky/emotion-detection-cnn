from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"

CASCADE_PATH = PROJECT_ROOT / "assets" / "haarcascade_frontalface_default.xml"

# Define constants
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64
NUM_CLASSES = 7
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 20
EPOCHS = 50

# Define class labels
CLASS_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']