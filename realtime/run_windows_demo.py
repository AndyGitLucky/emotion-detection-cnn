import sys
import os
import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# --- Make repo root importable ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Paths & infrastructure config ---
WSL_MODEL_PATH = "/home/andreas/emotion_detection_model_V6/checkpoints/emotion_model"
WSL_CONFIG_PATH = "/home/andreas/emotion_detection_model_V6/src/config.py"

WIN_PROJECT_ROOT = Path(r"H:\alfa-training\Deep Learning\_Abschlussprojekt\emotion_detection_model_V6")
LOCAL_MODEL_PATH = WIN_PROJECT_ROOT / "checkpoints" / "emotion_model"
LOCAL_CONFIG_PATH = WIN_PROJECT_ROOT / "shared" / "config.py"
CASCADE_PATH = WIN_PROJECT_ROOT / "assets" / "haarcascade_frontalface_default.xml"


def win_path_to_wsl_mount(win_path: Path) -> str:
    drive = win_path.drive[0].lower()
    rest = str(win_path)[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def copy_model_and_config_from_wsl():
    print("→ Syncing model + config from WSL...")

    if LOCAL_MODEL_PATH.exists():
        shutil.rmtree(LOCAL_MODEL_PATH)

    LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    wsl_target_model_path = win_path_to_wsl_mount(LOCAL_MODEL_PATH)
    wsl_target_config_path = win_path_to_wsl_mount(LOCAL_CONFIG_PATH)

    cmd_str = (
        f'mkdir -p "{wsl_target_model_path}" && '
        f'rm -rf "{wsl_target_model_path}" && '
        f'cp -r "{WSL_MODEL_PATH}" "{wsl_target_model_path}" && '
        f'cp "{WSL_CONFIG_PATH}" "{wsl_target_config_path}"'
    )

    cmd = ["wsl", "bash", "-lc", cmd_str]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError("WSL model/config sync failed")

    print("✔ Model + config synced successfully")




from realtime.realtime_detector_windows import EmotionDetector


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    resized = resized.astype("float32")
    resized = np.expand_dims(resized, axis=-1)
    resized = np.expand_dims(resized, axis=0)
    return resized


if __name__ == "__main__":
    copy_model_and_config_from_wsl()
    
    # --- Import synced config (Single Source of Truth) ---
    from shared.config import (
        IMG_HEIGHT,
        IMG_WIDTH,
        CLASS_LABELS
    )

    detector = EmotionDetector(
        model_loader=tf.keras.models.load_model,
        model_path=LOCAL_MODEL_PATH,
        cascade_path=CASCADE_PATH,
        class_labels=CLASS_LABELS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        preprocess_fn=preprocess_face,
        model_sync_fn=None,  # already synced above
    )

    detector.run()
