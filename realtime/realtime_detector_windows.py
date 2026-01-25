import cv2
import numpy as np
from pathlib import Path


class EmotionDetector:
    def __init__(
        self,
        model_loader,
        model_path,
        cascade_path,
        class_labels,
        img_height,
        img_width,
        preprocess_fn,
        model_sync_fn=None,
    ):
        # Injected dependencies
        self.model_loader = model_loader
        self.model_path = Path(model_path)
        self.cascade_path = Path(cascade_path)
        self.class_labels = class_labels
        self.img_height = img_height
        self.img_width = img_width
        self.preprocess_fn = preprocess_fn
        self.model_sync_fn = model_sync_fn

        # Runtime state
        self.model = None
        self.face_cascade = None

    # -------------------------
    # Loading
    # -------------------------
    def load_model(self):
        if self.model_sync_fn is not None:
            self.model_sync_fn()

        print(f"→ Loading model from: {self.model_path}")
        self.model = self.model_loader(self.model_path)
        print("✔ Model loaded")

    def load_face_detector(self):
        if not self.cascade_path.exists():
            raise FileNotFoundError(f"Haar cascade not found: {self.cascade_path}")

        self.face_cascade = cv2.CascadeClassifier(str(self.cascade_path))
        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade: {self.cascade_path}")

    # -------------------------
    # Inference
    # -------------------------
    def predict(self, face_bgr: np.ndarray):
        input_img = self.preprocess_fn(face_bgr)
        preds = self.model.predict(input_img, verbose=0)[0]

        class_idx = int(np.argmax(preds))
        label = self.class_labels[class_idx]
        confidence = float(preds[class_idx])

        return label, confidence

    # -------------------------
    # Rendering
    # -------------------------
    def draw_overlay(self, frame, x, y, w, h, label, confidence):
        text = f"{label} ({confidence:.2f})"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    # -------------------------
    # Main loop
    # -------------------------
    def run(self, camera_index=0):
        self.load_model()
        self.load_face_detector()

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("❌ Could not open webcam")

        print("✔ Webcam opened. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(48, 48),
            )

            for (x, y, w, h) in faces:
                face_img = frame[y : y + h, x : x + w]
                label, confidence = self.predict(face_img)
                self.draw_overlay(frame, x, y, w, h, label, confidence)

            cv2.imshow("Emotion Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
