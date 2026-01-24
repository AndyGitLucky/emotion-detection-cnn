import cv2
import numpy as np

class RealtimeEmotionDetector:
    def __init__(self, model, class_labels, img_size, cascade_path):
        self.model = model
        self.class_labels = class_labels
        self.img_size = img_size
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def preprocess_face(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.img_size)
        resized = resized.astype("float32") / 255.0

        if resized.ndim == 2:
            resized = resized[..., np.newaxis]

        input_img = np.expand_dims(resized, axis=0)
        return input_img

    def run(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(48, 48)
            )

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                input_img = self.preprocess_face(face_img)

                preds = self.model.predict(input_img, verbose=0)
                idx = int(np.argmax(preds))
                label = self.class_labels[idx]
                prob = float(np.max(preds)) * 100

                text = f"{label} ({prob:.1f}%)"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2
                )

            cv2.imshow("Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
