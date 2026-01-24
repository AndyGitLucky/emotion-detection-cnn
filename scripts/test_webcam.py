import cv2
from pathlib import Path

cascade_path = Path(__file__).resolve().parents[1] / "assets" / "haarcascade_frontalface_default.xml"

print("Cascade path:", cascade_path)
print("Exists:", cascade_path.exists())

face_cascade = cv2.CascadeClassifier(str(cascade_path))
print("Cascade empty:", face_cascade.empty())

cap = cv2.VideoCapture(1)
print("Camera opened:", cap.isOpened())

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(48, 48)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
