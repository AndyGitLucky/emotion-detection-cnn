import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from src.config import CLASS_LABELS


def evaluate_model(model, test_data):
    test_loss, test_accuracy = model.evaluate(test_data)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    y_true = []
    y_pred = []

    for images, labels in test_data:
        preds = model.predict(images)
        y_true.extend(labels.numpy().argmax(axis=1))
        y_pred.extend(preds.argmax(axis=1))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    print(
        "Classification Report:\n",
        classification_report(y_true, y_pred, target_names=CLASS_LABELS)
    )

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print("Macro F1:", macro_f1)
    print("Weighted F1:", weighted_f1)

    metrics = {
        "loss": float(test_loss),
        "accuracy": float(test_accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }

    return metrics
