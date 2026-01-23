import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


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

    print("Classification Report:\n",
          classification_report(y_true, y_pred))

    return test_loss, test_accuracy
