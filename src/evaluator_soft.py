# evaluator_soft.py
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# --------------------------------------------------
# Soft metrics (FER+ compatible)
# --------------------------------------------------

def mean_kl(y_true, y_pred, eps=1e-8):
    """
    Mean KL-Divergence between soft ground-truth and predictions.
    Primary FER+ evaluation metric (lower is better).
    """
    return float(np.mean(
        np.sum(
            y_true * np.log((y_true + eps) / (y_pred + eps)),
            axis=1
        )
    ))


def soft_f1(y_true, y_pred, eps=1e-8):
    """
    Soft-F1 score averaged over classes.
    Interpretable but secondary to KL.
    """
    tp = np.sum(y_true * y_pred, axis=0)
    fp = np.sum((1 - y_true) * y_pred, axis=0)
    fn = np.sum(y_true * (1 - y_pred), axis=0)

    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


# --------------------------------------------------
# Evaluation entry point
# --------------------------------------------------

def evaluate_model_soft(model, val_data, config, confidence_threshold=None):
    """
    FER+ soft-label evaluation.

    Returns:
        metrics dict with:
        - mean_kl  (primary)
        - soft_f1  (secondary)

    Prints:
        - soft metrics
        - optional hard classification report (reporting only)
    """

    y_true_soft = []
    y_pred_soft = []

    for batch in val_data:
        images = batch[0]
        labels = batch[1]

        preds = model.predict(images, verbose=0)

        y_true_soft.append(labels.numpy())
        y_pred_soft.append(preds)

    y_true_soft = np.concatenate(y_true_soft)
    y_pred_soft = np.concatenate(y_pred_soft)

    # -------------------------
    # Soft evaluation (core)
    # -------------------------
    kl = mean_kl(y_true_soft, y_pred_soft)
    sf1 = soft_f1(y_true_soft, y_pred_soft)

    print("\n=== SOFT EVALUATION (FER+) ===")
    print(f"mean_KL  : {kl:.4f}   (lower is better)")
    print(f"soft_F1  : {sf1:.4f}")

    metrics = {
        "mean_kl": kl,
        "soft_f1": sf1,
    }

    # -------------------------
    # Optional hard reporting
    # -------------------------
    y_true_hard = np.argmax(y_true_soft, axis=1)
    y_pred_hard = np.argmax(y_pred_soft, axis=1)

    if confidence_threshold is not None:
        conf = np.max(y_pred_soft, axis=1)
        mask = conf >= confidence_threshold

        print(
            f"\n=== HARD REPORT (confidence ≥ {confidence_threshold:.2f}) ==="
        )
        print(f"coverage: {mask.mean() * 100:.1f}%")

        y_true_hard = y_true_hard[mask]
        y_pred_hard = y_pred_hard[mask]

    else:
        print("\n=== HARD REPORT (ARGMAX, REPORTING ONLY) ===")

    print(
        classification_report(
            y_true_hard,
            y_pred_hard,
            target_names=config.CLASS_LABELS,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true_hard, y_pred_hard))

    return metrics
