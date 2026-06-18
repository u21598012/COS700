"""Classification metrics shared by every algorithm (framework spec, Phase 10).

Macro averaging is used for precision/recall/F1 because the MIT-BIH classes are
heavily imbalanced and we care about minority arrhythmia classes.
"""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred):
    """Return a dict of accuracy / precision / recall / macro-F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def get_confusion_matrix(y_true, y_pred, num_classes=None):
    labels = list(range(num_classes)) if num_classes else None
    return confusion_matrix(y_true, y_pred, labels=labels)
