import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

def compute_turn_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Calculates turn detection metrics:
    - F1 Score
    - Precision
    - Recall
    - Accuracy
    - False Early-End Rate (% of CONTINUE turns incorrectly predicted as END)
    - Confusion Matrix [TN, FP, FN, TP]
    """
    y_pred = (y_pred_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # False Early-End Rate = FP / (FP + TN)
    total_continue = tn + fp
    false_early_end_rate = (fp / total_continue) * 100.0 if total_continue > 0 else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_early_end_rate_pct": float(false_early_end_rate),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }
