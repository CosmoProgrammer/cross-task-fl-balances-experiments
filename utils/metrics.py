"""
Metrics for forecasting and anomaly detection.

Anomaly threshold: MUST be determined on VALIDATION set, then applied to test.
"""
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, accuracy_score,
)


# ── Forecasting Metrics ──

def compute_forecasting_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mse_val = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse_val,
        "rmse": float(np.sqrt(mse_val)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(_mape(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _mape(y_true, y_pred, eps=1e-8):
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# ── Anomaly Detection Metrics ──

def find_threshold_on_validation(val_scores, val_labels):
    """Find best F1 threshold on VALIDATION data only."""
    val_scores = np.array(val_scores).flatten()
    val_labels = np.array(val_labels).flatten()

    percentiles = np.arange(1, 100, 1)
    thresholds = np.percentile(val_scores, percentiles)
    best_f1, best_thresh = 0.0, float(np.median(val_scores))

    for thresh in thresholds:
        y_pred = (val_scores >= thresh).astype(int)
        f1_val = f1_score(val_labels, y_pred, zero_division=0)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thresh = float(thresh)

    return best_thresh


def compute_anomaly_metrics(y_true, y_scores, threshold):
    """Compute anomaly metrics using a PRE-DETERMINED threshold."""
    y_true = np.array(y_true).flatten()
    y_scores = np.array(y_scores).flatten()

    try:
        auc_roc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc_roc = 0.0
    try:
        auc_pr = average_precision_score(y_true, y_scores)
    except ValueError:
        auc_pr = 0.0

    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
