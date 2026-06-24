"""
Metrics for forecasting and anomaly detection.

Anomaly threshold: MUST be determined on VALIDATION set, then applied to test.
"""
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, accuracy_score,
    matthews_corrcoef,
)


# ── Forecasting Metrics ──

def compute_forecasting_metrics(y_true, y_pred):
    # Keep the structured (N, H, C) arrays for MASE's per-window naive scale
    # before flattening the rest.
    yt_struct = np.asarray(y_true)
    yp_struct = np.asarray(y_pred)
    y_true = yt_struct.flatten()
    y_pred = yp_struct.flatten()
    mse_val = float(mean_squared_error(y_true, y_pred))
    rmse_val = float(np.sqrt(mse_val))
    mae_val = float(mean_absolute_error(y_true, y_pred))
    denom = float(np.mean(np.abs(y_true)))
    naive = _naive_scale(yt_struct)
    return {
        "mse": mse_val,
        "rmse": rmse_val,
        "mae": mae_val,
        "mape": float(_mape(y_true, y_pred)),
        "smape": float(_smape(y_true, y_pred)),
        "wape": float(_wape(y_true, y_pred)),
        # mean-normalized RMSE -> scale-free, comparable across cohorts/buildings
        "nrmse": float(rmse_val / denom) if denom > 0 else 0.0,
        # MASE scaled by the in-window one-step (persistence) naive MAE. Self
        # contained -- does not need the training series. NOTE: this is the
        # within-window naive (m=1); a strict per-building train-scaled MASE
        # would require threading building identity + train series into eval.
        "mase": float(mae_val / naive) if naive > 0 else 0.0,
        # RSE + CORR: the Informer/LSTNet reporting pair -> direct comparability
        # with those baselines.
        "rse": float(_rse(y_true, y_pred)),
        "corr": float(_corr(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_per_horizon_metrics(y_true, y_pred):
    """Per-step error across the forecast horizon.

    Expects the structured (N, H, C) arrays from evaluate_forecasting (NOT
    flattened). Returns one (mae, rmse) per horizon step h=1..H so the error
    growth over the horizon can be tabulated/plotted -- extra reporting depth
    from runs that are already done.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim < 2:
        return []  # nothing structured to break down
    H = y_true.shape[1]
    out = []
    for h in range(H):
        t = y_true[:, h, ...].flatten()
        p = y_pred[:, h, ...].flatten()
        mse_h = float(mean_squared_error(t, p))
        out.append({
            "horizon": h + 1,
            "mae": float(mean_absolute_error(t, p)),
            "rmse": float(np.sqrt(mse_h)),
        })
    return out


def _mape(y_true, y_pred, eps=1e-8):
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def _smape(y_true, y_pred, eps=1e-8):
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > eps
    if mask.sum() == 0:
        return 0.0
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100


def _wape(y_true, y_pred, eps=1e-8):
    denom = np.sum(np.abs(y_true))
    if denom < eps:
        return 0.0
    return np.sum(np.abs(y_true - y_pred)) / denom * 100


def _naive_scale(y_true_struct, eps=1e-8):
    """Mean in-window one-step |diff| of the targets -- the MASE denominator.

    Diffs are taken WITHIN each window (along the horizon axis) so window
    boundaries are never crossed; needs the (N, H, ...) structured array.
    """
    a = np.asarray(y_true_struct)
    if a.ndim < 2:
        d = np.abs(np.diff(a))
    else:
        d = np.abs(np.diff(a, axis=1))
    return float(np.mean(d)) if d.size else 0.0


def _rse(y_true, y_pred, eps=1e-8):
    """Root relative squared error (Informer/LSTNet definition)."""
    denom = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom < eps:
        return 0.0
    return np.sqrt(np.sum((y_true - y_pred) ** 2)) / denom


def _corr(y_true, y_pred, eps=1e-8):
    """Pearson correlation between predictions and targets."""
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    denom = np.sqrt(np.sum(yt ** 2) * np.sum(yp ** 2))
    if denom < eps:
        return 0.0
    return float(np.sum(yt * yp) / denom)


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

    tpr = tp / (tp + fn) if (tp + fn) else 0.0   # recall / sensitivity
    tnr = tn / (tn + fp) if (tn + fp) else 0.0   # specificity
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float((tpr + tnr) / 2),
        # false-alarm rate -- operationally important for energy monitoring
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        # Matthews correlation: balanced single-number score, robust to the
        # class imbalance in anomaly detection (uses all of tp/fp/tn/fn).
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if (tp + fn) and (tn + fp) else 0.0,
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
