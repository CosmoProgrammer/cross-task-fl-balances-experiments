
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# -- Publication style --
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Color palette (colorblind-friendly)
C_FED = "#2171b5"       # blue  — federated
C_CENT = "#e6550d"      # orange — centralized
C_FORE = "#2171b5"      # blue  — forecasting
C_ANOM = "#d94801"      # red-orange — anomaly
C_BACKBONE = "#3182bd"  # backbone
C_FC_HEAD = "#6baed6"   # forecasting head
C_AN_HEAD = "#fd8d3c"   # anomaly head

# Extended palette for multi-condition plots
CONDITION_COLORS = {
    "local_only_fedavg": "#9ecae1",    # light blue
    "single_task_fedavg": "#6baed6",   # medium blue
    "dual_fedavg": "#2171b5",          # dark blue (proposed)
    "dual_fedprox": "#08519c",         # navy
    "dual_scaffold": "#08306b",        # deep navy
    "centralized": "#e6550d",          # orange
    # Baselines
    "lstm": "#a1d99b",
    "informer": "#74c476",
    "msd_mixer": "#31a354",
    "lstm_ae": "#fdae6b",
    "ann_ae": "#fd8d3c",
}

CONDITION_LABELS = {
    "local_only_fedavg": "Local-Only",
    "single_task_fedavg": "Single-Task FL",
    "dual_fedavg": "Cross-Task FL (Ours)",
    "dual_fedprox": "Cross-Task FL + FedProx",
    "dual_scaffold": "Cross-Task FL + SCAFFOLD",
    "centralized": "Centralized",
}


def _save(fig, path, log=None):
    """Save figure as both PDF and PNG."""
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png")
    plt.close(fig)
    if log:
        log.info("  Saved %s.{pdf,png}", os.path.basename(path))


def _apply_style():
    plt.rcParams.update(STYLE)


# =====================================================================
#  1. FL Convergence Curves
# =====================================================================

def plot_fl_convergence(history, out_path, log=None):
    """Dual y-axis plot: forecasting MSE (left) and anomaly MSE (right)."""
    _apply_style()
    rounds = history["round"]
    fc_loss = history["forecast_test_loss"]
    an_loss = history["anomaly_test_loss"]

    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    ax2 = ax1.twinx()

    ln1 = ax1.plot(rounds, fc_loss, "o-", color=C_FORE, markersize=5,
                   linewidth=1.8, label="Forecasting MSE")
    ln2 = ax2.plot(rounds, an_loss, "s--", color=C_ANOM, markersize=5,
                   linewidth=1.8, label="Anomaly Recon. MSE")

    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Forecasting Test MSE", color=C_FORE)
    ax2.set_ylabel("Anomaly Test MSE", color=C_ANOM)
    ax1.tick_params(axis="y", labelcolor=C_FORE)
    ax2.tick_params(axis="y", labelcolor=C_ANOM)
    ax1.set_xticks(rounds)

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper right", framealpha=0.9)
    ax1.set_title("Federated Learning Convergence")

    fig.tight_layout()
    _save(fig, out_path, log)


# =====================================================================
#  2. Forecasting Metrics Comparison
# =====================================================================

def plot_forecasting_comparison(fed_metrics, cent_metrics, out_path, log=None):
    """Grouped bar chart comparing federated vs centralized forecasting."""
    _apply_style()
    metrics = ["MSE", "RMSE", "MAE"]
    keys = ["mse", "rmse", "mae"]

    fed_vals = [fed_metrics[k] for k in keys]
    cent_vals = [cent_metrics[k] for k in keys]

    x = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(4.5, 3))
    bars1 = ax.bar(x - width / 2, fed_vals, width, label="Federated",
                   color=C_FED, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, cent_vals, width, label="Centralized",
                   color=C_CENT, edgecolor="white", linewidth=0.5)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Error")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_title("Forecasting: Federated vs Centralized")

    fig.tight_layout()
    _save(fig, out_path, log)


def plot_r2_comparison(fed_metrics, cent_metrics, out_path, log=None):
    """Separate R2 comparison (since scale differs from error metrics)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(3, 3))

    vals = [fed_metrics["r2"], cent_metrics["r2"]]
    colors = [C_FED, C_CENT]
    labels = ["Federated", "Centralized"]
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="white")

    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.4f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("R$^2$ Score")
    ax.set_ylim(min(vals) * 0.95, 1.0)
    ax.set_title("Forecasting R$^2$")

    fig.tight_layout()
    _save(fig, out_path, log)


# =====================================================================
#  3. Anomaly Detection Metrics Comparison
# =====================================================================

def plot_anomaly_comparison(fed_metrics, cent_metrics, out_path, log=None):
    """Grouped bar chart for anomaly detection metrics."""
    _apply_style()
    metrics = ["F1", "AUC-ROC", "AUC-PR", "Precision", "Recall"]
    keys = ["f1", "auc_roc", "auc_pr", "precision", "recall"]

    fed_vals = [fed_metrics[k] for k in keys]
    cent_vals = [cent_metrics[k] for k in keys]

    x = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars1 = ax.bar(x - width / 2, fed_vals, width, label="Federated",
                   color=C_FED, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, cent_vals, width, label="Centralized",
                   color=C_CENT, edgecolor="white", linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc="upper left")
    ax.set_title("Anomaly Detection: Federated vs Centralized")

    fig.tight_layout()
    _save(fig, out_path, log)


# =====================================================================
#  4. Confusion Matrix Heatmap
# =====================================================================

def plot_confusion_matrix(anomaly_metrics, title, out_path, log=None):
    """Heatmap of TP/FP/FN/TN with counts and percentages."""
    _apply_style()

    tp = anomaly_metrics["tp"]
    fp = anomaly_metrics["fp"]
    fn = anomaly_metrics["fn"]
    tn = anomaly_metrics["tn"]
    total = tp + fp + fn + tn

    cm = np.array([[tn, fp], [fn, tp]])
    labels = np.array([
        [f"TN\n{tn:,}\n({tn/total*100:.1f}%)", f"FP\n{fp:,}\n({fp/total*100:.1f}%)"],
        [f"FN\n{fn:,}\n({fn/total*100:.1f}%)", f"TP\n{tp:,}\n({tp/total*100:.1f}%)"],
    ])

    fig, ax = plt.subplots(figsize=(3.5, 3))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, labels[i, j], ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    fig.tight_layout()
    _save(fig, out_path, log)


# =====================================================================
#  5. Model Parameter Breakdown
# =====================================================================

def plot_parameter_breakdown(model_params, out_path, log=None):
    """Stacked horizontal bar showing backbone vs head params."""
    _apply_style()

    backbone = model_params["shared_backbone"]
    fc_head = model_params["forecasting_head"]
    an_head = model_params["anomaly_head"]

    fig, ax = plt.subplots(figsize=(5, 2.2))

    labels = ["Forecasting\nModel", "Anomaly\nModel"]
    backbone_vals = [backbone, backbone]
    head_vals = [fc_head, an_head]
    head_colors = [C_FC_HEAD, C_AN_HEAD]

    y = np.arange(len(labels))
    h = 0.4

    ax.barh(y, backbone_vals, h, label="Shared Backbone",
            color=C_BACKBONE, edgecolor="white")
    for i in range(len(labels)):
        ax.barh(y[i], head_vals[i], h, left=backbone_vals[i],
                label="Task Head" if i == 0 else None,
                color=head_colors[i], edgecolor="white")

    # Annotations
    for i in range(len(labels)):
        total = backbone_vals[i] + head_vals[i]
        ax.text(total + total * 0.01, y[i],
                f"{total:,} total", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of Parameters")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("Model Architecture: Parameter Sharing")

    fig.tight_layout()
    _save(fig, out_path, log)


# =====================================================================
#  6. Training Time Comparison
# =====================================================================

def plot_training_time(fed_time, cent_fc_time, cent_an_time, out_path, log=None):
    """Bar chart of wall-clock training times."""
    _apply_style()

    fig, ax = plt.subplots(figsize=(4, 3))

    labels = ["Federated\n(both tasks)", "Centralized\nForecasting",
              "Centralized\nAnomaly"]
    times_h = [fed_time / 3600, cent_fc_time / 3600, cent_an_time / 3600]
    colors = [C_FED, C_CENT, "#fdae6b"]

    bars = ax.bar(labels, times_h, color=colors, width=0.5, edgecolor="white")

    for bar, t in zip(bars, times_h):
        ax.annotate(f"{t:.1f}h", xy=(bar.get_x() + bar.get_width() / 2,
                                      bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Training Time (hours)")
    ax.set_title("Training Time Comparison")

    fig.tight_layout()
    _save(fig, out_path, log)


# =====================================================================
#  7. Centralized Learning Curves
# =====================================================================

def plot_centralized_learning_curves(history, task_name, out_path, log=None):
    """Train vs validation loss over epochs for centralized training."""
    _apply_style()

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, ax = plt.subplots(figsize=(4.5, 3))

    ax.plot(epochs, history["train_loss"], "-", color=C_FED,
            linewidth=1.5, label="Train Loss", alpha=0.8)
    ax.plot(epochs, history["val_loss"], "-", color=C_CENT,
            linewidth=1.5, label="Val Loss", alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    ax.set_title(f"Centralized {task_name}: Learning Curves")

    fig.tight_layout()
    _save(fig, out_path, log)


# =====================================================================
#  8. Summary Metrics Table (as figure)
# =====================================================================

def plot_metrics_table(fed_results, cent_results, out_path, log=None):
    """Render a comparison metrics table as a figure for paper inclusion."""
    _apply_style()

    # Forecasting rows
    fc_fed = fed_results["forecasting_metrics"]
    fc_cent = cent_results["centralized_forecasting"]["metrics"]
    an_fed = fed_results["anomaly_metrics"]
    an_cent = cent_results["centralized_anomaly"]["metrics"]

    headers = ["Task", "Metric", "Federated", "Centralized", "Delta"]
    rows = []

    for m, fmt in [("mse", ".4f"), ("rmse", ".4f"), ("mae", ".4f"),
                    ("mape", ".2f"), ("r2", ".4f")]:
        fv, cv = fc_fed[m], fc_cent[m]
        delta = fv - cv
        sign = "+" if delta > 0 else ""
        label = "Forecast" if m == "mse" else ""
        rows.append([label, m.upper(),
                     f"{fv:{fmt}}", f"{cv:{fmt}}", f"{sign}{delta:{fmt}}"])

    for m in ["f1", "auc_roc", "auc_pr", "precision", "recall"]:
        fv, cv = an_fed[m], an_cent[m]
        delta = fv - cv
        sign = "+" if delta > 0 else ""
        label = "Anomaly" if m == "f1" else ""
        rows.append([label, m.upper().replace("_", "-"),
                     f"{fv:.4f}", f"{cv:.4f}", f"{sign}{delta:.4f}"])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#d9e2f3")
        table[(0, j)].set_text_props(weight="bold")

    # Alternate row shading
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f2f2f2")

    ax.set_title("Federated vs Centralized: Full Comparison", fontsize=11,
                 pad=10, weight="bold")

    fig.tight_layout()
    _save(fig, out_path, log)


# =====================================================================
#  9. FL Per-Round Loss Table
# =====================================================================

def plot_round_table(history, out_path, log=None):
    """Tabular figure showing per-round FL losses."""
    _apply_style()

    rounds = history["round"]
    fc_losses = history["forecast_test_loss"]
    an_losses = history["anomaly_test_loss"]

    headers = ["Round", "Forecast MSE", "Anomaly MSE",
               "Forecast Improv.", "Anomaly Improv."]
    rows = []
    for i, r in enumerate(rounds):
        fc_imp = "" if i == 0 else f"{(fc_losses[i-1]-fc_losses[i])/fc_losses[i-1]*100:.1f}%"
        an_imp = "" if i == 0 else f"{(an_losses[i-1]-an_losses[i])/an_losses[i-1]*100:.1f}%"
        rows.append([str(r), f"{fc_losses[i]:.6f}", f"{an_losses[i]:.6f}",
                     fc_imp, an_imp])

    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.4 * len(rows))))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#d9e2f3")
        table[(0, j)].set_text_props(weight="bold")

    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f2f2f2")

    ax.set_title("Federated Learning: Per-Round Test Loss", fontsize=11,
                 pad=10, weight="bold")

    fig.tight_layout()
    _save(fig, out_path, log)


# =====================================================================
#  10. Anomaly Score Distribution
# =====================================================================

def plot_anomaly_score_distribution(anomaly_metrics, out_path, log=None):
    """Visualize threshold on anomaly scores (using summary stats)."""
    _apply_style()

    tp = anomaly_metrics["tp"]
    fp = anomaly_metrics["fp"]
    fn = anomaly_metrics["fn"]
    tn = anomaly_metrics["tn"]
    threshold = anomaly_metrics["threshold"]

    fig, ax = plt.subplots(figsize=(4.5, 3))

    # Stacked bar showing classification outcome
    categories = ["Below Threshold\n(Predicted Normal)",
                  "Above Threshold\n(Predicted Anomaly)"]
    normal_counts = [tn, fp]
    anomaly_counts = [fn, tp]

    x = np.arange(len(categories))
    width = 0.5

    ax.bar(x, normal_counts, width, label="True Normal", color="#a1d99b",
           edgecolor="white")
    ax.bar(x, anomaly_counts, width, bottom=normal_counts,
           label="True Anomaly", color="#fc9272", edgecolor="white")

    # Annotations
    for i, (nc, ac) in enumerate(zip(normal_counts, anomaly_counts)):
        ax.text(i, nc / 2, f"{nc:,}", ha="center", va="center", fontsize=7)
        ax.text(i, nc + ac / 2, f"{ac:,}", ha="center", va="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("Window Count")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Anomaly Detection Outcomes (threshold={threshold:.4f})")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))

    fig.tight_layout()
    _save(fig, out_path, log)


# =====================================================================
#  11. Multi-Condition Forecasting Comparison
# =====================================================================

def plot_multi_condition_forecasting(conditions, out_path, log=None):
    """Grouped bar chart: forecasting MSE across all conditions.

    Args:
        conditions: dict of {label: {"forecasting_metrics": {...}}}
    """
    _apply_style()
    metric_keys = ["mse", "rmse", "mae"]
    metric_labels = ["MSE", "RMSE", "MAE"]

    labels = list(conditions.keys())
    n_cond = len(labels)
    n_metrics = len(metric_keys)

    x = np.arange(n_metrics)
    width = 0.7 / n_cond

    fig, ax = plt.subplots(figsize=(max(5, n_cond * 1.2), 3.5))

    for i, label in enumerate(labels):
        vals = [conditions[label]["forecasting_metrics"][k]
                for k in metric_keys]
        color = CONDITION_COLORS.get(
            label, plt.cm.tab10(i / max(n_cond, 1)))
        display = CONDITION_LABELS.get(label, label)
        offset = (i - n_cond / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=display, color=color,
                      edgecolor="white", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6, rotation=45)

    ax.set_ylabel("Error")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=7, loc="upper right", ncol=1)
    ax.set_title("Forecasting: Multi-Condition Comparison")
    fig.tight_layout()
    _save(fig, out_path, log)


# =====================================================================
#  12. Multi-Condition Anomaly Comparison
# =====================================================================

def plot_multi_condition_anomaly(conditions, out_path, log=None):
    """Grouped bar chart: anomaly metrics across all conditions.

    Args:
        conditions: dict of {label: {"anomaly_metrics": {...}}}
    """
    _apply_style()
    metric_keys = ["f1", "auc_roc", "auc_pr", "precision", "recall"]
    metric_labels = ["F1", "AUC-ROC", "AUC-PR", "Precision", "Recall"]

    labels = list(conditions.keys())
    n_cond = len(labels)
    n_metrics = len(metric_keys)

    x = np.arange(n_metrics)
    width = 0.7 / n_cond

    fig, ax = plt.subplots(figsize=(max(6, n_cond * 1.5), 3.5))

    for i, label in enumerate(labels):
        vals = [conditions[label]["anomaly_metrics"][k]
                for k in metric_keys]
        color = CONDITION_COLORS.get(
            label, plt.cm.tab10(i / max(n_cond, 1)))
        display = CONDITION_LABELS.get(label, label)
        offset = (i - n_cond / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=display, color=color,
                      edgecolor="white", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6, rotation=45)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=7, loc="upper left", ncol=1)
    ax.set_title("Anomaly Detection: Multi-Condition Comparison")
    fig.tight_layout()
    _save(fig, out_path, log)


# =====================================================================
#  13. Multi-Condition Summary Table
# =====================================================================

def plot_multi_condition_table(conditions, out_path, log=None):
    """Render a comprehensive comparison table as a figure.

    Args:
        conditions: dict of {label: {"forecasting_metrics": {...},
                                      "anomaly_metrics": {...}}}
    """
    _apply_style()

    labels = list(conditions.keys())
    display_labels = [CONDITION_LABELS.get(l, l) for l in labels]

    # Build rows: one per metric
    fc_metrics = [("MSE", "mse", ".4f"), ("RMSE", "rmse", ".4f"),
                  ("MAE", "mae", ".4f"), ("R2", "r2", ".4f")]
    an_metrics = [("F1", "f1", ".4f"), ("AUC-ROC", "auc_roc", ".4f"),
                  ("AUC-PR", "auc_pr", ".4f")]

    headers = ["Task", "Metric"] + display_labels
    rows = []

    for i, (display, key, fmt) in enumerate(fc_metrics):
        task_label = "Forecast" if i == 0 else ""
        row = [task_label, display]
        for label in labels:
            v = conditions[label]["forecasting_metrics"].get(key, float("nan"))
            row.append(f"{v:{fmt}}")
        rows.append(row)

    for i, (display, key, fmt) in enumerate(an_metrics):
        task_label = "Anomaly" if i == 0 else ""
        row = [task_label, display]
        for label in labels:
            v = conditions[label]["anomaly_metrics"].get(key, float("nan"))
            row.append(f"{v:{fmt}}")
        rows.append(row)

    n_cols = len(headers)
    fig_w = max(6, 1.5 * n_cols)
    fig, ax = plt.subplots(figsize=(fig_w, 3.5))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)

    for j in range(n_cols):
        table[(0, j)].set_facecolor("#d9e2f3")
        table[(0, j)].set_text_props(weight="bold")

    for i in range(1, len(rows) + 1):
        for j in range(n_cols):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f2f2f2")

    ax.set_title("Multi-Condition Comparison", fontsize=11, pad=10,
                 weight="bold")
    fig.tight_layout()
    _save(fig, out_path, log)


# =====================================================================
#  14. FL Convergence Overlay (multiple conditions)
# =====================================================================

def plot_multi_fl_convergence(fed_results_all, out_path, log=None):
    """Overlay FL convergence curves from multiple conditions."""
    _apply_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    for key, data in fed_results_all.items():
        hist = data.get("history", {})
        rounds = hist.get("round", [])
        fc_loss = hist.get("forecast_test_loss", [])
        an_loss = hist.get("anomaly_test_loss", [])
        if not rounds:
            continue

        color = CONDITION_COLORS.get(key, None)
        display = CONDITION_LABELS.get(key, key)

        ax1.plot(rounds, fc_loss, "o-", color=color, markersize=4,
                 linewidth=1.5, label=display)
        ax2.plot(rounds, an_loss, "s-", color=color, markersize=4,
                 linewidth=1.5, label=display)

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Forecasting Test MSE")
    ax1.legend(fontsize=7)
    ax1.set_title("Forecasting Convergence")

    ax2.set_xlabel("Round")
    ax2.set_ylabel("Anomaly Test MSE")
    ax2.legend(fontsize=7)
    ax2.set_title("Anomaly Convergence")

    fig.tight_layout()
    _save(fig, out_path, log)


# =====================================================================
#  GENERATE ALL PLOTS
# =====================================================================

def generate_all_plots(fed_results, cent_results, figures_dir, log=None,
                       fed_results_all=None, baseline_results=None):
    """Generate all research-paper plots from result JSON data.

    Args:
        fed_results: dict from primary federated result (dual_fedavg) or None
        cent_results: dict from centralized_results.json (or None)
        figures_dir: output directory for figures
        log: logger instance
        fed_results_all: dict of {mode_strategy: result_dict} for all FL runs
        baseline_results: dict from baseline_results.json (or None)
    """
    os.makedirs(figures_dir, exist_ok=True)
    if log:
        log.info("Generating figures in %s", figures_dir)

    # 1. FL convergence
    if fed_results and "history" in fed_results:
        plot_fl_convergence(
            fed_results["history"],
            os.path.join(figures_dir, "fig1_fl_convergence"), log)

        plot_round_table(
            fed_results["history"],
            os.path.join(figures_dir, "fig9_round_table"), log)

    # 2-3. Comparison plots (need both results)
    if fed_results and cent_results:
        fc_fed = fed_results["forecasting_metrics"]
        fc_cent = cent_results["centralized_forecasting"]["metrics"]
        an_fed = fed_results["anomaly_metrics"]
        an_cent = cent_results["centralized_anomaly"]["metrics"]

        plot_forecasting_comparison(
            fc_fed, fc_cent,
            os.path.join(figures_dir, "fig2_forecasting_comparison"), log)

        plot_r2_comparison(
            fc_fed, fc_cent,
            os.path.join(figures_dir, "fig3_r2_comparison"), log)

        plot_anomaly_comparison(
            an_fed, an_cent,
            os.path.join(figures_dir, "fig4_anomaly_comparison"), log)

        plot_metrics_table(
            fed_results, cent_results,
            os.path.join(figures_dir, "fig8_metrics_table"), log)

        # Training time
        fed_time = fed_results.get("training_time_seconds", 0)
        cent_fc_time = cent_results["centralized_forecasting"].get(
            "training_time_seconds", 0)
        cent_an_time = cent_results["centralized_anomaly"].get(
            "training_time_seconds", 0)
        if fed_time > 0 or cent_fc_time > 0:
            plot_training_time(
                fed_time, cent_fc_time, cent_an_time,
                os.path.join(figures_dir, "fig6_training_time"), log)

    # 4. Confusion matrix
    if fed_results and "anomaly_metrics" in fed_results:
        an = fed_results["anomaly_metrics"]
        if all(k in an for k in ["tp", "fp", "fn", "tn"]):
            plot_confusion_matrix(
                an, "Federated Anomaly Detection",
                os.path.join(figures_dir, "fig5a_confusion_matrix_fed"), log)

    if cent_results and "centralized_anomaly" in cent_results:
        an = cent_results["centralized_anomaly"]["metrics"]
        if all(k in an for k in ["tp", "fp", "fn", "tn"]):
            plot_confusion_matrix(
                an, "Centralized Anomaly Detection",
                os.path.join(figures_dir, "fig5b_confusion_matrix_cent"), log)

    # 5. Parameter breakdown
    if fed_results and "model_params" in fed_results:
        plot_parameter_breakdown(
            fed_results["model_params"],
            os.path.join(figures_dir, "fig7_parameter_breakdown"), log)

    # 7. Centralized learning curves
    if cent_results:
        fc_hist = cent_results.get("centralized_forecasting", {}).get("history")
        if fc_hist and fc_hist.get("train_loss"):
            plot_centralized_learning_curves(
                fc_hist, "Forecasting",
                os.path.join(figures_dir, "fig10_cent_fc_learning_curve"), log)

        an_hist = cent_results.get("centralized_anomaly", {}).get("history")
        if an_hist and an_hist.get("train_loss"):
            plot_centralized_learning_curves(
                an_hist, "Anomaly Detection",
                os.path.join(figures_dir, "fig11_cent_an_learning_curve"), log)

    # 10. Anomaly score distribution
    if fed_results and "anomaly_metrics" in fed_results:
        plot_anomaly_score_distribution(
            fed_results["anomaly_metrics"],
            os.path.join(figures_dir, "fig12_anomaly_score_dist"), log)

    # ── Multi-condition plots (need 2+ conditions) ──
    if fed_results_all is None:
        fed_results_all = {}

    # Build unified conditions dict for multi-condition plots
    conditions = {}
    for key, data in fed_results_all.items():
        if "forecasting_metrics" in data and "anomaly_metrics" in data:
            conditions[key] = data

    if cent_results:
        conditions["centralized"] = {
            "forecasting_metrics": cent_results["centralized_forecasting"]["metrics"],
            "anomaly_metrics": cent_results["centralized_anomaly"]["metrics"],
        }

    if baseline_results:
        # Add forecasting baselines
        for bname in ["lstm", "informer", "msd_mixer"]:
            bkey = f"forecasting_{bname}"
            if bkey in baseline_results and "metrics" in baseline_results[bkey]:
                bdata = baseline_results[bkey]
                if bname not in conditions:
                    conditions[bname] = {
                        "forecasting_metrics": bdata["metrics"],
                        "anomaly_metrics": {},
                    }
                else:
                    conditions[bname]["forecasting_metrics"] = bdata["metrics"]

        # Add anomaly baselines
        for bname in ["lstm_ae", "ann_ae"]:
            bkey = f"anomaly_{bname}"
            if bkey in baseline_results and "metrics" in baseline_results[bkey]:
                bdata = baseline_results[bkey]
                if bname not in conditions:
                    conditions[bname] = {
                        "forecasting_metrics": {},
                        "anomaly_metrics": bdata["metrics"],
                    }
                else:
                    conditions[bname]["anomaly_metrics"] = bdata["metrics"]

        # MSD-Mixer anomaly
        if "anomaly_msd_mixer" in baseline_results:
            bdata = baseline_results["anomaly_msd_mixer"]
            if "msd_mixer" in conditions:
                conditions["msd_mixer"]["anomaly_metrics"] = bdata.get("metrics", {})

    if len(conditions) >= 2:
        # Filter conditions that have forecasting metrics
        fc_conditions = {k: v for k, v in conditions.items()
                        if v.get("forecasting_metrics")}
        an_conditions = {k: v for k, v in conditions.items()
                        if v.get("anomaly_metrics")}

        if len(fc_conditions) >= 2:
            plot_multi_condition_forecasting(
                fc_conditions,
                os.path.join(figures_dir, "fig13_multi_forecasting"), log)

        if len(an_conditions) >= 2:
            plot_multi_condition_anomaly(
                an_conditions,
                os.path.join(figures_dir, "fig14_multi_anomaly"), log)

        # Table with conditions that have BOTH metrics
        both_conditions = {k: v for k, v in conditions.items()
                          if v.get("forecasting_metrics") and
                          v.get("anomaly_metrics")}
        if len(both_conditions) >= 2:
            plot_multi_condition_table(
                both_conditions,
                os.path.join(figures_dir, "fig15_multi_table"), log)

    # Multi-FL convergence overlay
    if len(fed_results_all) >= 2:
        plot_multi_fl_convergence(
            fed_results_all,
            os.path.join(figures_dir, "fig16_multi_convergence"), log)
