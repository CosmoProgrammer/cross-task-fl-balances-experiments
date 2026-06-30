"""Matched-operating-point anomaly analysis across cohorts (laptop, no GPU/data).

Consumes the raw per-window score dumps written by
`eval_from_checkpoint.py --dump-scores` (the re-derivable artifact) and answers
the question the result JSONs cannot: did *more training data* improve the
anomaly DETECTOR, or did it only move the validation-selected operating point?

The c50->c100 result JSONs show F1 0.526 -> 0.488 (down) while AUC-ROC/AUC-PR
are flat -- because the best-F1-on-validation threshold dropped ~9x, flipping a
precision-heavy detector (P.99/R.36) into a recall-heavy one (P.35/R.80). This
script makes that concrete and separates three things that the single reported
F1 conflates:
  1. discrimination     -> threshold-free: AUC-ROC, AUC-PR (same fixed test set)
  2. selection transfer -> apply each model's val threshold to the OTHER's test
                           scores; and read each model's TEST-oracle best F1
  3. matched sensitivity -> precision @ a grid of fixed recall targets (scale-free)

Both cohorts use the IDENTICAL fixed 15-building test set, so the score arrays
are directly comparable.

    conda run -n crosstask python analyze_anomaly_operating_point.py \
        --a anomaly_scores_c50.npz --b anomaly_scores_c100.npz
"""
import argparse

import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_recall_curve)


def _load(path):
    d = np.load(path)
    return {
        "cohort": int(d["cohort"]) if "cohort" in d else -1,
        "val_scores": d["val_scores"], "val_labels": d["val_labels"],
        "test_scores": d["test_scores"], "test_labels": d["test_labels"],
        "val_threshold": float(d["val_threshold"]) if "val_threshold" in d else None,
    }


def _prf_at(scores, labels, thresh):
    """Precision/recall/F1 of `scores >= thresh` against `labels`."""
    pred = (scores >= thresh).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _oracle_test_f1(scores, labels):
    """Best F1 achievable on TEST by any threshold (selection-free ceiling)."""
    p, r, thr = precision_recall_curve(labels, scores)
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    i = int(np.argmax(f1))
    # precision_recall_curve: thr has len-1 vs p/r; last point is recall=0
    t = float(thr[min(i, len(thr) - 1)])
    return float(f1[i]), t


def _precision_at_recall(scores, labels, recall_targets):
    """Best precision achievable at recall >= each target (off the PR curve)."""
    p, r, _ = precision_recall_curve(labels, scores)
    out = {}
    for rt in recall_targets:
        mask = r >= rt
        out[rt] = float(p[mask].max()) if mask.any() else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first npz dump (e.g. c50, or dual)")
    ap.add_argument("--b", required=True, help="second npz dump (e.g. c100, or single_task)")
    ap.add_argument("--label-a", default=None,
                    help="label for --a (default cN from the dump). Use for "
                         "same-cohort comparisons, e.g. --label-a dual --label-b single_task")
    ap.add_argument("--label-b", default=None, help="label for --b")
    args = ap.parse_args()

    A, B = _load(args.a), _load(args.b)
    la = args.label_a or f"c{A['cohort']}"
    lb = args.label_b or f"c{B['cohort']}"

    # Sanity: same fixed test set -> identical label vector (count + composition)
    same_n = A["test_labels"].shape == B["test_labels"].shape
    same_pos = int(A["test_labels"].sum()) == int(B["test_labels"].sum())
    print(f"\n[test set] {la}: {A['test_labels'].size} windows, "
          f"{int(A['test_labels'].sum())} pos | "
          f"{lb}: {B['test_labels'].size} windows, {int(B['test_labels'].sum())} pos "
          f"| identical={'YES' if (same_n and same_pos) else 'NO (NOT comparable!)'}")

    # 1. Threshold-free discrimination
    print("\n=== 1. Discrimination (threshold-free) ===")
    print(f"{'':<16}{la:>12}{lb:>12}")
    for name, fn in [("AUC-ROC", roc_auc_score), ("AUC-PR", average_precision_score)]:
        va = fn(A["test_labels"], A["test_scores"])
        vb = fn(B["test_labels"], B["test_scores"])
        print(f"{name:<16}{va:>12.4f}{vb:>12.4f}   d={vb - va:+.4f}")

    # 2a. As-reported: each model's own val-selected threshold -> its test
    print("\n=== 2a. As-reported (own val-best-F1 threshold) ===")
    print(f"{'':<16}{'thresh':>10}{'prec':>8}{'recall':>8}{'F1':>8}")
    for lab, M in [(la, A), (lb, B)]:
        t = M["val_threshold"]
        p, r, f1 = _prf_at(M["test_scores"], M["test_labels"], t)
        print(f"{lab:<16}{t:>10.4f}{p:>8.3f}{r:>8.3f}{f1:>8.3f}")

    # 2b. Threshold transfer: is the F1 swing just the operating point?
    print("\n=== 2b. Threshold transfer (cross-applied val thresholds) ===")
    for src, dst, ls, ld in [(A, B, la, lb), (B, A, lb, la)]:
        p, r, f1 = _prf_at(dst["test_scores"], dst["test_labels"], src["val_threshold"])
        print(f"  {ls}'s threshold ({src['val_threshold']:.4f}) on {ld} test: "
              f"P={p:.3f} R={r:.3f} F1={f1:.3f}")

    # 2c. Oracle test threshold: best F1 each model could reach (model ceiling)
    print("\n=== 2c. Oracle TEST-best F1 (selection-free ceiling) ===")
    for lab, M in [(la, A), (lb, B)]:
        f1, t = _oracle_test_f1(M["test_scores"], M["test_labels"])
        print(f"  {lab}: best-F1={f1:.3f} @ thresh={t:.4f}")

    # 3. Matched sensitivity: precision at fixed recall (scale-free apples-to-apples)
    print("\n=== 3. Precision @ fixed recall (matched sensitivity) ===")
    targets = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    pa = _precision_at_recall(A["test_scores"], A["test_labels"], targets)
    pb = _precision_at_recall(B["test_scores"], B["test_labels"], targets)
    print(f"{'recall':>8}{la:>12}{lb:>12}{'delta':>10}")
    for rt in targets:
        print(f"{rt:>8.2f}{pa[rt]:>12.3f}{pb[rt]:>12.3f}{pb[rt] - pa[rt]:>+10.3f}")

    print("\nReading: AUC + P@R + oracle-F1 are the threshold-FREE quality of the "
          "detector; if they differ, that's a real difference. If they're flat but "
          "the as-reported F1 (2a) differs, the gap is threshold SELECTION (2b/2c), "
          "not discrimination.\n")


if __name__ == "__main__":
    main()
