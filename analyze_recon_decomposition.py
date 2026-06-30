"""Per-window reconstruction-error decomposition across anomaly detectors.

Laptop-side, numpy/sklearn/scipy only (no GPU / no data / no retrain). Loads the
`anomaly_scores_*.npz` dumps produced by `eval_from_checkpoint.py --dump-scores`
and decomposes each detector's per-window recon error by window type (normal vs
anomalous) on the FIXED shared test set.

Purpose (see EXPERIMENT_RESULTS.md §6d): test the autoencoder over-generalization /
negative-transfer mechanism DIRECTLY, rather than inferring it from aggregate recon
MSE. Aggregate MSE is dominated by the ~20% anomaly windows and therefore *measures*
over-generalization rather than reconstruction quality; the decomposition separates
the two.

Usage:
    python analyze_recon_decomposition.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

FILES = {
    'dual':        'anomaly_scores_c100.npz',
    'single_task': 'anomaly_scores_c100_single_task.npz',
    'local_only':  'anomaly_scores_c100_local_only.npz',
    'centralized': 'anomaly_scores_c100_centralised.npz',
    'dual_c50':    'anomaly_scores_c50.npz',
}


def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp


def main():
    data = {k: np.load(v, allow_pickle=True) for k, v in FILES.items()}

    ref = data['dual']['test_labels']
    print("Test windows:", ref.shape[0], " positives:", int(ref.sum()),
          " prevalence: %.4f" % ref.mean())
    for k, d in data.items():
        print(f"  {k:12s} test_labels identical to dual: "
              f"{np.array_equal(d['test_labels'], ref)}")
    print()

    print("=" * 100)
    print("PER-DETECTOR recon-error separation: NORMAL (y=0) vs ANOMALOUS (y=1) test windows")
    print("=" * 100)
    hdr = (f"{'detector':12s} {'AUC-ROC':>8s} {'AUC-PR':>7s} | {'mu_norm':>8s} {'mu_anom':>8s} "
           f"{'ratio':>6s} | {'med_norm':>9s} {'med_anom':>9s} {'mratio':>6s} | "
           f"{'cohen_d':>7s} {'cd_log':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for k, d in data.items():
        s = d['test_scores']; y = d['test_labels']
        s0, s1 = s[y == 0], s[y == 1]
        auc = roc_auc_score(y, s); ap = average_precision_score(y, s)
        mu0, mu1 = s0.mean(), s1.mean()
        md0, md1 = np.median(s0), np.median(s1)
        cd = cohens_d(s1, s0)
        cdl = cohens_d(np.log(s1), np.log(s0))
        print(f"{k:12s} {auc:8.4f} {ap:7.4f} | {mu0:8.4f} {mu1:8.4f} {mu1/mu0:6.2f} | "
              f"{md0:9.4f} {md1:9.4f} {md1/md0:6.2f} | {cd:7.3f} {cdl:7.3f}")

    print("\n  mu_norm  = mean recon error on NORMAL windows (lower => 'reconstructs better')")
    print("  ratio    = mu_anom/mu_norm (scale-free; HIGHER = stronger detection signal)")
    print("  cd_log   = log-domain Cohen's d (orders detectors by AUC; errors are multiplicative)\n")

    print("=" * 100)
    print("PAIRED dual vs single_task (each normalized by its OWN normal-window mean => normal=1.0)")
    print("=" * 100)
    sd, ss, y = data['dual']['test_scores'], data['single_task']['test_scores'], ref
    nd, ns = sd / sd[y == 0].mean(), ss / ss[y == 0].mean()
    print(f"  normalized ANOM mean: dual={nd[y == 1].mean():.3f}  single={ns[y == 1].mean():.3f}"
          f"  => single separates anomalies {ns[y == 1].mean() / nd[y == 1].mean():.2f}x more")

    print("\nSpearman rank-corr of test scores between detectors:")
    names = ['dual', 'single_task', 'local_only', 'centralized']
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = spearmanr(data[names[i]]['test_scores'], data[names[j]]['test_scores']).correlation
            print(f"  {names[i]:12s} vs {names[j]:12s}: rho = {r:.4f}")


if __name__ == '__main__':
    main()
