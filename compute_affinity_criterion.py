"""Theorem-3 share-or-split criterion, instantiated on committed update-conflict dumps.

Computes the debiased plug-in statistic T_hat for a task PAIR from the per-client
backbone deltas that `analyze_update_conflict.py` already dumped. This is the
laptop-side, no-retrain instantiation of theory/NOTES_theorem3.md on real data, and
the object frozen in PREREGISTERED_PREDICTIONS.md.

Theorem 3 (target (a), NOTES_theorem3.md):
    S_t        := task-t mean backbone delta          (the aggregate d_fc / d_an)
    tau_hat^2  := pooled within-task scatter (unbiased):
                    tau_hat^2 = [ sum_t sum_{i in t} ||d_i - mu_t||^2 ] / [ sum_t (n_t - 1) ]
    gamma_hat^2 := tau_hat^2 / n         (per-task estimation error of the mean)
    delta*^2    := 2 gamma_hat^2         (the when-to-share threshold)
    T_hat       := ||S_1 - S_2||^2 - 4 gamma_hat^2      (debiased; E[T_hat] = delta^2 - 2 gamma^2)
    RULE: share (dual) iff T_hat < 0;  else split (single_task).
The debiasing (-4 gamma^2) removes the delta*^2 = 2 gamma^2 inflation of ||S_1-S_2||^2
(each mean contributes a gamma^2); it is load-bearing NEAR the threshold, not here.

All inner products are reconstructed EXACTLY from the dumped per-client delta norms
+ full pairwise cosine matrix (their Gram matrix), so no per-client delta vectors are
needed (they aren't dumped). Units = total squared L2 over all backbone scalars,
matching the v0/Theorem-3 convention (E||xi||^2 = sigma^2 over the whole vector).

    conda run -n crosstask python compute_affinity_criterion.py
    conda run -n crosstask python compute_affinity_criterion.py --tag dual_c100
"""
import argparse
import json
import os

import numpy as np


def criterion_from_dump(npz_path, js_path):
    z = np.load(npz_path, allow_pickle=False)
    tasks = z["tasks"]
    norms = z["delta_norms"].astype(np.float64)
    cos = z["cos_matrix"].astype(np.float64)
    labels = sorted(set(tasks.tolist()))
    assert len(labels) == 2, f"expected a task PAIR, got {labels}"
    a, b = labels
    ma, mb = tasks == a, tasks == b
    n_a, n_b = int(ma.sum()), int(mb.sum())
    n = n_a  # equal group sizes by design (fixed cohort split)

    # Gram matrix G_ij = ||d_i|| ||d_j|| cos_ij ; exact diagonal
    G = np.outer(norms, norms) * cos
    np.fill_diagonal(G, norms ** 2)

    def blk(m1, m2):
        return G[np.ix_(m1, m2)].sum()

    Saa, Sbb, Sab = blk(ma, ma), blk(mb, mb), blk(ma, mb)
    # within-task scatter around the uniform task mean; pooled unbiased tau^2
    SS_a = G[ma, ma].sum() - Saa / n_a
    SS_b = G[mb, mb].sum() - Sbb / n_b
    tau2 = (SS_a + SS_b) / ((n_a - 1) + (n_b - 1))
    # ||mu_a - mu_b||^2 with uniform means (internally consistent with the model)
    D2_uni = Saa / n_a ** 2 + Sbb / n_b ** 2 - 2 * Sab / (n_a * n_b)

    # ||S_a - S_b||^2 from the WEIGHTED aggregates the trainer actually averages
    # (the real-data map). Prefer the per-task aggregate delta VECTORS dumped in
    # the npz (generic, any pair); fall back to the JSON norms+cos of the
    # committed FC-AD anchor dumps (which predate the vector dump), so the frozen
    # pre-registration number stays byte-reproducible.
    D2_wt = None
    ga, gb = f"agg__{a}", f"agg__{b}"
    if ga in z.files and gb in z.files:
        da, db = z[ga].astype(np.float64), z[gb].astype(np.float64)
        D2_wt = float(np.sum((da - db) ** 2))
    elif os.path.exists(js_path):
        js = json.load(open(js_path))
        cos = js.get("headline_cos_agg")
        agg = js.get("agg_norms")            # generic {label: norm}
        if agg is not None and cos is not None and a in agg and b in agg:
            D2_wt = agg[a] ** 2 + agg[b] ** 2 - 2 * agg[a] * agg[b] * cos
        else:                                 # legacy FC-AD keys
            nf, na_, ca = js.get("norm_fc"), js.get("norm_an"), cos
            if None not in (nf, na_, ca):
                D2_wt = nf ** 2 + na_ ** 2 - 2 * nf * na_ * ca

    gamma2 = tau2 / n
    dstar2 = 2 * gamma2
    out = {"pair": f"{a}-{b}", "n": n, "tau2": tau2, "gamma2": gamma2,
           "dstar2": dstar2, "D2_uniform": D2_uni, "D2_weighted": D2_wt}
    for key, D2 in (("uniform", D2_uni), ("weighted", D2_wt)):
        if D2 is None:
            continue
        That = D2 - 4 * gamma2
        out[f"That_{key}"] = That
        out[f"delta2hat_{key}"] = D2 - 2 * gamma2
        out[f"verdict_{key}"] = "SHARE (dual)" if That < 0 else "SPLIT (single_task)"
        out[f"ratio_{key}"] = D2 / dstar2
    return out


def _fmt(o):
    lines = [f"pair {o['pair']}   n={o['n']}   tau_hat^2={o['tau2']:.3f}   "
             f"gamma_hat^2={o['gamma2']:.4f}   delta*^2=2gamma^2={o['dstar2']:.4f}"]
    for key in ("weighted", "uniform"):
        if f"That_{key}" not in o:
            continue
        lines.append(
            f"  [{key:>8} S_t] ||S_1-S_2||^2={o['D2_'+key]:.3f}  "
            f"delta_hat^2={o['delta2hat_'+key]:.3f}  "
            f"T_hat={o['That_'+key]:+.3f}  ({o['ratio_'+key]:.1f}x threshold)  "
            f"==> {o['verdict_'+key]}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["dual_c50", "dual_c100"])
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    for tag in args.tags:
        npz = os.path.join(args.results_dir, f"update_conflict_{tag}.npz")
        js = os.path.join(args.results_dir, f"update_conflict_{tag}_results.json")
        print(f"===== {tag} =====")
        print(_fmt(criterion_from_dump(npz, js)))
        print()


if __name__ == "__main__":
    main()
