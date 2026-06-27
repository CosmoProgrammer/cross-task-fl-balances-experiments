"""Aggregate multi-seed results into mean +/- std per condition.

Laptop-side, stdlib + numpy only. Scans results/ for the seed-tagged JSONs written
by run_multiseed_weekend.sh and reports, per aggregation mode, the mean and sample
std of the headline metrics across seeds -- i.e. the error bars the single-seed
work was missing. Also runs a paired (per-seed) dual-vs-single_task comparison on
anomaly AUC-ROC and forecasting R², the crux of the pivot.

Run it anytime as results stream in from the server (it just reads whatever JSONs
have landed):
    python summarize_multiseed.py              # cohort 50
    python summarize_multiseed.py --cohort 100
"""
import argparse, glob, json, os, re
import numpy as np

FED_METRICS = {           # (json_section, key) per headline metric
    'AN AUC-ROC': ('anomaly_metrics', 'auc_roc'),
    'AN AUC-PR':  ('anomaly_metrics', 'auc_pr'),
    'AN F1':      ('anomaly_metrics', 'f1'),
    'FC R2':      ('forecasting_metrics', 'r2'),
    'FC MAPE':    ('forecasting_metrics', 'mape'),
}
CEN_METRICS = {
    'AN AUC-ROC': ('centralized_anomaly', 'auc_roc'),
    'AN AUC-PR':  ('centralized_anomaly', 'auc_pr'),
    'AN F1':      ('centralized_anomaly', 'f1'),
    'FC R2':      ('centralized_forecasting', 'r2'),
    'FC MAPE':    ('centralized_forecasting', 'mape'),
}
MODES = ['local_only', 'dual', 'single_task', 'centralized']


def load(path):
    with open(path) as f:
        return json.load(f)


def collect(results_dir, ctag):
    """Return {mode: {metric: {seed: value}}} for the given cohort tag."""
    out = {m: {} for m in MODES}
    # federated
    for m in ['dual', 'single_task', 'local_only']:
        pat = os.path.join(results_dir, f"federated_{m}_fedavg{ctag}_s*_results.json")
        for p in glob.glob(pat):
            seed = int(re.search(r"_s(\d+)_results", p).group(1))
            d = load(p)
            for name, (sec, key) in FED_METRICS.items():
                v = d.get(sec, {}).get(key)
                if v is not None:
                    out[m].setdefault(name, {})[seed] = v
    # centralized (metrics nested under .../['metrics'])
    for p in glob.glob(os.path.join(results_dir, f"centralized{ctag}_s*_results.json")):
        seed = int(re.search(r"_s(\d+)_results", p).group(1))
        d = load(p)
        for name, (sec, key) in CEN_METRICS.items():
            v = d.get(sec, {}).get('metrics', {}).get(key)
            if v is not None:
                out['centralized'].setdefault(name, {})[seed] = v
    return out


def fmt(vals):
    a = np.array(vals, dtype=float)
    if len(a) == 1:
        return f"{a[0]:.4f}  (n=1)"
    return f"{a.mean():.4f} +/- {a.std(ddof=1):.4f}  (n={len(a)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', type=int, default=50)
    ap.add_argument('--results-dir', default='results')
    args = ap.parse_args()
    ctag = "" if args.cohort == 50 else f"_c{args.cohort}"

    data = collect(args.results_dir, ctag)
    metrics = list(FED_METRICS.keys())

    print(f"\n=== Multi-seed summary, cohort {args.cohort} "
          f"(mean +/- sample std across seeds) ===\n")
    hdr = f"{'mode':13s}" + "".join(f"{m:>22s}" for m in metrics)
    print(hdr); print("-" * len(hdr))
    for mode in MODES:
        row = f"{mode:13s}"
        for m in metrics:
            seeds_vals = data[mode].get(m, {})
            row += f"{(fmt(list(seeds_vals.values())) if seeds_vals else '--'):>22s}"
        print(row)

    # seeds present per mode
    print("\nseeds present:")
    for mode in MODES:
        seeds = sorted({s for mv in data[mode].values() for s in mv})
        print(f"  {mode:13s}: {seeds}")

    # paired crux comparison (only seeds where BOTH dual and single ran)
    print("\n=== Paired dual vs single_task (seeds with BOTH present) ===")
    for metric in ['AN AUC-ROC', 'AN AUC-PR', 'FC R2', 'FC MAPE']:
        dv = data['dual'].get(metric, {}); sv = data['single_task'].get(metric, {})
        common = sorted(set(dv) & set(sv))
        if not common:
            print(f"  {metric:11s}: (no paired seeds yet)"); continue
        diffs = np.array([sv[s] - dv[s] for s in common])  # single - dual
        win = "single_task" if diffs.mean() > 0 else "dual"
        better = "higher" if metric != 'FC MAPE' else "lower"
        note = "" if len(diffs) > 1 else "  (n=1, no test)"
        sd = diffs.std(ddof=1) if len(diffs) > 1 else float('nan')
        print(f"  {metric:11s}: single-dual = {diffs.mean():+.4f} +/- "
              f"{sd:.4f} over n={len(diffs)} seeds {common}  "
              f"=> {win} {better}{note}")


if __name__ == '__main__':
    main()
