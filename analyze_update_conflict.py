"""Backbone UPDATE-conflict analysis — the faithful FedAvg pseudo-gradient.

Companion to `analyze_gradient_conflict.py`. That script measures the one-step
RAW gradient ∇L(w); this one measures the actual per-client backbone DELTA
(w_after_local_training - w_before) after running the real local AdamW loop
(local_epochs, lr, grad-clip, masking, aux loss) — i.e. exactly the object
FedAvg averages into the shared backbone.

WHY THIS EXISTS: the raw-gradient run found a ~15x forecasting>anomaly magnitude
imbalance, but AdamW rescales per-coordinate over multiple local steps, which can
erase a raw-magnitude gap. The "forecasting drowns anomaly in the average" claim
hinges on whether that imbalance SURVIVES in the AdamW deltas. This script answers
that directly. (The cosine/orthogonality finding is scale-invariant and already
robust; this is specifically to settle the MAGNITUDE confound.)

ARBITRARY TASK PAIRS (Track 3.5): `--tasks a,b` picks any two tasks from
{fc, an, imp}, so the debiased Theorem-3 statistic T̂ can be computed for the new
imputation pairs (AN–IMP, FC–IMP) exactly as it was for the FC–AD anchor. The
per-client deltas + full pairwise Gram matrix dumped here are what
`compute_affinity_criterion.py` consumes. The FC–AD default is unchanged.

Computes, for the chosen pair (a,b): headline cos(Δ_a, Δ_b), ||Δ_a||/||Δ_b||
(THE magnitude number), within- vs cross-task pairwise cosine, per-module
breakdown, and the per-task WEIGHTED aggregate delta (the S_t of Theorem 3).

Runs at a shared dual backbone (default = the committed FC–AD checkpoints).
SERVER-SIDE ONLY (needs data). COST ~ one un-aggregated FL round (real local
training per client) -> ~20 min at c50 contended. Mirrors main.py's kernel setup.

  # FC–AD anchor (unchanged default):
  conda run -n yuv_fl python analyze_update_conflict.py
  conda run -n yuv_fl python analyze_update_conflict.py --cohort 100 \
      --fc-ckpt checkpoints/fed_dual_fedavg_c100_forecasting_model.pt \
      --an-ckpt checkpoints/fed_dual_fedavg_c100_anomaly_model.pt --tag dual_c100

  # New imputation pairs (from the dual_tasks-*  run checkpoints):
  conda run -n yuv_fl python analyze_update_conflict.py --tasks an,imp \
      --ckpts checkpoints/fed_dual_fedavg_tasks-an-imp_anomaly_model.pt,\
checkpoints/fed_dual_fedavg_tasks-an-imp_imputation_model.pt
  conda run -n yuv_fl python analyze_update_conflict.py --tasks fc,imp \
      --ckpts checkpoints/fed_dual_fedavg_tasks-fc-imp_forecasting_model.pt,\
checkpoints/fed_dual_fedavg_tasks-fc-imp_imputation_model.pt

Outputs results/update_conflict_<tag>_results.json + .npz (pairwise matrix,
per-client delta norms, per-task aggregate deltas), the laptop-side artifact for
`compute_affinity_criterion.py`.

NOTE the committed `update_conflict_dual_c{50,100}` dumps are the FROZEN
pre-registration reference (PREREGISTERED_PREDICTIONS.md); re-running here may
differ in the last digits due to batch-order RNG — do not regenerate them.
"""
import argparse
import copy
import os
import time

import numpy as np
import torch

import models.mamba_mixer as mamba_mixer
from main import (build_aux_loss, get_device, save_results,
                  _build_task_spec, _TASK_META)
from trainers.multitask_fed_trainer import MultiTaskFederatedTrainer
from utils.logging_utils import setup_logger
from configs.config import ExperimentConfig
# pure helpers shared with the gradient-conflict script (no behavioural coupling)
from analyze_gradient_conflict import (_backbone_param_names, _cos, _module_masks,
                                       _load)

# Legacy per-task checkpoint defaults (the FC–AD anchor invocation). imp has no
# canonical default — its checkpoints come from a specific dual_tasks-* run.
_LEGACY_CKPT = {
    "fc": "checkpoints/fed_dual_fedavg_forecasting_model.pt",
    "an": "checkpoints/fed_dual_fedavg_anomaly_model.pt",
    "imp": None,
}


def _flat_backbone(model, names):
    """Flatten backbone parameter VALUES (not grads) into one float64 vector."""
    params = dict(model.named_parameters())
    return np.concatenate([params[n].detach().double().flatten().cpu().numpy()
                           for n in names])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="fc,an",
                    help="exactly two task tokens from {fc,an,imp} (default fc,an)")
    ap.add_argument("--ckpts", default=None,
                    help="comma list of checkpoint paths, order matching --tasks; "
                         "overrides the per-task --*-ckpt defaults")
    # legacy per-task checkpoint flags (keep the FC–AD default invocation working)
    ap.add_argument("--fc-ckpt", default=_LEGACY_CKPT["fc"])
    ap.add_argument("--an-ckpt", default=_LEGACY_CKPT["an"])
    ap.add_argument("--imp-ckpt", default=_LEGACY_CKPT["imp"])
    ap.add_argument("--tag", default=None,
                    help="output tag (default: dual_c{N} for fc,an; else "
                         "'{a}-{b}_c{N}')")
    ap.add_argument("--cohort", type=int, default=ExperimentConfig.cohort_size)
    args = ap.parse_args()

    tokens = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(tokens) != 2:
        ap.error(f"--tasks needs exactly two tokens (got {tokens})")
    bad = [t for t in tokens if t not in _TASK_META]
    if bad:
        ap.error(f"--tasks: unknown {bad}; choose from {list(_TASK_META)}")

    # resolve one checkpoint per task
    if args.ckpts:
        paths = [p.strip() for p in args.ckpts.split(",") if p.strip()]
        if len(paths) != len(tokens):
            ap.error(f"--ckpts count ({len(paths)}) must match --tasks ({len(tokens)})")
        ckpt = dict(zip(tokens, paths))
    else:
        legacy = {"fc": args.fc_ckpt, "an": args.an_ckpt, "imp": args.imp_ckpt}
        ckpt = {t: legacy[t] for t in tokens}
    missing = [t for t in tokens if not ckpt[t]]
    if missing:
        ap.error(f"no checkpoint for task(s) {missing}; pass --ckpts "
                 f"(or --{missing[0]}-ckpt)")

    if args.tag:
        tag = args.tag
    elif tokens == ["fc", "an"]:
        tag = f"dual_c{args.cohort}"          # committed-anchor naming
    else:
        tag = "-".join(tokens) + f"_c{args.cohort}"

    config = ExperimentConfig(cohort_size=args.cohort)
    device = get_device(config)
    log = setup_logger("update_conflict", config.log_dir)

    mamba_mixer.USE_KERNELS = config.use_mamba_kernel
    kernels_active = (config.use_mamba_kernel and mamba_mixer._HAS_SELECTIVE_SCAN
                      and device.type == "cuda")
    log.info("Device: %s | kernels: %s | cohort: %d | tasks: %s", device,
             "ON" if kernels_active else "OFF", config.cohort_size, tokens)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Build each task's global model + FL loaders via main.py's registry, then
    # load the shared-backbone checkpoint. `_build_task_spec` builds the model
    # BEFORE the loaders (same convention as the training path).
    specs = [_build_task_spec(tok, config, log) for tok in tokens]
    for tok, spec in zip(tokens, specs):
        spec.model = spec.model.to(device)
        _load(spec.model, ckpt[tok], device)
        log.info("Loaded %-12s <- %s", spec.name, ckpt[tok])

    names = _backbone_param_names(specs[0].model)
    # The trainer supplies the EXACT local-training loops (AdamW, grad-clip, mask,
    # aux) via `_train_fn_for(kind)`. FedProx off (fedavg) -> plain local training.
    # from_tasks() also asserts the two backbones are shape-compatible.
    trainer = MultiTaskFederatedTrainer.from_tasks(
        specs, device, config, aux_loss_fn=build_aux_loss(config), logger=log)

    labels = [tok.upper() for tok in tokens]
    la, lb = labels
    n_clients = {spec.name: len([c for c in spec.client_loaders
                                 if spec.client_loaders[c]["train"] is not None])
                 for spec in specs}
    log.info("Clients: %s", " + ".join(f"{v} {k}" for k, v in n_clients.items()))

    # Re-seed right before the delta loop so batch-order RNG is deterministic for
    # THIS invocation regardless of how many models were built above.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    deltas, weights, task_lbls = [], [], []

    def _client_delta(global_model, loader, train_fn):
        client = copy.deepcopy(global_model).to(device)
        before = _flat_backbone(client, names)
        train_fn(client, loader, device)        # real local AdamW loop
        after = _flat_backbone(client, names)
        return after - before

    for spec, lbl in zip(specs, labels):
        train_fn = trainer._train_fn_for(spec.kind)
        ids = [c for c in spec.client_loaders
               if spec.client_loaders[c]["train"] is not None]
        log.info("--- %s client deltas (local AdamW), %d clients ---",
                 spec.name, len(ids))
        t0 = time.time()
        for i, cid in enumerate(ids, 1):
            deltas.append(_client_delta(spec.model, spec.client_loaders[cid]["train"],
                                        train_fn))
            weights.append(spec.client_loaders[cid]["n_samples"])
            task_lbls.append(lbl)
            if i % 5 == 0 or i == len(ids):
                log.info("  %s %d/%d done (%.0fs)", lbl, i, len(ids), time.time() - t0)

    deltas = np.stack(deltas)
    weights = np.asarray(weights, dtype=np.float64)
    task_lbls = np.asarray(task_lbls)
    ma, mb = task_lbls == la, task_lbls == lb
    norms = np.linalg.norm(deltas, axis=1)

    def _agg(mask):
        w = weights[mask]
        return (deltas[mask] * (w / w.sum())[:, None]).sum(axis=0)
    d_a, d_b = _agg(ma), _agg(mb)
    cos_agg = _cos(d_a, d_b)
    norm_a, norm_b = float(np.linalg.norm(d_a)), float(np.linalg.norm(d_b))
    mag_ratio = norm_a / norm_b if norm_b > 0 else float("inf")

    unit = deltas / (norms[:, None] + 1e-12)
    cosmat = unit @ unit.T
    iu = np.triu_indices(len(deltas), k=1)
    pt = np.array([f"{a}-{b}" for a, b in zip(task_lbls[iu[0]], task_lbls[iu[1]])])
    pc = cosmat[iu]
    within_a = pc[pt == f"{la}-{la}"]
    within_b = pc[pt == f"{lb}-{lb}"]
    cross = pc[(pt == f"{la}-{lb}") | (pt == f"{lb}-{la}")]
    frac_cross_neg = float((cross < 0).mean()) if cross.size else float("nan")

    masks = _module_masks(specs[0].model, names)
    per_module = {mod: {"cos_agg": _cos(d_a[m], d_b[m]),
                        f"norm_{la.lower()}": float(np.linalg.norm(d_a[m])),
                        f"norm_{lb.lower()}": float(np.linalg.norm(d_b[m])),
                        "n_scalars": int(m.sum())}
                  for mod, m in masks.items()}

    def _mean(a): return float(a.mean()) if a.size else float("nan")
    log.info("=" * 64)
    log.info("UPDATE (AdamW delta)  cos(Δ_%s, Δ_%s) = %.4f", la, lb, cos_agg)
    log.info("  ||Δ_%s||=%.4e  ||Δ_%s||=%.4e  ratio %s/%s=%.3f  <-- confound test",
             la, norm_a, lb, norm_b, la, lb, mag_ratio)
    log.info("  pairwise within-%s %.3f | within-%s %.3f | CROSS %.3f (frac<0 %.3f)",
             la, _mean(within_a), lb, _mean(within_b), _mean(cross), frac_cross_neg)
    for mod, d in per_module.items():
        log.info("  %-20s cos=%+.4f ||%s||=%.3e ||%s||=%.3e", mod, d["cos_agg"],
                 la, d[f"norm_{la.lower()}"], lb, d[f"norm_{lb.lower()}"])
    log.info("=" * 64)

    results = {
        "experiment": "update_conflict_adamw_delta", "tag": tag,
        "tasks": tokens, "labels": labels,
        "cohort": config.cohort_size,
        "checkpoints": {tok: ckpt[tok] for tok in tokens},
        "n_clients": {la: int(ma.sum()), lb: int(mb.sum())},
        "headline_cos_agg": cos_agg,
        "agg_norms": {la: norm_a, lb: norm_b},
        "mag_ratio": mag_ratio,
        "pairwise": {f"within_{la.lower()}_mean": _mean(within_a),
                     f"within_{lb.lower()}_mean": _mean(within_b),
                     "cross_mean": _mean(cross),
                     "frac_cross_negative": frac_cross_neg},
        "per_module": per_module,
    }
    # Legacy key aliases so committed FC–AD tooling keeps reading the same names.
    if set(labels) == {"FC", "AN"}:
        results["norm_fc"] = norm_a if la == "FC" else norm_b
        results["norm_an"] = norm_a if la == "AN" else norm_b
        results["mag_ratio_fc_over_an"] = results["norm_fc"] / results["norm_an"]

    save_results(results, os.path.join(config.results_dir,
                 f"update_conflict_{tag}_results.json"), log)

    npz = {
        "cos_matrix": cosmat.astype(np.float32),
        "tasks": task_lbls,
        "delta_norms": norms,
        "weights": weights,
        f"agg__{la}": d_a.astype(np.float32),
        f"agg__{lb}": d_b.astype(np.float32),
        "cohort": np.int64(config.cohort_size),
    }
    # Legacy aggregate-delta aliases for the FC–AD pair.
    if set(labels) == {"FC", "AN"}:
        npz["d_fc"] = (d_a if la == "FC" else d_b).astype(np.float32)
        npz["d_an"] = (d_a if la == "AN" else d_b).astype(np.float32)
    np.savez_compressed(
        os.path.join(config.results_dir, f"update_conflict_{tag}.npz"), **npz)
    log.info("Saved -> results/update_conflict_%s.npz", tag)


if __name__ == "__main__":
    main()
