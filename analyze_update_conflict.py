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

Computes the same statistics as analyze_gradient_conflict.py but on deltas:
  headline cos(Δ_FC, Δ_AN), ||Δ_FC||/||Δ_AN|| (THE number that settles the
  confound), within- vs cross-task pairwise cosine, per-module breakdown.

Runs at the shared dual backbone (default checkpoints). SERVER-SIDE ONLY (needs
data). COST ~ one un-aggregated FL round (real local training per client, no
optimizer-free shortcut) -> ~20 min at c50 contended; heavier than the gradient
script but still cheap vs a full run. Mirrors main.py's kernel setup.

  conda run -n yuv_fl python analyze_update_conflict.py                  # dual c50
  conda run -n yuv_fl python analyze_update_conflict.py --cohort 100 \
      --fc-ckpt checkpoints/fed_dual_fedavg_c100_forecasting_model.pt \
      --an-ckpt checkpoints/fed_dual_fedavg_c100_anomaly_model.pt --tag dual_c100

Outputs results/update_conflict_<tag>_results.json + .npz (pairwise matrix +
per-client delta norms), the laptop-side plotting artifact.
"""
import argparse
import copy
import os

import numpy as np
import torch

import models.mamba_mixer as mamba_mixer
from main import (build_forecasting_model, build_anomaly_model, build_aux_loss,
                  get_device, save_results)
from data_provider.ashrae_dataset import get_ashrae_fl_data
from data_provider.lead_dataset import get_lead_fl_data
from trainers.multitask_fed_trainer import MultiTaskFederatedTrainer
from utils.logging_utils import setup_logger
from configs.config import ExperimentConfig
# pure helpers shared with the gradient-conflict script (no behavioural coupling)
from analyze_gradient_conflict import (_backbone_param_names, _cos, _module_masks,
                                       _load)


def _flat_backbone(model, names):
    """Flatten backbone parameter VALUES (not grads) into one float64 vector."""
    params = dict(model.named_parameters())
    return np.concatenate([params[n].detach().double().flatten().cpu().numpy()
                           for n in names])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc-ckpt",
                    default="checkpoints/fed_dual_fedavg_forecasting_model.pt")
    ap.add_argument("--an-ckpt",
                    default="checkpoints/fed_dual_fedavg_anomaly_model.pt")
    ap.add_argument("--tag", default="dual_c50")
    ap.add_argument("--cohort", type=int, default=ExperimentConfig.cohort_size)
    args = ap.parse_args()

    config = ExperimentConfig(cohort_size=args.cohort)
    device = get_device(config)
    log = setup_logger("update_conflict", config.log_dir)

    mamba_mixer.USE_KERNELS = config.use_mamba_kernel
    kernels_active = (config.use_mamba_kernel and mamba_mixer._HAS_SELECTIVE_SCAN
                      and device.type == "cuda")
    log.info("Device: %s | kernels: %s | cohort: %d", device,
             "ON" if kernels_active else "OFF", config.cohort_size)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    fc_model = build_forecasting_model(config).to(device)
    an_model = build_anomaly_model(config).to(device)
    _load(fc_model, args.fc_ckpt, device)
    _load(an_model, args.an_ckpt, device)
    log.info("Loaded fc=%s an=%s", args.fc_ckpt, args.an_ckpt)

    names = _backbone_param_names(fc_model)
    # The trainer supplies the EXACT local-training loops (AdamW, grad-clip, mask,
    # aux). FedProx is off (fedavg) so global_params=None -> plain local training.
    trainer = MultiTaskFederatedTrainer(fc_model, an_model, device, config,
                                        aux_loss_fn=build_aux_loss(config), logger=log)

    fc_clients, _ = get_ashrae_fl_data(
        config.ashrae_processed_dir, config.seq_len, config.pred_len,
        config.batch_size, num_workers=config.num_workers)
    an_clients, _, _ = get_lead_fl_data(
        config.lead_processed_dir, config.seq_len, config.batch_size,
        clean_only=config.clean_only, num_workers=config.num_workers)
    fc_ids = [c for c in fc_clients if fc_clients[c]["train"] is not None]
    an_ids = [c for c in an_clients if an_clients[c]["train"] is not None]
    log.info("Clients: %d forecasting + %d anomaly", len(fc_ids), len(an_ids))

    deltas, weights, tasks = [], [], []

    def _client_delta(global_model, loader, train_fn):
        client = copy.deepcopy(global_model).to(device)
        before = _flat_backbone(client, names)
        train_fn(client, loader, device)        # real local AdamW loop
        after = _flat_backbone(client, names)
        return after - before

    log.info("--- Forecasting client deltas (local AdamW) ---")
    for cid in fc_ids:
        deltas.append(_client_delta(fc_model, fc_clients[cid]["train"],
                                    trainer._train_forecasting_client_on))
        weights.append(fc_clients[cid]["n_samples"]); tasks.append("FC")

    log.info("--- Anomaly client deltas (local AdamW) ---")
    for cid in an_ids:
        deltas.append(_client_delta(an_model, an_clients[cid]["train"],
                                    trainer._train_anomaly_client_on))
        weights.append(an_clients[cid]["n_samples"]); tasks.append("AN")

    deltas = np.stack(deltas)
    weights = np.asarray(weights, dtype=np.float64)
    tasks = np.asarray(tasks)
    fc_mask, an_mask = tasks == "FC", tasks == "AN"
    norms = np.linalg.norm(deltas, axis=1)

    def _agg(mask):
        w = weights[mask]
        return (deltas[mask] * (w / w.sum())[:, None]).sum(axis=0)
    d_fc, d_an = _agg(fc_mask), _agg(an_mask)
    cos_agg = _cos(d_fc, d_an)
    norm_fc, norm_an = float(np.linalg.norm(d_fc)), float(np.linalg.norm(d_an))
    mag_ratio = norm_fc / norm_an if norm_an > 0 else float("inf")

    unit = deltas / (norms[:, None] + 1e-12)
    cosmat = unit @ unit.T
    iu = np.triu_indices(len(deltas), k=1)
    pt = np.array([f"{a}-{b}" for a, b in zip(tasks[iu[0]], tasks[iu[1]])])
    pc = cosmat[iu]
    within_fc = pc[pt == "FC-FC"]; within_an = pc[pt == "AN-AN"]
    cross = pc[(pt == "FC-AN") | (pt == "AN-FC")]
    frac_cross_neg = float((cross < 0).mean()) if cross.size else float("nan")

    masks = _module_masks(fc_model, names)
    per_module = {mod: {"cos_agg": _cos(d_fc[m], d_an[m]),
                        "norm_fc": float(np.linalg.norm(d_fc[m])),
                        "norm_an": float(np.linalg.norm(d_an[m])),
                        "n_scalars": int(m.sum())}
                  for mod, m in masks.items()}

    def _mean(a): return float(a.mean()) if a.size else float("nan")
    log.info("=" * 64)
    log.info("UPDATE (AdamW delta)  cos(Δ_FC, Δ_AN) = %.4f", cos_agg)
    log.info("  ||Δ_FC||=%.4e  ||Δ_AN||=%.4e  ratio FC/AN=%.3f  <-- the confound test",
             norm_fc, norm_an, mag_ratio)
    log.info("  pairwise within-FC %.3f | within-AN %.3f | CROSS %.3f (frac<0 %.3f)",
             _mean(within_fc), _mean(within_an), _mean(cross), frac_cross_neg)
    for mod, d in per_module.items():
        log.info("  %-20s cos=%+.4f ||FC||=%.3e ||AN||=%.3e", mod, d["cos_agg"],
                 d["norm_fc"], d["norm_an"])
    log.info("=" * 64)

    results = {
        "experiment": "update_conflict_adamw_delta", "tag": args.tag,
        "cohort": config.cohort_size,
        "checkpoints": {"forecasting": args.fc_ckpt, "anomaly": args.an_ckpt},
        "n_clients": {"forecasting": int(fc_mask.sum()), "anomaly": int(an_mask.sum())},
        "headline_cos_agg": cos_agg, "norm_fc": norm_fc, "norm_an": norm_an,
        "mag_ratio_fc_over_an": mag_ratio,
        "pairwise": {"within_fc_mean": _mean(within_fc),
                     "within_an_mean": _mean(within_an),
                     "cross_mean": _mean(cross),
                     "frac_cross_negative": frac_cross_neg},
        "per_module": per_module,
    }
    save_results(results, os.path.join(config.results_dir,
                 f"update_conflict_{args.tag}_results.json"), log)
    np.savez_compressed(
        os.path.join(config.results_dir, f"update_conflict_{args.tag}.npz"),
        cos_matrix=cosmat.astype(np.float32), tasks=tasks,
        delta_norms=norms, weights=weights,
        d_fc=d_fc.astype(np.float32), d_an=d_an.astype(np.float32),
        cohort=np.int64(config.cohort_size))
    log.info("Saved -> results/update_conflict_%s.npz", args.tag)


if __name__ == "__main__":
    main()
