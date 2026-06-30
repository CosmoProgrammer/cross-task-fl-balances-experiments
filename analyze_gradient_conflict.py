"""Backbone gradient-conflict analysis — the causal keystone for the pivot.

Measures, DIRECTLY, whether the forecasting and anomaly tasks pull the SHARED
MambaMixer backbone in conflicting directions. This upgrades the mechanism behind
the negative result (dual <= single_task) from *inferred* (over-generalization /
negative transfer read off recon-error decompositions, EXPERIMENT_RESULTS.md §6d)
to *measured*: gradient conflict at the aggregation point is the optimization-level
cause, and the cosine statistics here are the empirical anchor for any convergence
/ separation theorem ("when does cross-task federated backbone sharing help vs hurt").

WHAT IT COMPUTES (at a single SHARED backbone state w):
  For every client i, the backbone gradient g_i = ∇_{backbone} L_i(w) over that
  client's local training data (forecasting MSE or masked-reconstruction loss +
  the residual aux loss — the exact training objective). Then:
    1. HEADLINE: cos(g_FC, g_AN) where g_FC, g_AN are the FedAvg-weighted (by
       n_samples) aggregated task gradients. Negative/low => the averaged backbone
       update partially cancels each task's progress => the dual pathology.
    2. Pairwise-cosine distribution: within-FC, within-AN, and cross-task. The
       claim "averaging ACROSS tasks is the problem" is exactly within-task
       agreement >> cross-task agreement.
    3. Per-module breakdown (patch_encoders / patch_decoders / cross_scale_gates):
       which part of the backbone conflicts most.
    4. Magnitude imbalance ||g_FC||/||g_AN|| and the PCGrad conflicting component
       (one task's gradient projected onto the negative of the other) — FedAvg
       conflict comes from BOTH direction (negative cosine) and scale dominance.

WHERE TO MEASURE (the shared point w):
  The clean choice is the `dual` global backbone: dual aggregation loads an
  IDENTICAL backbone into both the forecasting and anomaly models, so both tasks
  genuinely share w and the cross-task cosine is well-defined. (single_task keeps
  two separate backbones, so there is no single shared point there — measuring its
  "conflict" would compare gradients at two different w's, which is not meaningful;
  hence the dual checkpoints are the default.) `--sync-backbone` force-copies the
  forecasting backbone into the anomaly model so the shared-point invariant holds
  even for non-dual or random-init pairs (e.g. to show conflict exists at round 0).

DESIGN CHOICES (documented so they are not silently re-litigated):
  * model.eval() during the backward pass: we want the deterministic ∇L(w), not a
    dropout-noised sample. Anomaly masking is task-intrinsic (not regularization) so
    it stays on, with a FIXED seed for reproducibility.
  * Gradients are accumulated over ALL of a client's batches (mean over the local
    set) — the local full-batch gradient direction, i.e. the round-1 linearization
    of the FedAvg pseudo-gradient. (FedAvg averages post-local-step WEIGHTS, a
    multi-step pseudo-gradient; the one-step gradient is its first-order proxy and
    the standard object for MTL conflict analysis. Noted, not hidden.)
  * No grad clipping: clipping rescales (preserves direction → cosine unaffected,
    but would distort the magnitude-ratio statistic), so raw gradients are used.

RUNS SERVER-SIDE ONLY: needs the processed data (git-ignored, server-only) and a
checkpoint. It is CHEAP — one forward+backward per client, no optimizer loop, no
multi-round training — so it can share the GPU with an ongoing sweep, and works on
CPU too (slower). Mirrors main.py's kernel setup (fp-equivalent to eager).

  conda run -n yuv_fl python analyze_gradient_conflict.py                 # dual c50 backbone
  conda run -n yuv_fl python analyze_gradient_conflict.py --cohort 100 \
      --fc-ckpt checkpoints/fed_dual_fedavg_forecasting_model.pt \
      --an-ckpt checkpoints/fed_dual_fedavg_anomaly_model.pt --tag dual_c100

Outputs: a summary to the log, results/grad_conflict_<tag>_results.json (all
scalars), and results/grad_conflict_<tag>.npz (the full pairwise cosine matrix +
per-client backbone-grad norms + the FC/AN client-id split) — the re-derivable
artifact for laptop-side plotting, mirroring the anomaly-score npz convention.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import models.mamba_mixer as mamba_mixer
from main import (build_forecasting_model, build_anomaly_model, build_aux_loss,
                  get_device, save_results)
from data_provider.ashrae_dataset import get_ashrae_fl_data
from data_provider.lead_dataset import get_lead_fl_data
from trainers.multitask_fed_trainer import _is_backbone, BACKBONE_PREFIXES
from utils.logging_utils import setup_logger
from configs.config import ExperimentConfig


def _backbone_param_names(model):
    """Backbone parameter names in a FIXED (sorted) order for stable flattening."""
    return sorted(n for n, p in model.named_parameters()
                  if _is_backbone(n) and p.requires_grad)


def _module_of(name):
    """Top-level backbone module a parameter belongs to (for the breakdown)."""
    for pref in BACKBONE_PREFIXES:
        if name.startswith(pref):
            return pref.rstrip(".")
    return "other"


def _load(model, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state)  # strict: dims must match config defaults
    return model


def _forecasting_grad(model, loader, aux_fn, device, names, max_batches=0):
    """Mean local backbone gradient for one forecasting client at w (eval mode).

    Returns a flat float64 numpy vector over `names`, or None if the client has no
    usable batches.
    """
    model.eval()                       # deterministic ∇L(w): no dropout noise
    model.zero_grad(set_to_none=True)
    loss_fn = nn.MSELoss()
    n = 0
    for x, y in loader:
        x = x.float().to(device)
        y = y.float().to(device)
        loss = loss_fn(model(x), y) + aux_fn(model)
        loss.backward()                # grads accumulate across batches
        n += 1
        if max_batches and n >= max_batches:
            break
    if n == 0:
        return None
    return _collect_grad(model, names, scale=1.0 / n)


def _anomaly_grad(model, loader, aux_fn, device, names, mask_rate,
                  mask_seed=0, max_batches=0):
    """Mean local backbone gradient for one anomaly client at w (eval mode).

    Masking is task-intrinsic, so it stays on but is seeded for reproducibility.
    Loss = masked-position reconstruction MSE + aux (matches the trainer exactly).
    """
    model.eval()
    model.zero_grad(set_to_none=True)
    loss_fn = nn.MSELoss(reduction="none")
    g = torch.Generator(device=device)
    g.manual_seed(mask_seed)
    n = 0
    for sequences, _labels in loader:
        sequences = sequences.float().to(device)
        if sequences.dim() == 2:
            sequences = sequences.unsqueeze(-1)
        mask = (torch.rand(sequences.shape, generator=g, device=device)
                > mask_rate).float()
        x_masked = sequences * mask
        recon = model(x_masked, x_mask=mask)
        per_elem = loss_fn(recon, sequences)
        masked_pos = (mask == 0)
        loss = (per_elem[masked_pos].mean() if masked_pos.any()
                else per_elem.mean())
        loss = loss + aux_fn(model)
        loss.backward()
        n += 1
        if max_batches and n >= max_batches:
            break
    if n == 0:
        return None
    return _collect_grad(model, names, scale=1.0 / n)


def _collect_grad(model, names, scale=1.0):
    """Flatten the accumulated .grad over `names` into one float64 vector."""
    params = dict(model.named_parameters())
    chunks = []
    for nm in names:
        gr = params[nm].grad
        if gr is None:
            chunks.append(np.zeros(params[nm].numel(), dtype=np.float64))
        else:
            chunks.append((gr.detach().double() * scale).flatten().cpu().numpy())
    return np.concatenate(chunks)


def _cos(a, b, eps=1e-12):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _module_masks(model, names):
    """Boolean mask over the flat backbone vector for each top-level module."""
    params = dict(model.named_parameters())
    mod_per_scalar = np.concatenate([
        np.full(params[nm].numel(), _module_of(nm), dtype=object) for nm in names])
    return {mod: (mod_per_scalar == mod) for mod in set(mod_per_scalar)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc-ckpt",
                    default="checkpoints/fed_dual_fedavg_forecasting_model.pt")
    ap.add_argument("--an-ckpt",
                    default="checkpoints/fed_dual_fedavg_anomaly_model.pt")
    ap.add_argument("--tag", default="dual_c50",
                    help="label for outputs (results/grad_conflict_<tag>_*).")
    ap.add_argument("--cohort", type=int, default=ExperimentConfig.cohort_size,
                    help="Cohort of the checkpoints (drives the processed-data dir).")
    ap.add_argument("--max-batches", type=int, default=0,
                    help="Cap batches per client (0 = all = true local-set gradient).")
    ap.add_argument("--random-init", action="store_true",
                    help="Skip loading checkpoints: measure conflict at random init "
                         "(a round-0 baseline; shows conflict is not a converged "
                         "artifact). Implies --sync-backbone.")
    ap.add_argument("--sync-backbone", action="store_true",
                    help="Copy the forecasting backbone into the anomaly model so "
                         "both share an IDENTICAL w (the shared-point invariant). "
                         "Auto-on for dual checkpoints anyway; required for non-dual "
                         "or random-init pairs.")
    args = ap.parse_args()

    config = ExperimentConfig(cohort_size=args.cohort)
    device = get_device(config)
    log = setup_logger("grad_conflict", config.log_dir)

    mamba_mixer.USE_KERNELS = config.use_mamba_kernel
    kernels_active = (config.use_mamba_kernel
                      and mamba_mixer._HAS_SELECTIVE_SCAN
                      and device.type == "cuda")
    log.info("Device: %s | Mamba fused kernels: %s | cohort: %d", device,
             "ON" if kernels_active else "OFF (eager)", config.cohort_size)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ── Models + the shared backbone state w ──
    fc_model = build_forecasting_model(config).to(device)
    an_model = build_anomaly_model(config).to(device)
    if not args.random_init:
        _load(fc_model, args.fc_ckpt, device)
        _load(an_model, args.an_ckpt, device)
        log.info("Loaded fc=%s an=%s", args.fc_ckpt, args.an_ckpt)
    else:
        log.info("RANDOM INIT (round-0 baseline) — no checkpoints loaded")

    names = _backbone_param_names(fc_model)
    n_backbone = sum(dict(fc_model.named_parameters())[n].numel() for n in names)
    log.info("Backbone: %d parameter tensors, %d scalars", len(names), n_backbone)

    # Enforce the shared-point invariant: both models must hold the SAME backbone.
    fc_params, an_params = dict(fc_model.named_parameters()), dict(an_model.named_parameters())
    max_diff = max(float((fc_params[n].detach() - an_params[n].detach()).abs().max())
                   for n in names)
    if args.random_init or args.sync_backbone or max_diff > 1e-6:
        if max_diff > 1e-6 and not (args.random_init or args.sync_backbone):
            log.warning("fc/an backbones differ (max |Δ|=%.2e) — NOT a shared point. "
                        "Syncing an-backbone <- fc-backbone so the cosine is "
                        "well-defined. (Expected ~0 for dual checkpoints.)", max_diff)
        with torch.no_grad():
            for n in names:
                an_params[n].copy_(fc_params[n])
        max_diff = 0.0
    log.info("Shared-point check: max |fc_backbone - an_backbone| = %.2e", max_diff)

    aux_fn = build_aux_loss(config)

    # ── Data: per-client TRAIN loaders (this is what differs from eval_from_checkpoint) ──
    fc_clients, _ = get_ashrae_fl_data(
        config.ashrae_processed_dir, config.seq_len, config.pred_len,
        config.batch_size, num_workers=config.num_workers)
    an_clients, _, _ = get_lead_fl_data(
        config.lead_processed_dir, config.seq_len, config.batch_size,
        clean_only=config.clean_only, num_workers=config.num_workers)
    fc_ids = [c for c in fc_clients if fc_clients[c]["train"] is not None]
    an_ids = [c for c in an_clients if an_clients[c]["train"] is not None]
    log.info("Clients: %d forecasting + %d anomaly", len(fc_ids), len(an_ids))

    # ── Per-client backbone gradients at w ──
    grads, weights, tasks, cids = [], [], [], []

    log.info("--- Forecasting client gradients ---")
    for cid in fc_ids:
        g = _forecasting_grad(fc_model, fc_clients[cid]["train"], aux_fn, device,
                              names, max_batches=args.max_batches)
        if g is None:
            continue
        grads.append(g); weights.append(fc_clients[cid]["n_samples"])
        tasks.append("FC"); cids.append(int(cid) if str(cid).isdigit() else cid)

    log.info("--- Anomaly client gradients ---")
    for cid in an_ids:
        g = _anomaly_grad(an_model, an_clients[cid]["train"], aux_fn, device,
                          names, config.mask_rate, mask_seed=config.seed,
                          max_batches=args.max_batches)
        if g is None:
            continue
        grads.append(g); weights.append(an_clients[cid]["n_samples"])
        tasks.append("AN"); cids.append(int(cid) if str(cid).isdigit() else cid)

    grads = np.stack(grads)                       # (n_clients, n_backbone)
    weights = np.asarray(weights, dtype=np.float64)
    tasks = np.asarray(tasks)
    fc_mask = tasks == "FC"
    an_mask = tasks == "AN"
    norms = np.linalg.norm(grads, axis=1)

    # ── 1. HEADLINE: FedAvg-weighted aggregated task gradients ──
    def _agg(mask):
        w = weights[mask]
        return (grads[mask] * (w / w.sum())[:, None]).sum(axis=0)
    g_fc, g_an = _agg(fc_mask), _agg(an_mask)
    cos_agg = _cos(g_fc, g_an)
    norm_fc, norm_an = float(np.linalg.norm(g_fc)), float(np.linalg.norm(g_an))
    mag_ratio = norm_fc / norm_an if norm_an > 0 else float("inf")
    # PCGrad conflicting component: how much of g_AN opposes g_FC (and vice-versa).
    conflict_an_on_fc = max(0.0, -np.dot(g_an, g_fc)) / (norm_fc + 1e-12)
    conflict_fc_on_an = max(0.0, -np.dot(g_fc, g_an)) / (norm_an + 1e-12)

    # ── 2. Pairwise cosine distribution ──
    unit = grads / (norms[:, None] + 1e-12)
    cosmat = unit @ unit.T                         # (n, n)
    iu = np.triu_indices(len(grads), k=1)
    pair_task = np.array([f"{a}-{b}" for a, b in zip(tasks[iu[0]], tasks[iu[1]])])
    pair_cos = cosmat[iu]
    within_fc = pair_cos[(pair_task == "FC-FC")]
    within_an = pair_cos[(pair_task == "AN-AN")]
    cross = pair_cos[(pair_task == "FC-AN") | (pair_task == "AN-FC")]
    frac_cross_neg = float((cross < 0).mean()) if cross.size else float("nan")

    # ── 3. Per-module breakdown (aggregated task gradients) ──
    masks = _module_masks(fc_model, names)
    per_module = {}
    for mod, m in masks.items():
        per_module[mod] = {
            "cos_agg": _cos(g_fc[m], g_an[m]),
            "norm_fc": float(np.linalg.norm(g_fc[m])),
            "norm_an": float(np.linalg.norm(g_an[m])),
            "n_scalars": int(m.sum()),
        }

    # ── Report ──
    def _stats(a):
        return (dict(mean=float(a.mean()), std=float(a.std()),
                     median=float(np.median(a)), n=int(a.size))
                if a.size else dict(mean=float("nan"), std=float("nan"),
                                    median=float("nan"), n=0))
    log.info("=" * 64)
    log.info("HEADLINE  cos(g_FC, g_AN) on shared backbone = %.4f", cos_agg)
    log.info("  ||g_FC||=%.4e  ||g_AN||=%.4e  ratio FC/AN=%.3f", norm_fc, norm_an, mag_ratio)
    log.info("  PCGrad conflict: AN-on-FC=%.4e  FC-on-AN=%.4e",
             conflict_an_on_fc, conflict_fc_on_an)
    log.info("Pairwise cosine  within-FC: mean %.3f (n=%d) | within-AN: mean %.3f (n=%d) "
             "| CROSS: mean %.3f (n=%d), frac<0 %.3f",
             _stats(within_fc)["mean"], within_fc.size,
             _stats(within_an)["mean"], within_an.size,
             _stats(cross)["mean"], cross.size, frac_cross_neg)
    log.info("Per-module cos(g_FC,g_AN):")
    for mod, d in per_module.items():
        log.info("  %-20s cos=%+.4f  ||FC||=%.3e ||AN||=%.3e (%d scalars)",
                 mod, d["cos_agg"], d["norm_fc"], d["norm_an"], d["n_scalars"])
    log.info("=" * 64)

    # ── Persist ──
    results = {
        "experiment": "gradient_conflict",
        "tag": args.tag,
        "cohort": config.cohort_size,
        "checkpoints": {"forecasting": args.fc_ckpt, "anomaly": args.an_ckpt,
                        "random_init": bool(args.random_init)},
        "n_clients": {"forecasting": int(fc_mask.sum()), "anomaly": int(an_mask.sum())},
        "headline_cos_agg": cos_agg,
        "norm_fc": norm_fc, "norm_an": norm_an, "mag_ratio_fc_over_an": mag_ratio,
        "pcgrad_conflict_an_on_fc": conflict_an_on_fc,
        "pcgrad_conflict_fc_on_an": conflict_fc_on_an,
        "pairwise": {"within_fc": _stats(within_fc), "within_an": _stats(within_an),
                     "cross": _stats(cross), "frac_cross_negative": frac_cross_neg},
        "per_module": per_module,
    }
    out_json = os.path.join(config.results_dir,
                            f"grad_conflict_{args.tag}_results.json")
    save_results(results, out_json, log)

    out_npz = os.path.join(config.results_dir, f"grad_conflict_{args.tag}.npz")
    np.savez_compressed(
        out_npz,
        cos_matrix=cosmat.astype(np.float32),
        tasks=tasks, client_ids=np.asarray(cids, dtype=object),
        grad_norms=norms, weights=weights,
        g_fc=g_fc.astype(np.float32), g_an=g_an.astype(np.float32),
        cohort=np.int64(config.cohort_size),
    )
    log.info("Saved pairwise/grad artifact -> %s", out_npz)


if __name__ == "__main__":
    main()
