"""
Multi-Task Federated Learning Experiment: MambaMixer

Experiment:
  - 50 ASHRAE buildings (35 train + 15 test) -> forecasting
  - 50 LEAD buildings (35 train + 15 test) -> anomaly detection
  - Cross-task backbone aggregation via FedAvg across all 70 clients
  - Task heads aggregated within each task group

Runs:
  1. Preprocessing (if needed)
  2. Multi-task FL training (configurable mode + strategy)
  3. Centralized baselines (same data splits)
  4. SOTA baselines (LSTM, Informer, MSD-Mixer, LSTM-AE, ANN-AE)
  5. Evaluation and comparison
  6. Visualization (research-paper plots)

Usage:
    python main.py                                    # run FL (dual/fedavg) + centralized + plots
    python main.py --preprocess                       # preprocessing only
    python main.py --federated                        # FL training only (default: dual/fedavg)
    python main.py --federated --mode single_task     # single-task FL
    python main.py --federated --mode local_only      # local-only baseline
    python main.py --federated --strategy fedprox     # FedProx strategy
    python main.py --centralized                      # centralized baselines only
    python main.py --baselines                        # SOTA baselines only
    python main.py --visualize                        # generate plots from saved results
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import ExperimentConfig
from models.mamba_mixer import MambaMixer, residual_loss_fn
from data_provider.ashrae_dataset import (
    get_ashrae_fl_data, get_ashrae_centralized_loaders)
from data_provider.lead_dataset import (
    get_lead_fl_data, get_lead_centralized_loaders)
from trainers.multitask_fed_trainer import MultiTaskFederatedTrainer
from trainers.centralized_trainer import CentralizedTrainer
from utils.metrics import (
    compute_forecasting_metrics, compute_per_horizon_metrics,
    compute_anomaly_metrics, find_threshold_on_validation)
from utils.logging_utils import setup_logger, CSVMetricsLogger


def get_device(config):
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def build_forecasting_model(config):
    """MambaMixer for forecasting: out_len=24."""
    return MambaMixer(
        in_len=config.seq_len, out_len=config.pred_len,
        in_chn=config.in_chn, ex_chn=config.ex_chn, out_chn=config.out_chn,
        patch_sizes=config.patch_sizes, hid_len=config.hid_len,
        hid_chn=config.hid_chn, hid_pch=config.hid_pch,
        hid_pred=config.hid_pred, d_ssm=config.d_ssm,
        state_size=config.state_size, expand=config.expand,
        conv_kernel=config.conv_kernel, last_norm=config.last_norm,
        drop=config.drop,
    )


def build_anomaly_model(config):
    """MambaMixer for anomaly (reconstruction): out_len=seq_len=128."""
    return MambaMixer(
        in_len=config.seq_len, out_len=config.seq_len,
        in_chn=config.in_chn, ex_chn=config.ex_chn, out_chn=config.out_chn,
        patch_sizes=config.patch_sizes, hid_len=config.hid_len,
        hid_chn=config.hid_chn, hid_pch=config.hid_pch,
        hid_pred=config.hid_pred, d_ssm=config.d_ssm,
        state_size=config.state_size, expand=config.expand,
        conv_kernel=config.conv_kernel, last_norm=config.last_norm,
        drop=config.drop,
    )


def build_aux_loss(config):
    """Residual auxiliary loss for MambaMixer."""
    def aux_fn(model):
        if hasattr(model, 'last_residual') and model.last_residual is not None:
            return residual_loss_fn(
                model.last_residual,
                config.lambda_mse, config.lambda_acf, config.acf_cutoff)
        return 0.0
    return aux_fn


def save_results(results, path, log):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", path)


def _fed_result_filename(mode, strategy, tag=""):
    """Generate mode/strategy(/cohort)-specific result filename."""
    return f"federated_{mode}_{strategy}{tag}_results.json"


# =====================================================================
#  1. PREPROCESSING
# =====================================================================

def run_preprocessing(config, log):
    from preprocess import preprocess_ashrae, preprocess_lead
    log.info("=" * 60)
    log.info("STEP 1: PREPROCESSING (cohort=%d)", config.cohort_size)
    log.info("=" * 60)
    ashrae_meta = preprocess_ashrae(config.cohort_size)
    lead_meta = preprocess_lead(config.cohort_size)
    log.info("ASHRAE: %d train / %d test buildings",
             ashrae_meta['n_train_buildings'], ashrae_meta['n_test_buildings'])
    log.info("LEAD:   %d train / %d test buildings",
             lead_meta['n_train_buildings'], lead_meta['n_test_buildings'])
    return ashrae_meta, lead_meta


# =====================================================================
#  2. MULTI-TASK FEDERATED LEARNING
# =====================================================================

def run_federated(config, device, log):
    mode = config.aggregation_mode
    strategy = config.fl_strategy

    log.info("=" * 60)
    log.info("STEP 2: MULTI-TASK FEDERATED LEARNING")
    log.info("  Mode: %s | Strategy: %s", mode, strategy)
    log.info("=" * 60)

    # Load FL data
    fc_client_loaders, fc_test_loader = get_ashrae_fl_data(
        config.ashrae_processed_dir, config.seq_len, config.pred_len,
        config.batch_size, num_workers=config.num_workers)

    an_client_loaders, an_val_loader, an_test_loader = get_lead_fl_data(
        config.lead_processed_dir, config.seq_len, config.batch_size,
        clean_only=config.clean_only, num_workers=config.num_workers)

    # Build models
    fc_model = build_forecasting_model(config)
    an_model = build_anomaly_model(config)
    aux_fn = build_aux_loss(config)

    # Count and log parameters
    fc_params = sum(p.numel() for p in fc_model.parameters())
    an_params = sum(p.numel() for p in an_model.parameters())
    backbone_params = sum(
        p.numel() for n, p in fc_model.named_parameters()
        if MambaMixer.is_backbone_param(n))
    log.info("Forecasting model params: %s", f"{fc_params:,}")
    log.info("Anomaly model params:     %s", f"{an_params:,}")
    log.info("Shared backbone params:   %s", f"{backbone_params:,}")
    log.info("Forecasting head params:  %s", f"{fc_params - backbone_params:,}")
    log.info("Anomaly head params:      %s", f"{an_params - backbone_params:,}")

    # CSV metric logger for per-round tracking
    csv_name = f"federated_{mode}_{strategy}{config.cohort_tag}_rounds.csv"
    csv_logger = CSVMetricsLogger(
        os.path.join(config.log_dir, csv_name),
        ["round", "forecast_test_mse", "anomaly_test_mse", "round_seconds"])

    # Train
    trainer = MultiTaskFederatedTrainer(
        fc_model, an_model, device, config, aux_loss_fn=aux_fn,
        logger=log, csv_logger=csv_logger)

    start_time = time.time()
    history = trainer.train(
        fc_client_loaders, an_client_loaders,
        fc_test_loader, an_test_loader)
    elapsed = time.time() - start_time
    log.info("FL training completed in %.1fs", elapsed)

    csv_logger.close()

    # -- Final evaluation: Forecasting --
    log.info("--- Federated Forecasting Evaluation ---")
    fc_preds, fc_targets = trainer.evaluate_forecasting(fc_test_loader)
    fc_metrics = compute_forecasting_metrics(fc_targets, fc_preds)
    fc_per_horizon = compute_per_horizon_metrics(fc_targets, fc_preds)
    for k, v in fc_metrics.items():
        fmt = f"{v:.6f}" if k != "mape" else f"{v:.2f}%"
        log.info("  %s: %s", k.upper(), fmt)

    # -- Final evaluation: Anomaly Detection --
    log.info("--- Federated Anomaly Detection Evaluation ---")
    val_scores, val_labels = trainer.evaluate_anomaly_detection(an_val_loader)
    threshold = find_threshold_on_validation(val_scores, val_labels)
    log.info("  Threshold (from val): %.6f", threshold)

    test_scores, test_labels = trainer.evaluate_anomaly_detection(
        an_test_loader)
    an_metrics = compute_anomaly_metrics(test_labels, test_scores, threshold)
    for k in ["f1", "auc_roc", "auc_pr", "precision", "recall"]:
        log.info("  %s: %.4f", k.upper(), an_metrics[k])

    # Save
    results = {
        "experiment": "multi_task_federated",
        "aggregation_mode": mode,
        "fl_strategy": strategy,
        "cohort_size": config.cohort_size,
        "config": {
            "cohort_size": config.cohort_size,
            "num_rounds": config.num_rounds,
            "local_epochs": config.local_epochs,
            "n_forecast_clients": len(fc_client_loaders),
            "n_anomaly_clients": len(an_client_loaders),
            "seq_len": config.seq_len,
            "pred_len": config.pred_len,
            "last_norm": config.last_norm,
            "fedprox_mu": config.fedprox_mu if strategy == "fedprox" else None,
        },
        "model_params": {
            "forecasting_total": fc_params,
            "anomaly_total": an_params,
            "shared_backbone": backbone_params,
            "forecasting_head": fc_params - backbone_params,
            "anomaly_head": an_params - backbone_params,
        },
        "forecasting_metrics": fc_metrics,
        "forecasting_per_horizon": fc_per_horizon,
        "anomaly_metrics": an_metrics,
        "history": history,
        "training_time_seconds": elapsed,
    }

    result_file = _fed_result_filename(mode, strategy, config.cohort_tag)
    save_results(results, os.path.join(config.results_dir, result_file), log)

    # Save models
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ckpt_prefix = f"fed_{mode}_{strategy}{config.cohort_tag}"
    torch.save(trainer.forecasting_model.state_dict(),
               os.path.join(config.checkpoint_dir,
                            f"{ckpt_prefix}_forecasting_model.pt"))
    torch.save(trainer.anomaly_model.state_dict(),
               os.path.join(config.checkpoint_dir,
                            f"{ckpt_prefix}_anomaly_model.pt"))
    log.info("Models saved to %s", config.checkpoint_dir)

    return results


# =====================================================================
#  3. CENTRALIZED BASELINES
# =====================================================================

def run_centralized(config, device, log):
    log.info("=" * 60)
    log.info("STEP 3: CENTRALIZED BASELINES")
    log.info("=" * 60)

    results = {"cohort_size": config.cohort_size}

    # -- Centralized Forecasting --
    log.info("--- Centralized Forecasting ---")
    fc_train, fc_val, fc_test = get_ashrae_centralized_loaders(
        config.ashrae_processed_dir, config.seq_len, config.pred_len,
        config.batch_size, config.num_workers)

    fc_model = build_forecasting_model(config)
    aux_fn = build_aux_loss(config)
    trainer_fc = CentralizedTrainer(fc_model, device, config, aux_loss_fn=aux_fn,
                                    logger=log)

    fc_csv = CSVMetricsLogger(
        os.path.join(config.log_dir, "centralized_forecasting_epochs.csv"),
        ["epoch", "train_loss", "val_loss"])

    start = time.time()
    trainer_fc.train_forecasting(fc_train, fc_val, csv_logger=fc_csv)
    fc_time = time.time() - start
    fc_csv.close()

    fc_preds, fc_targets = trainer_fc.evaluate_forecasting(fc_test)
    fc_metrics = compute_forecasting_metrics(fc_targets, fc_preds)
    fc_per_horizon = compute_per_horizon_metrics(fc_targets, fc_preds)
    for k, v in fc_metrics.items():
        fmt = f"{v:.6f}" if k != "mape" else f"{v:.2f}%"
        log.info("  %s: %s", k.upper(), fmt)

    results["centralized_forecasting"] = {
        "metrics": fc_metrics,
        "per_horizon": fc_per_horizon,
        "training_time_seconds": fc_time,
        "history": trainer_fc.history,
    }

    torch.save(trainer_fc.model.state_dict(),
               os.path.join(config.checkpoint_dir,
                            f"centralized_forecasting_model{config.cohort_tag}.pt"))

    # -- Centralized Anomaly Detection --
    log.info("--- Centralized Anomaly Detection ---")
    an_train, an_val, an_test = get_lead_centralized_loaders(
        config.lead_processed_dir, config.seq_len, config.batch_size,
        config.num_workers, clean_only=config.clean_only)

    an_model = build_anomaly_model(config)
    trainer_an = CentralizedTrainer(an_model, device, config,
                                    aux_loss_fn=build_aux_loss(config),
                                    logger=log)

    an_csv = CSVMetricsLogger(
        os.path.join(config.log_dir, "centralized_anomaly_epochs.csv"),
        ["epoch", "train_loss", "val_loss"])

    start = time.time()
    trainer_an.train_anomaly_detection(an_train, an_val,
                                       mask_rate=config.mask_rate,
                                       csv_logger=an_csv)
    an_time = time.time() - start
    an_csv.close()

    # Threshold from validation
    val_scores, val_labels = trainer_an.evaluate_anomaly_detection(an_val)
    threshold = find_threshold_on_validation(val_scores, val_labels)
    log.info("  Threshold (from val): %.6f", threshold)

    test_scores, test_labels = trainer_an.evaluate_anomaly_detection(an_test)
    an_metrics = compute_anomaly_metrics(test_labels, test_scores, threshold)
    for k in ["f1", "auc_roc", "auc_pr", "precision", "recall"]:
        log.info("  %s: %.4f", k.upper(), an_metrics[k])

    results["centralized_anomaly"] = {
        "metrics": an_metrics,
        "training_time_seconds": an_time,
        "history": trainer_an.history,
    }

    torch.save(trainer_an.model.state_dict(),
               os.path.join(config.checkpoint_dir,
                            f"centralized_anomaly_model{config.cohort_tag}.pt"))

    save_results(results, os.path.join(config.results_dir,
                                        f"centralized{config.cohort_tag}_results.json"), log)
    return results


# =====================================================================
#  4. SOTA BASELINES
# =====================================================================

def run_baselines(config, device, log):
    log.info("=" * 60)
    log.info("STEP 4: SOTA BASELINES")
    log.info("=" * 60)

    from experiments.run_baselines import run_all_baselines
    results = run_all_baselines(config, device, log)
    return results


# =====================================================================
#  5. COMPARISON SUMMARY
# =====================================================================

def log_comparison(fed_results, cent_results, log):
    log.info("=" * 60)
    log.info("COMPARISON: Federated vs Centralized")
    log.info("=" * 60)

    log.info("FORECASTING:")
    log.info("  %-10s %12s %12s %10s", "Metric", "Federated", "Centralized",
             "Diff")
    log.info("  %s", "-" * 44)
    for m in ["mse", "rmse", "mae", "r2"]:
        fed_v = fed_results["forecasting_metrics"][m]
        cent_v = cent_results["centralized_forecasting"]["metrics"][m]
        diff = fed_v - cent_v
        sign = "+" if diff > 0 else ""
        log.info("  %-10s %12.6f %12.6f %s%.6f", m, fed_v, cent_v, sign, diff)

    log.info("ANOMALY DETECTION:")
    log.info("  %-10s %12s %12s %10s", "Metric", "Federated", "Centralized",
             "Diff")
    log.info("  %s", "-" * 44)
    for m in ["f1", "auc_roc", "auc_pr", "precision", "recall"]:
        fed_v = fed_results["anomaly_metrics"][m]
        cent_v = cent_results["centralized_anomaly"]["metrics"][m]
        diff = fed_v - cent_v
        sign = "+" if diff > 0 else ""
        log.info("  %-10s %12.4f %12.4f %s%.4f", m, fed_v, cent_v, sign, diff)


# =====================================================================
#  6. VISUALIZATION
# =====================================================================

def run_visualization(config, log):
    log.info("=" * 60)
    log.info("STEP 6: GENERATING RESEARCH PAPER PLOTS")
    log.info("=" * 60)

    from visualization.plots import generate_all_plots

    def _file_cohort_tag(stem):
        """Extract a trailing `_c<N>` cohort tag from a result-file stem, else ''.
        (Cross-cohort comparison plots are a separate follow-up; for now --visualize
        scopes to one cohort so cohorts aren't silently mixed into one figure.)"""
        last = stem.rsplit("_", 1)[-1]
        if last.startswith("c") and last[1:].isdigit():
            return f"_{last}"
        return ""

    # Load federated results for the ACTIVE cohort only (config.cohort_tag).
    fed_results_all = {}
    for fname in os.listdir(config.results_dir):
        if fname.startswith("federated_") and fname.endswith("_results.json"):
            stem = fname[len("federated_"):-len("_results.json")]
            if _file_cohort_tag(stem) != config.cohort_tag:
                continue
            fpath = os.path.join(config.results_dir, fname)
            with open(fpath) as f:
                data = json.load(f)
            mode = data.get("aggregation_mode", "dual")
            strategy = data.get("fl_strategy", "fedavg")
            key = f"{mode}_{strategy}"
            fed_results_all[key] = data
            log.info("  Loaded federated results: %s", fname)

    # Also check legacy federated_results.json
    legacy_path = os.path.join(config.results_dir, "federated_results.json")
    if os.path.exists(legacy_path) and "dual_fedavg" not in fed_results_all:
        with open(legacy_path) as f:
            fed_results_all["dual_fedavg"] = json.load(f)
        log.info("  Loaded legacy federated_results.json as dual_fedavg")

    # Load centralized results
    cent_path = os.path.join(config.results_dir,
                             f"centralized{config.cohort_tag}_results.json")
    cent_results = None
    if os.path.exists(cent_path):
        with open(cent_path) as f:
            cent_results = json.load(f)
        log.info("  Loaded centralized results")

    # Load baseline results
    baseline_path = os.path.join(config.results_dir, "baseline_results.json")
    baseline_results = None
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_results = json.load(f)
        log.info("  Loaded baseline results")

    if not fed_results_all and cent_results is None:
        log.warning("No result files found in %s — skipping visualization.",
                     config.results_dir)
        return

    # Use dual_fedavg as primary federated result for backward-compatible plots
    fed_primary = fed_results_all.get("dual_fedavg")

    figures_dir = os.path.join(config.results_dir, "figures")
    generate_all_plots(fed_primary, cent_results, figures_dir, log,
                       fed_results_all=fed_results_all,
                       baseline_results=baseline_results)
    log.info("All figures saved to %s", figures_dir)


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Task FL Experiment: MambaMixer")
    parser.add_argument("--preprocess", action="store_true",
                        help="Run preprocessing only")
    parser.add_argument("--federated", action="store_true",
                        help="Run federated training only")
    parser.add_argument("--centralized", action="store_true",
                        help="Run centralized baselines only")
    parser.add_argument("--baselines", action="store_true",
                        help="Run SOTA baselines (LSTM, Informer, etc.)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate plots from saved results only")
    parser.add_argument("--mode",
                        choices=["dual", "single_task", "local_only"],
                        default="dual",
                        help="FL aggregation mode (default: dual)")
    parser.add_argument("--strategy",
                        choices=["fedavg", "fedprox", "scaffold"],
                        default="fedavg",
                        help="FL strategy (default: fedavg)")
    parser.add_argument("--cohort", type=int, default=ExperimentConfig.cohort_size,
                        help="Buildings per dataset (default 50; ladder 100/200/400). "
                             "Drives the processed-data dir AND the output tag, so "
                             "preprocess and training/eval stay in sync. Run "
                             "`preprocess.py --cohort N` first to build the data.")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override config.num_rounds. Use for a quick 1-round "
                             "GPU/2-GPU validation (`--rounds 1`) BEFORE the sweep. "
                             "WARNING: a short run still writes the canonical tagged "
                             "result/checkpoints -> delete them before the real sweep, "
                             "or the sweep script will SKIP that condition as 'done'.")
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed,
                        help="Training seed (model init + batch order). The DATA "
                             "split stays fixed at the preprocess seed, so multi-seed "
                             "varies only training randomness (correct for error bars). "
                             "Seeds != 42 tag all outputs `_s{seed}` so runs coexist. "
                             "Default 42 (untagged, back-compat).")
    parser.add_argument("--eval-every", type=int, default=None, dest="eval_every",
                        help="Override config.eval_every: run the per-round TEST eval "
                             "only every K rounds (the final round ALWAYS evals, so "
                             "REPORTED metrics are unchanged -- only the per-round "
                             "history becomes sparse). K>1 trims the per-round eval "
                             "cost (~12%% of a round); handy for long multi-seed "
                             "sweeps. Default = config (1).")
    args = parser.parse_args()

    run_all = not (args.preprocess or args.federated or args.centralized
                   or args.baselines or args.visualize)

    # cohort_size must be set at construction (it drives __post_init__, which
    # repoints the processed-data dirs); mode/strategy are plain attributes.
    config = ExperimentConfig(cohort_size=args.cohort)
    config.aggregation_mode = args.mode
    config.fl_strategy = args.strategy
    config.seed = args.seed  # plain attr: not used in __post_init__; read by
                             # cohort_tag (filenames) + the seeding block below
    if args.rounds is not None:
        config.num_rounds = args.rounds
    if args.eval_every is not None:
        config.eval_every = args.eval_every

    # Setup logging
    os.makedirs(config.log_dir, exist_ok=True)
    log = setup_logger("experiment", config.log_dir)

    device = get_device(config)
    log.info("Device: %s", device)
    log.info("Cohort: %d buildings/dataset (tag=%r) | ashrae=%s | lead=%s",
             config.cohort_size, config.cohort_tag or "(none)",
             config.ashrae_processed_dir, config.lead_processed_dir)

    # Fused mamba-ssm / causal-conv1d kernels (no-op fallback if not installed)
    import models.mamba_mixer as mamba_mixer
    mamba_mixer.USE_KERNELS = config.use_mamba_kernel
    kernels_active = (config.use_mamba_kernel
                      and mamba_mixer._HAS_SELECTIVE_SCAN
                      and device.type == "cuda")
    log.info("Mamba fused kernels: %s (selective_scan=%s, causal_conv1d=%s)",
             "ON" if kernels_active else "OFF (eager fallback)",
             mamba_mixer._HAS_SELECTIVE_SCAN, mamba_mixer._HAS_CAUSAL_CONV)

    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # -- Preprocessing (only when explicitly requested) --
    if args.preprocess:
        run_preprocessing(config, log)
        return

    # Check that preprocessed data exists
    for d in [config.ashrae_processed_dir, config.lead_processed_dir]:
        meta = os.path.join(d, "split_metadata.json")
        if not os.path.exists(meta):
            log.error("%s not found. Run with --preprocess first.", meta)
            sys.exit(1)

    # -- Visualization only --
    if args.visualize:
        run_visualization(config, log)
        log.info("[DONE] Visualization complete.")
        return

    # -- SOTA Baselines only --
    if args.baselines:
        run_baselines(config, device, log)
        log.info("[DONE] Baselines complete.")
        return

    # -- Federated --
    fed_results = None
    if args.federated or run_all:
        fed_results = run_federated(config, device, log)

    # -- Centralized --
    cent_results = None
    if args.centralized or run_all:
        cent_results = run_centralized(config, device, log)

    # -- Comparison --
    if fed_results and cent_results:
        log_comparison(fed_results, cent_results, log)

        combined = {
            "federated": fed_results,
            "centralized": cent_results,
        }
        save_results(combined, os.path.join(config.results_dir,
                                             "comparison.json"), log)

    # -- Generate plots --
    if run_all or (fed_results or cent_results):
        run_visualization(config, log)

    log.info("[DONE] Experiment complete.")


if __name__ == "__main__":
    main()
