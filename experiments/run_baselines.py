"""
Centralized baseline runner for architecture comparison.

Trains LSTM, Informer, MSD-Mixer (forecasting) and LSTM-AE, ANN-AE,
MSD-Mixer (anomaly) on the EXACT same data splits and evaluates with
the EXACT same metrics as the MambaMixer experiments.

Usage:
    python -m experiments.run_baselines                    # run all
    python -m experiments.run_baselines --task forecasting  # forecasting only
    python -m experiments.run_baselines --task anomaly      # anomaly only
    python -m experiments.run_baselines --model lstm        # specific model
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import ExperimentConfig
from data_provider.ashrae_dataset import get_ashrae_centralized_loaders
from data_provider.lead_dataset import get_lead_centralized_loaders
from trainers.centralized_trainer import CentralizedTrainer
from utils.metrics import (
    compute_forecasting_metrics, compute_anomaly_metrics,
    find_threshold_on_validation)
from utils.logging_utils import setup_logger


# -- Model builders --

def build_forecasting_baseline(name, config):
    """Build a forecasting baseline model matching the experiment's data shape."""
    if name == "lstm":
        from models.baselines import LSTMForecaster
        return LSTMForecaster(
            in_chn=config.in_chn, out_chn=config.out_chn,
            seq_len=config.seq_len, pred_len=config.pred_len,
            hidden_dim=128, num_layers=2, dropout=0.2)
    elif name == "informer":
        from models.baselines import Informer
        return Informer(
            in_chn=config.in_chn, out_chn=config.out_chn,
            seq_len=config.seq_len, pred_len=config.pred_len,
            d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.1)
    elif name == "msd_mixer":
        from models.baselines import MSDMixer
        return MSDMixer(
            in_len=config.seq_len, out_len=config.pred_len,
            in_chn=config.in_chn, ex_chn=config.ex_chn,
            out_chn=config.out_chn,
            patch_sizes=config.patch_sizes, hid_len=config.hid_len,
            hid_chn=config.hid_chn, hid_pch=config.hid_pch,
            hid_pred=config.hid_pred, last_norm=config.last_norm,
            drop=config.drop)
    else:
        raise ValueError(f"Unknown forecasting baseline: {name}")


def build_anomaly_baseline(name, config):
    """Build an anomaly detection baseline model."""
    if name == "lstm_ae":
        from models.baselines import LSTMAutoEncoder
        return LSTMAutoEncoder(
            seq_len=config.seq_len, input_dim=config.in_chn,
            hidden_dim=64, num_layers=2, dropout=0.2)
    elif name == "ann_ae":
        from models.baselines import ANNAutoEncoder
        return ANNAutoEncoder(
            seq_len=config.seq_len, input_dim=config.in_chn,
            hidden_dims=(64, 32, 16), dropout=0.2)
    elif name == "msd_mixer":
        from models.baselines import MSDMixer
        return MSDMixer(
            in_len=config.seq_len, out_len=config.seq_len,
            in_chn=config.in_chn, ex_chn=config.ex_chn,
            out_chn=config.out_chn,
            patch_sizes=config.patch_sizes, hid_len=config.hid_len,
            hid_chn=config.hid_chn, hid_pch=config.hid_pch,
            hid_pred=config.hid_pred, last_norm=config.last_norm,
            drop=config.drop)
    else:
        raise ValueError(f"Unknown anomaly baseline: {name}")


# -- Runner functions --

def run_forecasting_baseline(name, config, device, log):
    """Train and evaluate a single forecasting baseline."""
    log.info("--- Baseline Forecasting: %s ---", name.upper())

    model = build_forecasting_baseline(name, config)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("  Parameters: %s", f"{n_params:,}")

    # Use same data loaders as MambaMixer centralized
    train_loader, val_loader, test_loader = get_ashrae_centralized_loaders(
        config.ashrae_processed_dir, config.seq_len, config.pred_len,
        config.batch_size, config.num_workers)

    # aux_loss only for MSD-Mixer (has last_residual)
    aux_fn = None
    if hasattr(model, 'last_residual'):
        from models.mamba_mixer import residual_loss_fn
        def aux_fn(m):
            if hasattr(m, 'last_residual') and m.last_residual is not None:
                return residual_loss_fn(
                    m.last_residual,
                    config.lambda_mse, config.lambda_acf, config.acf_cutoff)
            return 0.0

    trainer = CentralizedTrainer(model, device, config, aux_loss_fn=aux_fn,
                                 logger=log)

    start = time.time()
    trainer.train_forecasting(train_loader, val_loader)
    elapsed = time.time() - start

    preds, targets = trainer.evaluate_forecasting(test_loader)
    metrics = compute_forecasting_metrics(targets, preds)

    for k, v in metrics.items():
        fmt = f"{v:.6f}" if k != "mape" else f"{v:.2f}%"
        log.info("  %s: %s", k.upper(), fmt)
    log.info("  Training time: %.1fs", elapsed)

    return {
        "model": name,
        "task": "forecasting",
        "metrics": metrics,
        "n_params": n_params,
        "training_time_seconds": elapsed,
    }


def run_anomaly_baseline(name, config, device, log):
    """Train and evaluate a single anomaly detection baseline."""
    log.info("--- Baseline Anomaly Detection: %s ---", name.upper())

    model = build_anomaly_baseline(name, config)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("  Parameters: %s", f"{n_params:,}")

    # Use same data loaders as MambaMixer centralized
    train_loader, val_loader, test_loader = get_lead_centralized_loaders(
        config.lead_processed_dir, config.seq_len, config.batch_size,
        config.num_workers, clean_only=config.clean_only)

    aux_fn = None
    if hasattr(model, 'last_residual'):
        from models.mamba_mixer import residual_loss_fn
        def aux_fn(m):
            if hasattr(m, 'last_residual') and m.last_residual is not None:
                return residual_loss_fn(
                    m.last_residual,
                    config.lambda_mse, config.lambda_acf, config.acf_cutoff)
            return 0.0

    trainer = CentralizedTrainer(model, device, config, aux_loss_fn=aux_fn,
                                 logger=log)

    start = time.time()
    trainer.train_anomaly_detection(train_loader, val_loader,
                                    mask_rate=config.mask_rate)
    elapsed = time.time() - start

    # Threshold from validation
    val_scores, val_labels = trainer.evaluate_anomaly_detection(val_loader)
    threshold = find_threshold_on_validation(val_scores, val_labels)
    log.info("  Threshold (from val): %.6f", threshold)

    test_scores, test_labels = trainer.evaluate_anomaly_detection(test_loader)
    metrics = compute_anomaly_metrics(test_labels, test_scores, threshold)

    for k in ["f1", "auc_roc", "auc_pr", "precision", "recall"]:
        log.info("  %s: %.4f", k.upper(), metrics[k])
    log.info("  Training time: %.1fs", elapsed)

    return {
        "model": name,
        "task": "anomaly_detection",
        "metrics": metrics,
        "n_params": n_params,
        "training_time_seconds": elapsed,
    }


def run_all_baselines(config, device, log):
    """Run all centralized baselines."""
    results = {}

    # Forecasting baselines
    for name in ["lstm", "informer", "msd_mixer"]:
        key = f"forecasting_{name}"
        results[key] = run_forecasting_baseline(name, config, device, log)

    # Anomaly baselines
    for name in ["lstm_ae", "ann_ae", "msd_mixer"]:
        key = f"anomaly_{name}"
        results[key] = run_anomaly_baseline(name, config, device, log)

    # Save
    out_path = os.path.join(config.results_dir, "baseline_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("All baseline results saved to %s", out_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run centralized baselines")
    parser.add_argument("--task", choices=["forecasting", "anomaly"],
                        default=None, help="Run specific task only")
    parser.add_argument("--model", type=str, default=None,
                        help="Run specific model only")
    args = parser.parse_args()

    config = ExperimentConfig()

    log = setup_logger("baselines", config.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if args.model:
        # Run a specific model
        if args.task == "anomaly" or args.model in ["lstm_ae", "ann_ae"]:
            result = run_anomaly_baseline(args.model, config, device, log)
        else:
            result = run_forecasting_baseline(args.model, config, device, log)
        out_path = os.path.join(config.results_dir,
                                f"baseline_{args.model}_results.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
    elif args.task == "forecasting":
        for name in ["lstm", "informer", "msd_mixer"]:
            run_forecasting_baseline(name, config, device, log)
    elif args.task == "anomaly":
        for name in ["lstm_ae", "ann_ae", "msd_mixer"]:
            run_anomaly_baseline(name, config, device, log)
    else:
        run_all_baselines(config, device, log)

    log.info("[DONE] Baselines complete.")


if __name__ == "__main__":
    main()
