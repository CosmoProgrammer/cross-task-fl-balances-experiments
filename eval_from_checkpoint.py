"""Eval-only: re-emit the (now richer) metric set from existing checkpoints
WITHOUT retraining.

Loads trained forecasting + anomaly checkpoints, runs the same evaluation path
as run_federated (validation-only anomaly threshold), and writes a results JSON
with the full metric set including the newly added smape/wape/nrmse/mcc and the
per-horizon forecasting breakdown. Use it to deepen the tables on runs that are
already done (e.g. the committed 50-building proposed model).

    conda run -n crosstask python eval_from_checkpoint.py                # committed proposed weights
    conda run -n crosstask python eval_from_checkpoint.py --tag dual_fedavg \
        --fc-ckpt checkpoints/fed_dual_fedavg_forecasting_model.pt \
        --an-ckpt checkpoints/fed_dual_fedavg_anomaly_model.pt

Mirrors main.py's kernel setup (kernels on -> faster eval; fp-equivalent to eager
per test_kernel_parity.py). Safe to delete; nothing imports it.
"""
import argparse
import os

import numpy as np
import torch

import models.mamba_mixer as mamba_mixer
from main import (build_forecasting_model, build_anomaly_model, build_aux_loss,
                  get_device, save_results)
from data_provider.ashrae_dataset import get_ashrae_fl_data
from data_provider.lead_dataset import get_lead_fl_data
from trainers.multitask_fed_trainer import MultiTaskFederatedTrainer
from utils.metrics import (compute_forecasting_metrics, compute_per_horizon_metrics,
                           compute_anomaly_metrics, find_threshold_on_validation)
from utils.logging_utils import setup_logger
from configs.config import ExperimentConfig


def _load(model, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)  # strict: dims must match config defaults
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc-ckpt", default="checkpoints/fed_forecasting_model.pt")
    ap.add_argument("--an-ckpt", default="checkpoints/fed_anomaly_model.pt")
    ap.add_argument("--tag", default="reeval",
                    help="label for the output JSON (results/reeval_<tag>_results.json)")
    ap.add_argument("--cohort", type=int, default=ExperimentConfig.cohort_size,
                    help="Cohort size of the checkpoints being eval'd (default 50). "
                         "Must match the run, so the right processed-data dir is read.")
    ap.add_argument("--dump-scores", default=None,
                    help="If set, save raw per-window anomaly (val/test) scores+labels "
                         "to this .npz path. These arrays are the re-derivable artifact: "
                         "all matched-operating-point analysis (P@R, threshold transfer, "
                         "PR/ROC curves) runs off them on any machine with no GPU/data.")
    args = ap.parse_args()

    config = ExperimentConfig(cohort_size=args.cohort)
    device = get_device(config)
    log = setup_logger("eval_from_checkpoint", config.log_dir)

    # Mirror main.py: activate fused kernels if available (fp-equivalent to eager).
    mamba_mixer.USE_KERNELS = config.use_mamba_kernel
    kernels_active = (config.use_mamba_kernel
                      and mamba_mixer._HAS_SELECTIVE_SCAN
                      and device.type == "cuda")
    log.info("Device: %s | Mamba fused kernels: %s", device,
             "ON" if kernels_active else "OFF (eager fallback)")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Data (only the eval splits are needed)
    _, fc_test_loader = get_ashrae_fl_data(
        config.ashrae_processed_dir, config.seq_len, config.pred_len,
        config.batch_size, num_workers=config.num_workers)
    _, an_val_loader, an_test_loader = get_lead_fl_data(
        config.lead_processed_dir, config.seq_len, config.batch_size,
        clean_only=config.clean_only, num_workers=config.num_workers)

    # Models + checkpoints
    fc_model = build_forecasting_model(config)
    an_model = build_anomaly_model(config)
    trainer = MultiTaskFederatedTrainer(
        fc_model, an_model, device, config,
        aux_loss_fn=build_aux_loss(config), logger=log)
    _load(trainer.forecasting_model, args.fc_ckpt, device)
    _load(trainer.anomaly_model, args.an_ckpt, device)
    log.info("Loaded fc=%s an=%s", args.fc_ckpt, args.an_ckpt)

    # Forecasting eval (+ per-horizon)
    log.info("--- Forecasting eval ---")
    fc_preds, fc_targets = trainer.evaluate_forecasting(fc_test_loader)
    fc_metrics = compute_forecasting_metrics(fc_targets, fc_preds)
    fc_per_horizon = compute_per_horizon_metrics(fc_targets, fc_preds)
    for k, v in fc_metrics.items():
        log.info("  %s: %.6f", k.upper(), v)
    log.info("  per-horizon MAE h1/h12/h24: %.4f / %.4f / %.4f",
             fc_per_horizon[0]["mae"], fc_per_horizon[11]["mae"],
             fc_per_horizon[23]["mae"])

    # Anomaly eval (validation-only threshold, then test)
    log.info("--- Anomaly eval ---")
    val_scores, val_labels = trainer.evaluate_anomaly_detection(an_val_loader)
    threshold = find_threshold_on_validation(val_scores, val_labels)
    test_scores, test_labels = trainer.evaluate_anomaly_detection(an_test_loader)
    an_metrics = compute_anomaly_metrics(test_labels, test_scores, threshold)
    for k in ["f1", "mcc", "balanced_accuracy", "fpr", "auc_roc", "auc_pr",
              "precision", "recall", "accuracy"]:
        log.info("  %s: %.4f", k.upper(), an_metrics[k])

    if args.dump_scores:
        # Raw per-window scores+labels: the artifact for matched-operating-point
        # analysis. Same fixed 15-building test set across cohorts -> directly
        # comparable. scp this .npz off the server and analyze on the laptop.
        np.savez_compressed(
            args.dump_scores,
            val_scores=np.asarray(val_scores).flatten(),
            val_labels=np.asarray(val_labels).flatten(),
            test_scores=np.asarray(test_scores).flatten(),
            test_labels=np.asarray(test_labels).flatten(),
            val_threshold=np.float64(threshold),
            cohort=np.int64(config.cohort_size),
        )
        log.info("  dumped anomaly scores -> %s", args.dump_scores)

    results = {
        "experiment": "eval_from_checkpoint",
        "checkpoints": {"forecasting": args.fc_ckpt, "anomaly": args.an_ckpt},
        "forecasting_metrics": fc_metrics,
        "forecasting_per_horizon": fc_per_horizon,
        "anomaly_metrics": an_metrics,
    }
    out = os.path.join(config.results_dir, f"reeval_{args.tag}_results.json")
    save_results(results, out, log)


if __name__ == "__main__":
    main()
