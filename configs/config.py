"""
Configuration for Multi-Task Federated Learning experiment.

MambaMixer (SSD-Net) with cross-task backbone aggregation.
  - 50 ASHRAE buildings: 35 forecasting train clients + 15 test
  - 50 LEAD buildings:   35 anomaly train clients + 15 test
  - Backbone aggregated across ALL 70 clients (cross-task)
  - Task heads aggregated within each task group (35 each)
"""
import os
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


@dataclass
class ExperimentConfig:
    """Single config for the entire multi-task FL experiment."""

    # ── Paths ──
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"
    project_root: str = PROJECT_ROOT
    results_dir: str = os.path.join(PROJECT_ROOT, "results")
    checkpoint_dir: str = os.path.join(PROJECT_ROOT, "checkpoints")
    log_dir: str = os.path.join(PROJECT_ROOT, "logs")

    # Processed data paths (pre-processed data lives here)
    ashrae_processed_dir: str = os.path.join(DATA_DIR, "ashrae", "processed")
    lead_processed_dir: str = os.path.join(DATA_DIR, "lead", "processed")

    # ── Building splits (fixed) ──
    n_total_ashrae: int = 50      # total ASHRAE buildings to select
    n_forecast_train: int = 35    # forecasting training clients
    n_forecast_test: int = 15     # forecasting test buildings

    n_total_lead: int = 50        # total LEAD buildings to select
    n_anomaly_train: int = 35     # anomaly training clients
    n_anomaly_test: int = 15      # anomaly test buildings

    # ── Shared model architecture (MambaMixer / SSD-Net) ──
    seq_len: int = 128            # input sequence length (same for both tasks)
    pred_len: int = 24            # forecasting prediction horizon
    in_chn: int = 1               # univariate
    ex_chn: int = 0               # no exogenous features
    out_chn: int = 1

    patch_sizes: tuple = (24, 12, 6, 2, 1)
    hid_len: int = 128
    hid_chn: int = 256
    hid_pch: int = 64
    hid_pred: int = 128
    d_ssm: int = 64
    state_size: int = 32
    expand: int = 2
    conv_kernel: int = 4
    drop: float = 0.15
    last_norm: bool = True        # True for BOTH tasks (standardized)

    # ── Auxiliary residual loss ──
    lambda_mse: float = 0.1
    lambda_acf: float = 0.3
    acf_cutoff: int = 2

    # ── Federated learning ──
    num_rounds: int = 10
    local_epochs: int = 5
    client_lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    participation_rate: float = 1.0  # 100% — all 70 clients per round
    max_federated_gpus: int = 1      # safe default: avoid threaded multi-GPU CUDA

    # ── FL aggregation mode and strategy ──
    aggregation_mode: str = "dual"    # "dual", "single_task", "local_only"
    fl_strategy: str = "fedavg"       # "fedavg", "fedprox", "scaffold" (Scaffold is dropped now)
    fedprox_mu: float = 0.01          # proximal term weight for FedProx
    scaffold_lr: float = 1.0          # SCAFFOLD server learning rate (scaffold is dropped now)

    # ── Anomaly-specific ──
    mask_rate: float = 0.25
    clean_only: bool = True       # train AEs on clean data only

    # ── Centralized baseline ──
    centralized_max_epochs: int = 100
    centralized_lr: float = 1e-3
    centralized_early_stop: int = 15
    centralized_warmup: int = 5
    centralized_min_lr: float = 1e-7

    # ── DataLoader ──
    batch_size: int = 32
    num_workers: int = 0

    # ── Performance ──
    # Fused mamba-ssm / causal-conv1d kernels: used when importable AND on CUDA
    # (auto-falls back to the pure-PyTorch scan otherwise, e.g. the dev laptop).
    # This is the main training-speed lever -- the eager scan is launch/memory
    # bound, not compute bound. See profile_speed.py.
    use_mamba_kernel: bool = True
    # torch.compile is NOT wired into the FL trainer (its per-round deepcopy of
    # clients is incompatible with compile caching). Evaluate it via
    # profile_speed.py --compile on the Linux server instead.

    # ── Preprocessing ──
    temporal_train: float = 0.70
    temporal_val: float = 0.20
    temporal_test: float = 0.10
