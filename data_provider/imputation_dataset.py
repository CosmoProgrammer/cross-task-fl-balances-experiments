"""
Imputation data provider (mask-and-reconstruct on ASHRAE).

The imputation task reuses the anomaly task's mask-and-reconstruct machinery but
on a THIRD, DISJOINT group of ASHRAE buildings (decouples data domain from
objective type — see PREREGISTERED_PREDICTIONS.md). Data layout mirrors the
forecasting ASHRAE provider (building_id, timestamp, meter_reading; log1p'd), so
windows are length-`seq_len` slices reconstructed under a random mask.

  - seq_len=128 (standardized with the other tasks)
  - No forecast target and no anomaly label: the window IS the reconstruction
    target; the mask is applied at train/eval time (in the trainer), not here.
  - The dataset yields (seq, 0) so it plugs straight into the reconstruct
    training path, which expects a (sequence, label) pair like LEAD. The dummy
    label is unused (imputation trains on ALL windows — there are no anomalies).

Eval masking is generated deterministically in the trainer's imputation eval
(seeded, in dataloader order) so masked-position metrics are reproducible; the
dataset itself is mask-agnostic.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def _load_metadata(processed_dir):
    with open(os.path.join(processed_dir, "split_metadata.json")) as f:
        return json.load(f)


def _load_clean_csv(processed_dir):
    return pd.read_csv(os.path.join(processed_dir, "imputation_clean.csv"),
                       parse_dates=["timestamp"])


def _get_building_values(df, building_id):
    bdf = df[df["building_id"] == building_id].sort_values("timestamp")
    return bdf["meter_reading"].values.astype(np.float32)


class ImputationWindowDataset(Dataset):
    """Sliding-window dataset for masked reconstruction (imputation).

    flag="train"/"val"/"test": windows from the temporal portion.
    flag="full": windows over the entire time series (for FL test buildings).
    Trains on ALL windows (no clean-only filter — ASHRAE has no anomaly labels).
    """

    def __init__(self, processed_dir, building_ids, flag, seq_len):
        meta = _load_metadata(processed_dir)
        df = _load_clean_csv(processed_dir)

        self.seq_len = seq_len
        self.building_data = {}
        self.window_index = []

        for bid in building_ids:
            vals = _get_building_values(df, bid)
            self.building_data[bid] = vals
            sp = meta["per_building_splits"].get(str(bid), {})

            if flag == "full":
                start, end = 0, len(vals)
            else:
                start, end = sp[flag]

            for s in range(start, end - seq_len + 1):
                self.window_index.append((bid, s))

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        bid, s = self.window_index[idx]
        vals = self.building_data[bid]
        seq = vals[s:s + self.seq_len].reshape(-1, 1).copy()
        return torch.FloatTensor(seq), 0


# ── Centralized DataLoaders ──

def get_imputation_centralized_loaders(processed_dir, seq_len=128, batch_size=32,
                                       num_workers=0):
    """Centralized: use the TRAIN buildings only (fair comparison with FL)."""
    meta = _load_metadata(processed_dir)
    train_bids = meta["train_building_ids"]
    test_bids = meta["test_building_ids"]

    train_ds = ImputationWindowDataset(processed_dir, train_bids, "train", seq_len)
    val_ds = ImputationWindowDataset(processed_dir, train_bids, "val", seq_len)
    test_ds = ImputationWindowDataset(processed_dir, test_bids, "full", seq_len)

    print(f"[IMP-Centralized] {len(train_ds)} train, {len(val_ds)} val, "
          f"{len(test_ds)} test windows "
          f"({len(train_bids)} train + {len(test_bids)} test buildings)")

    kw = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   drop_last=True, **kw),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw),
    )


# ── FL DataLoaders ──

def get_imputation_fl_data(processed_dir, seq_len=128, batch_size=32,
                           num_workers=0):
    """FL: 1 building = 1 client from train buildings.

    Returns:
        client_loaders: {bid: {"train": loader, "val": loader, "n_samples": int}}
        val_loader: DataLoader over train buildings' val portion (monitoring only)
        test_loader: DataLoader over held-out test buildings (full time series)
    """
    meta = _load_metadata(processed_dir)
    train_bids = meta["train_building_ids"]
    test_bids = meta["test_building_ids"]

    print(f"[FL-IMP] {len(train_bids)} clients, {len(test_bids)} test buildings")

    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=num_workers > 0)
    client_loaders = {}
    for bid in train_bids:
        train_ds = ImputationWindowDataset(processed_dir, [bid], "train", seq_len)
        val_ds = ImputationWindowDataset(processed_dir, [bid], "val", seq_len)
        if len(train_ds) == 0:
            continue
        client_loaders[bid] = {
            "train": DataLoader(train_ds, batch_size=batch_size,
                                shuffle=True, drop_last=False, **kw),
            "val": DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, **kw) if len(val_ds) > 0 else None,
            "n_samples": len(train_ds),
        }

    val_ds = ImputationWindowDataset(processed_dir, train_bids, "val", seq_len)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)

    test_ds = ImputationWindowDataset(processed_dir, test_bids, "full", seq_len)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)

    print(f"[FL-IMP] {len(client_loaders)} active clients, "
          f"{len(val_ds)} val, {len(test_ds)} test windows")
    return client_loaders, val_loader, test_loader
