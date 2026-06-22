"""
LEAD data provider for anomaly detection (reconstruction) task.

Loads from preprocessed data + split_metadata.json.
  - seq_len=128 (standardized with forecasting task)
  - log1p applied (standardized with ASHRAE)
  - clean_only for AE training (exclude anomalous windows)
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
    return pd.read_csv(os.path.join(processed_dir, "lead_clean.csv"),
                       parse_dates=["timestamp"])


def _get_building_data(df, building_id):
    bdf = df[df["building_id"] == building_id].sort_values("timestamp")
    readings = bdf["meter_reading"].values.astype(np.float32)
    labels = bdf["anomaly"].values.astype(np.int64)
    return readings, labels


class LEADWindowDataset(Dataset):
    """Sliding-window dataset for autoencoder anomaly detection.

    flag="train"/"val"/"test": windows from temporal portion.
    flag="full": windows over entire time series (for FL test buildings).
    clean_only: if True, SKIP windows containing ANY anomaly (for AE training).
    """

    def __init__(self, processed_dir, building_ids, flag, seq_len,
                 clean_only=False):
        meta = _load_metadata(processed_dir)
        df = _load_clean_csv(processed_dir)

        self.seq_len = seq_len
        self.building_readings = {}
        self.building_labels = {}
        self.window_index = []

        for bid in building_ids:
            readings, labels = _get_building_data(df, bid)
            self.building_readings[bid] = readings
            self.building_labels[bid] = labels
            sp = meta["per_building_splits"].get(str(bid), {})

            if flag == "full":
                start, end = 0, len(readings)
            else:
                start, end = sp[flag]

            for s in range(start, end - seq_len + 1):
                if clean_only and labels[s:s + seq_len].any():
                    continue
                self.window_index.append((bid, s))

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        bid, s = self.window_index[idx]
        readings = self.building_readings[bid]
        labels = self.building_labels[bid]
        seq = readings[s:s + self.seq_len].reshape(-1, 1).copy()
        lab = int(labels[s:s + self.seq_len].any())
        return torch.FloatTensor(seq), lab


# ── Centralized DataLoaders ──

def get_lead_centralized_loaders(processed_dir, seq_len=128, batch_size=64,
                                 num_workers=0, clean_only=True):
    """Centralized: use the 35 TRAIN buildings (fair comparison with FL)."""
    meta = _load_metadata(processed_dir)
    train_bids = meta["train_building_ids"]
    test_bids = meta["test_building_ids"]

    train_ds = LEADWindowDataset(processed_dir, train_bids, "train", seq_len,
                                 clean_only=clean_only)
    val_ds = LEADWindowDataset(processed_dir, train_bids, "val", seq_len,
                               clean_only=False)
    test_ds = LEADWindowDataset(processed_dir, test_bids, "full", seq_len,
                                clean_only=False)

    clean_str = " (clean only)" if clean_only else ""
    print(f"[LEAD-Centralized] {len(train_ds)} train{clean_str}, "
          f"{len(val_ds)} val, {len(test_ds)} test windows "
          f"({len(train_bids)} train + {len(test_bids)} test buildings)")

    kw = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   drop_last=True, **kw),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw),
    )


# ── FL DataLoaders ──

def get_lead_fl_data(processed_dir, seq_len=128, batch_size=64,
                     clean_only=True, num_workers=0):
    """FL: 1 building = 1 client from train buildings.

    Returns:
        client_loaders: {bid: {"train": loader, "val": loader, "n_samples": int}}
        val_loader: DataLoader over train buildings' val portion (for threshold)
        test_loader: DataLoader over held-out test buildings (full time series)
    """
    meta = _load_metadata(processed_dir)
    train_bids = meta["train_building_ids"]
    test_bids = meta["test_building_ids"]

    print(f"[FL-LEAD] {len(train_bids)} clients, "
          f"{len(test_bids)} test buildings")

    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=num_workers > 0)
    client_loaders = {}
    for bid in train_bids:
        train_ds = LEADWindowDataset(processed_dir, [bid], "train", seq_len,
                                     clean_only=clean_only)
        val_ds = LEADWindowDataset(processed_dir, [bid], "val", seq_len,
                                   clean_only=False)
        if len(train_ds) == 0:
            continue
        client_loaders[bid] = {
            "train": DataLoader(train_ds, batch_size=batch_size,
                                shuffle=True, drop_last=False, **kw),
            "val": DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, **kw) if len(val_ds) > 0 else None,
            "n_samples": len(train_ds),
        }

    val_ds = LEADWindowDataset(processed_dir, train_bids, "val", seq_len,
                               clean_only=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)

    test_ds = LEADWindowDataset(processed_dir, test_bids, "full", seq_len,
                                clean_only=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             **kw)

    print(f"[FL-LEAD] {len(client_loaders)} active clients, "
          f"{len(val_ds)} val, {len(test_ds)} test windows")
    return client_loaders, val_loader, test_loader
