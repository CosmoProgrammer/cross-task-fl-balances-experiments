"""
ASHRAE data provider for forecasting task.

Loads from preprocessed data + split_metadata.json.
  - seq_len=128 (standardized with anomaly task)
  - pred_len=24
  - Provides centralized and FL data loaders
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
    return pd.read_csv(os.path.join(processed_dir, "ashrae_clean.csv"),
                       parse_dates=["timestamp"])


def _get_building_values(df, building_id):
    bdf = df[df["building_id"] == building_id].sort_values("timestamp")
    return bdf["meter_reading"].values.astype(np.float32)


class ASHRAEWindowDataset(Dataset):
    """Sliding-window dataset for forecasting.

    flag="train"/"val"/"test": windows from temporal portion.
    flag="full": windows over entire time series (for FL test buildings).
    """

    def __init__(self, processed_dir, building_ids, flag, seq_len, pred_len):
        meta = _load_metadata(processed_dir)
        df = _load_clean_csv(processed_dir)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.building_data = {}
        self.window_index = []

        for bid in building_ids:
            vals = _get_building_values(df, bid)
            self.building_data[bid] = vals
            sp = meta["per_building_splits"].get(str(bid), {})

            if flag == "full":
                target_start = seq_len
                target_end = len(vals)
            else:
                target_start, target_end = sp[flag]
                target_start = max(target_start, seq_len)

            for t in range(target_start, target_end - pred_len + 1):
                self.window_index.append((bid, t))

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        bid, t = self.window_index[idx]
        vals = self.building_data[bid]
        x = vals[t - self.seq_len:t].reshape(-1, 1).copy()
        y = vals[t:t + self.pred_len].reshape(-1, 1).copy()
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ── Centralized DataLoaders ──

def get_ashrae_centralized_loaders(processed_dir, seq_len=128, pred_len=24,
                                   batch_size=32, num_workers=0):
    """Centralized: use the 35 TRAIN buildings only (for fair comparison with FL)."""
    meta = _load_metadata(processed_dir)
    train_bids = meta["train_building_ids"]
    test_bids = meta["test_building_ids"]

    train_ds = ASHRAEWindowDataset(processed_dir, train_bids, "train",
                                   seq_len, pred_len)
    val_ds = ASHRAEWindowDataset(processed_dir, train_bids, "val",
                                 seq_len, pred_len)
    test_ds = ASHRAEWindowDataset(processed_dir, test_bids, "full",
                                  seq_len, pred_len)

    print(f"[ASHRAE-Centralized] {len(train_ds)} train, {len(val_ds)} val, "
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

def get_ashrae_fl_data(processed_dir, seq_len=128, pred_len=24, batch_size=32,
                       num_workers=0):
    """FL: 1 building = 1 client from train buildings.

    Returns:
        client_loaders: {bid: {"train": loader, "val": loader, "n_samples": int}}
        test_loader: DataLoader over held-out test buildings (full time series)
    """
    meta = _load_metadata(processed_dir)
    train_bids = meta["train_building_ids"]
    test_bids = meta["test_building_ids"]

    print(f"[FL-ASHRAE] {len(train_bids)} clients, "
          f"{len(test_bids)} test buildings")

    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=num_workers > 0)
    client_loaders = {}
    for bid in train_bids:
        train_ds = ASHRAEWindowDataset(processed_dir, [bid], "train",
                                       seq_len, pred_len)
        val_ds = ASHRAEWindowDataset(processed_dir, [bid], "val",
                                     seq_len, pred_len)
        if len(train_ds) == 0:
            continue
        client_loaders[bid] = {
            "train": DataLoader(train_ds, batch_size=batch_size,
                                shuffle=True, drop_last=False, **kw),
            "val": DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, **kw) if len(val_ds) > 0 else None,
            "n_samples": len(train_ds),
        }

    test_ds = ASHRAEWindowDataset(processed_dir, test_bids, "full",
                                  seq_len, pred_len)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             **kw)

    print(f"[FL-ASHRAE] {len(client_loaders)} active clients, "
          f"{len(test_ds)} test windows")
    return client_loaders, test_loader
