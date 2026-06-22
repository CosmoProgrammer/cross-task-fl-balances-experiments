"""
Preprocessing for Multi-Task Federated Learning experiment.

Selects 50 ASHRAE buildings (35 train + 15 test) for forecasting
and 50 LEAD buildings (35 train + 15 test) for anomaly detection.


Usage:
    python preprocess.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Raw data paths — set these environment variables or edit before running
ASHRAE_RAW = os.environ.get("ASHRAE_RAW_CSV", os.path.join(PROJECT_ROOT, "raw", "train_ashrae.csv"))
LEAD_RAW = os.environ.get("LEAD_RAW_CSV", os.path.join(PROJECT_ROOT, "raw", "train_lead.csv"))

ASHRAE_OUT = os.path.join(PROJECT_ROOT, "data", "ashrae", "processed")
LEAD_OUT = os.path.join(PROJECT_ROOT, "data", "lead", "processed")

# ── Constants ──
SEED = 42
N_SELECT = 50          # buildings to select from each dataset
N_TRAIN = 35           # training clients
N_TEST = 15            # held-out test buildings
TEMPORAL_TRAIN = 0.70
TEMPORAL_VAL = 0.20
TEMPORAL_TEST = 0.10


def preprocess_ashrae():
    """Clean ASHRAE, select 50 buildings, split 35/15, log1p, save."""
    os.makedirs(ASHRAE_OUT, exist_ok=True)

    print("[ASHRAE] Loading raw CSV ...")
    df = pd.read_csv(ASHRAE_RAW, parse_dates=["timestamp"])
    print(f"  Raw rows: {len(df):,}")

    # 1. Electricity only (meter == 0)
    df = df[df["meter"] == 0].drop(columns=["meter"]).reset_index(drop=True)
    print(f"  After meter==0 filter: {len(df):,}")

    # 2. Remove timestamps before 2016-05-01
    df = df[df["timestamp"] >= "2016-05-01"].reset_index(drop=True)
    print(f"  After May-2016 cutoff: {len(df):,}")

    # 3. Drop buildings with >75% zero readings
    zero_frac = df.groupby("building_id")["meter_reading"].apply(
        lambda s: (s == 0).mean())
    high_zero = zero_frac[zero_frac > 0.75].index.tolist()
    df = df[~df["building_id"].isin(high_zero)].reset_index(drop=True)
    print(f"  Dropped {len(high_zero)} buildings with >75% zeros")

    # 4. Drop sparse buildings (<90% of expected hours)
    ts_min, ts_max = df["timestamp"].min(), df["timestamp"].max()
    expected_hours = int((ts_max - ts_min).total_seconds() / 3600) + 1
    counts = df.groupby("building_id").size()
    threshold = int(expected_hours * 0.90)
    sparse = counts[counts < threshold].index.tolist()
    df = df[~df["building_id"].isin(sparse)].reset_index(drop=True)
    print(f"  Dropped {len(sparse)} sparse buildings (<{threshold} hours)")

    available = sorted(df["building_id"].unique().tolist())
    print(f"  Available buildings after cleaning: {len(available)}")

    if len(available) < N_SELECT:
        print(f"  WARNING: Only {len(available)} buildings available, "
              f"need {N_SELECT}. Using all.")
        selected = available
    else:
        rng = np.random.RandomState(SEED)
        selected = sorted(rng.choice(available, N_SELECT, replace=False).tolist())
    print(f"  Selected {len(selected)} buildings")

    # Filter to selected buildings only
    df = df[df["building_id"].isin(selected)].reset_index(drop=True)

    # 5. Fill NaN with per-building forward-fill then 0
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    df["meter_reading"] = df.groupby("building_id")["meter_reading"].transform(
        lambda s: s.ffill().fillna(0))

    # 6. log1p transform
    df["meter_reading"] = np.log1p(df["meter_reading"])

    # 7. Building split: 35 train, 15 test
    rng = np.random.RandomState(SEED + 1)  # different seed from selection
    shuffled = np.array(selected)
    rng.shuffle(shuffled)
    test_bids = sorted(shuffled[:N_TEST].tolist())
    train_bids = sorted(shuffled[N_TEST:].tolist())
    print(f"  Building split: {len(train_bids)} train, {len(test_bids)} test")

    # 8. Temporal split for ALL buildings (70/20/10)
    per_building_splits = {}
    for bid in selected:
        n = int((df["building_id"] == bid).sum())
        train_end = int(n * TEMPORAL_TRAIN)
        val_end = int(n * (TEMPORAL_TRAIN + TEMPORAL_VAL))
        per_building_splits[str(bid)] = {
            "train": [0, train_end],
            "val": [train_end, val_end],
            "test": [val_end, n],
            "n_rows": n,
        }

    # 9. Save
    out_csv = os.path.join(ASHRAE_OUT, "ashrae_clean.csv")
    df[["building_id", "timestamp", "meter_reading"]].to_csv(out_csv, index=False)
    print(f"  Saved {out_csv} ({len(df):,} rows)")

    metadata = {
        "building_ids": selected,
        "train_building_ids": train_bids,
        "test_building_ids": test_bids,
        "n_buildings": len(selected),
        "n_train_buildings": len(train_bids),
        "n_test_buildings": len(test_bids),
        "temporal_ratios": {"train": TEMPORAL_TRAIN, "val": TEMPORAL_VAL,
                            "test": TEMPORAL_TEST},
        "per_building_splits": per_building_splits,
        "log_transformed": True,
    }
    meta_path = os.path.join(ASHRAE_OUT, "split_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {meta_path}")
    return metadata


def preprocess_lead():
    """Clean LEAD, select 50 buildings, split 35/15, log1p, save."""
    os.makedirs(LEAD_OUT, exist_ok=True)

    print("\n[LEAD] Loading raw CSV ...")
    df = pd.read_csv(LEAD_RAW, parse_dates=["timestamp"])
    print(f"  Raw rows: {len(df):,}")

    available = sorted(df["building_id"].unique().tolist())
    print(f"  Available buildings: {len(available)}")
    print(f"  Anomaly rate: {df['anomaly'].mean():.4f}")

    # Select 50 buildings
    if len(available) < N_SELECT:
        print(f"  WARNING: Only {len(available)} buildings, using all.")
        selected = available
    else:
        rng = np.random.RandomState(SEED + 10)  # distinct seed
        selected = sorted(rng.choice(available, N_SELECT, replace=False).tolist())
    print(f"  Selected {len(selected)} buildings")

    # Filter to selected
    df = df[df["building_id"].isin(selected)].reset_index(drop=True)

    # Fill NaN meter_reading with per-building median, then 0
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    df["meter_reading"] = df.groupby("building_id")["meter_reading"].transform(
        lambda s: s.fillna(s.median()).fillna(0))

    # log1p transform (NEW: standardize with ASHRAE)
    df["meter_reading"] = np.log1p(df["meter_reading"])

    # Building split: 35 train, 15 test
    rng = np.random.RandomState(SEED + 11)
    shuffled = np.array(selected)
    rng.shuffle(shuffled)
    test_bids = sorted(shuffled[:N_TEST].tolist())
    train_bids = sorted(shuffled[N_TEST:].tolist())
    print(f"  Building split: {len(train_bids)} train, {len(test_bids)} test")

    # Temporal split for ALL buildings (70/20/10)
    per_building_splits = {}
    anomaly_stats = {}
    for bid in selected:
        mask = df["building_id"] == bid
        n = int(mask.sum())
        train_end = int(n * TEMPORAL_TRAIN)
        val_end = int(n * (TEMPORAL_TRAIN + TEMPORAL_VAL))
        per_building_splits[str(bid)] = {
            "train": [0, train_end],
            "val": [train_end, val_end],
            "test": [val_end, n],
            "n_rows": n,
        }
        bdf = df.loc[mask]
        anomaly_stats[str(bid)] = {
            "total": int(n),
            "anomaly_count": int(bdf["anomaly"].sum()),
            "anomaly_rate": float(bdf["anomaly"].mean()),
        }

    # Save
    out_csv = os.path.join(LEAD_OUT, "lead_clean.csv")
    df[["building_id", "timestamp", "meter_reading", "anomaly"]].to_csv(
        out_csv, index=False)
    print(f"  Saved {out_csv} ({len(df):,} rows)")

    metadata = {
        "building_ids": selected,
        "train_building_ids": train_bids,
        "test_building_ids": test_bids,
        "n_buildings": len(selected),
        "n_train_buildings": len(train_bids),
        "n_test_buildings": len(test_bids),
        "temporal_ratios": {"train": TEMPORAL_TRAIN, "val": TEMPORAL_VAL,
                            "test": TEMPORAL_TEST},
        "per_building_splits": per_building_splits,
        "anomaly_stats": anomaly_stats,
        "log_transformed": True,
    }
    meta_path = os.path.join(LEAD_OUT, "split_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {meta_path}")
    return metadata


if __name__ == "__main__":
    ashrae_meta = preprocess_ashrae()
    lead_meta = preprocess_lead()
    print("\n[DONE] Preprocessing complete.")
    print(f"  ASHRAE: {ashrae_meta['n_train_buildings']} train / "
          f"{ashrae_meta['n_test_buildings']} test buildings (forecasting)")
    print(f"  LEAD:   {lead_meta['n_train_buildings']} train / "
          f"{lead_meta['n_test_buildings']} test buildings (anomaly)")
