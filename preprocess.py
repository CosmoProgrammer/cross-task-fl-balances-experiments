"""
Preprocessing for Multi-Task Federated Learning experiment.

Selects ASHRAE buildings (forecasting) and LEAD buildings (anomaly detection),
splits each cohort into a held-out test set + training clients, applies log1p,
and writes <task>_clean.csv + split_metadata.json.

The cohort size is configurable (cohort-scaling experiment). The DEFAULT cohort
is 50 buildings/dataset (35 train + 15 test) — identical to the original paper.

    Cohort-scaling design (nested, fixed test set):
    - numpy's RandomState.choice(replace=False) is permutation(len)[:k], so
      choice(available, N) is a prefix-SUPERSET of choice(available, 50) for the
      same seed. Larger cohorts therefore NEST on top of the base-50 cohort.
    - The test set is the base-50 cohort's held-out split and stays FIXED as the
      cohort grows (only training buildings are added). This keeps the eval target
      constant across the 50→100→200→400 ladder ("more training data" is the only
      thing that changes) and reproduces the committed 50-cohort split byte-for-byte.

Usage:
    python preprocess.py                  # default 50-building cohort
    python preprocess.py --cohort 100     # 100-building cohort (85 train + 15 test)
    COHORT_SIZE=200 python preprocess.py  # via env var
"""
import argparse
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

# ── Constants ──
SEED = 42
BASE_COHORT = 50       # anchor cohort that defines the fixed test set
N_TEST = 15            # held-out test buildings (FIXED across all cohort sizes)
TEMPORAL_TRAIN = 0.70
TEMPORAL_VAL = 0.20
TEMPORAL_TEST = 0.10


def processed_dir(dataset, cohort_size):
    """Cohort-specific output dir. Cohort 50 keeps the original (untagged) path
    so the committed 50-cohort data/results stay valid; larger cohorts get a
    `processed_c{N}` dir so cohorts coexist instead of clobbering each other."""
    sub = "processed" if cohort_size == BASE_COHORT else f"processed_c{cohort_size}"
    return os.path.join(PROJECT_ROOT, "data", dataset, sub)


def select_buildings(available, n_select, sel_seed, split_seed):
    """Nested cohort selection with a FIXED held-out test set.

    Returns (selected, train_bids, test_bids). At n_select == BASE_COHORT this is
    byte-identical (as sets) to the original non-nested split; for larger cohorts
    the selection is a prefix-superset and the test set is unchanged."""
    n_select = min(n_select, len(available))
    base_n = min(BASE_COHORT, len(available))

    # Cohort selection — nested: a prefix of the same permutation for any n_select.
    selected = sorted(np.random.RandomState(sel_seed)
                      .choice(available, n_select, replace=False).tolist())

    # Canonical base cohort -> defines the FIXED test set (same shuffle as before).
    base = sorted(np.random.RandomState(sel_seed)
                  .choice(available, base_n, replace=False).tolist())
    shuffled = np.array(base)
    np.random.RandomState(split_seed).shuffle(shuffled)
    test_bids = sorted(shuffled[:N_TEST].tolist())

    test_set = set(test_bids)
    train_bids = sorted(b for b in selected if b not in test_set)
    return selected, train_bids, test_bids


def preprocess_ashrae(cohort_size=BASE_COHORT):
    """Clean ASHRAE, select `cohort_size` buildings (nested + fixed 15 test),
    temporal-split, log1p, save."""
    out_dir = processed_dir("ashrae", cohort_size)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[ASHRAE] cohort_size={cohort_size} -> {out_dir}")
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

    if len(available) < cohort_size:
        print(f"  WARNING: Only {len(available)} buildings available, "
              f"need {cohort_size}. Using all.")

    # Nested cohort selection + FIXED test set (seeds SEED / SEED+1 as before).
    selected, train_bids, test_bids = select_buildings(
        available, cohort_size, sel_seed=SEED, split_seed=SEED + 1)
    print(f"  Selected {len(selected)} buildings")

    # Filter to selected buildings only
    df = df[df["building_id"].isin(selected)].reset_index(drop=True)

    # 5. Fill NaN with per-building forward-fill then 0
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    df["meter_reading"] = df.groupby("building_id")["meter_reading"].transform(
        lambda s: s.ffill().fillna(0))

    # 6. log1p transform
    df["meter_reading"] = np.log1p(df["meter_reading"])

    # 7. Building split (fixed 15 test, rest train)
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
    out_csv = os.path.join(out_dir, "ashrae_clean.csv")
    df[["building_id", "timestamp", "meter_reading"]].to_csv(out_csv, index=False)
    print(f"  Saved {out_csv} ({len(df):,} rows)")

    metadata = {
        "cohort_size": cohort_size,
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
    meta_path = os.path.join(out_dir, "split_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {meta_path}")
    return metadata


def preprocess_lead(cohort_size=BASE_COHORT):
    """Clean LEAD, select `cohort_size` buildings (nested + fixed 15 test),
    temporal-split, log1p, save."""
    out_dir = processed_dir("lead", cohort_size)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[LEAD] cohort_size={cohort_size} -> {out_dir}")
    print("[LEAD] Loading raw CSV ...")
    df = pd.read_csv(LEAD_RAW, parse_dates=["timestamp"])
    print(f"  Raw rows: {len(df):,}")

    available = sorted(df["building_id"].unique().tolist())
    print(f"  Available buildings: {len(available)}")
    print(f"  Anomaly rate: {df['anomaly'].mean():.4f}")

    # Select `cohort_size` buildings (nested + FIXED 15 test; seeds SEED+10 / SEED+11)
    if len(available) < cohort_size:
        print(f"  WARNING: Only {len(available)} buildings, using all.")
    selected, train_bids, test_bids = select_buildings(
        available, cohort_size, sel_seed=SEED + 10, split_seed=SEED + 11)
    print(f"  Selected {len(selected)} buildings")

    # Filter to selected
    df = df[df["building_id"].isin(selected)].reset_index(drop=True)

    # Fill NaN meter_reading with per-building median, then 0
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    df["meter_reading"] = df.groupby("building_id")["meter_reading"].transform(
        lambda s: s.fillna(s.median()).fillna(0))

    # log1p transform (NEW: standardize with ASHRAE)
    df["meter_reading"] = np.log1p(df["meter_reading"])

    # Building split (fixed 15 test, rest train)
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
    out_csv = os.path.join(out_dir, "lead_clean.csv")
    df[["building_id", "timestamp", "meter_reading", "anomaly"]].to_csv(
        out_csv, index=False)
    print(f"  Saved {out_csv} ({len(df):,} rows)")

    metadata = {
        "cohort_size": cohort_size,
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
    meta_path = os.path.join(out_dir, "split_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {meta_path}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ASHRAE + LEAD cohorts.")
    parser.add_argument("--cohort", type=int,
                        default=int(os.environ.get("COHORT_SIZE", BASE_COHORT)),
                        help="Buildings per dataset (default 50; ladder 100/200/400). "
                             "LEAD has 400 buildings -> caps the symmetric cohort at 400.")
    args = parser.parse_args()

    ashrae_meta = preprocess_ashrae(args.cohort)
    lead_meta = preprocess_lead(args.cohort)
    print("\n[DONE] Preprocessing complete.")
    print(f"  ASHRAE: {ashrae_meta['n_train_buildings']} train / "
          f"{ashrae_meta['n_test_buildings']} test buildings (forecasting)")
    print(f"  LEAD:   {lead_meta['n_train_buildings']} train / "
          f"{lead_meta['n_test_buildings']} test buildings (anomaly)")
