"""
Day 3 — Feature engineering over the full Dask dataframe.
Adds: balance discrepancy, log-transformed amount, rolling velocity windows (1hr/3hr/24hr).
Outputs a parquet file ready for model training.

Expected runtime: ~3-6 minutes on 6.3M rows (filters to ~2.7M TRANSFER+CASH_OUT first).
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
import time
from pathlib import Path

DATASET_PATH = Path("Dataset/paysim.csv")
OUTPUT_PATH = Path("Dataset/features.parquet")

DTYPES = {
    "step": "int32",
    "type": "object",
    "amount": "float64",
    "nameOrig": "object",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "object",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

STEPS = [
    "Load CSV with Dask",
    "Filter to TRANSFER + CASH_OUT",
    "Compute to pandas",
    "Balance features",
    "Log amount",
    "Type encoding",
    "Velocity features",
    "Select & write parquet",
    "Sanity check",
]


class Timer:
    def __init__(self, total_steps: int):
        self.total = total_steps
        self.current = 0
        self.start = time.time()
        self.step_start = time.time()

    def tick(self, label: str):
        self.current += 1
        elapsed_step = time.time() - self.step_start
        elapsed_total = time.time() - self.start
        pct = int(self.current / self.total * 100)
        bar = ("█" * (pct // 5)).ljust(20)
        print(f"  [{bar}] {pct:>3}%  Step {self.current}/{self.total}: {label}  ({elapsed_step:.1f}s | total {elapsed_total:.0f}s)")
        self.step_start = time.time()

    def done(self):
        total = time.time() - self.start
        print(f"\n  Done in {total:.1f}s ({total/60:.1f} min)")


def engineer_features(
    input_path: Path = DATASET_PATH,
    output_path: Path = OUTPUT_PATH,
) -> None:
    t = Timer(len(STEPS))
    print(f"\nFeature engineering pipeline — {len(STEPS)} steps\n")

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    df_dask = dd.read_csv(input_path, dtype=DTYPES)
    t.tick(STEPS[0])

    # ── Step 2: Filter ────────────────────────────────────────────────────────
    # Fraud only exists in TRANSFER and CASH_OUT — drop everything else now
    df_dask = df_dask[df_dask["type"].isin(["TRANSFER", "CASH_OUT"])]
    t.tick(STEPS[1])

    # ── Step 3: Compute to pandas ─────────────────────────────────────────────
    # After filtering, ~2.7M rows — fits in RAM (~500MB), far faster than Dask map_partitions
    print("       (reading ~2.7M rows into memory...)")
    df: pd.DataFrame = df_dask.compute()
    df = df.reset_index(drop=True)
    t.tick(STEPS[2])

    # ── Step 4: Balance features ──────────────────────────────────────────────
    # Classic PaySim fraud signature: origin wiped, destination unchanged
    df["orig_balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    df["dest_balance_diff"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
    df["orig_zero_after"]   = (df["newbalanceOrig"] == 0).astype("int8")
    df["dest_unchanged"]    = (df["newbalanceDest"] == df["oldbalanceDest"]).astype("int8")
    t.tick(STEPS[3])

    # ── Step 5: Log amount ────────────────────────────────────────────────────
    df["log_amount"] = np.log1p(df["amount"])
    t.tick(STEPS[4])

    # ── Step 6: Type encoding ─────────────────────────────────────────────────
    type_map = {"TRANSFER": 1, "CASH_OUT": 2}
    df["type_encoded"] = df["type"].map(type_map).astype("int8")
    t.tick(STEPS[5])

    # ── Step 7: Velocity features ─────────────────────────────────────────────
    # Sort by account + time, then use cumcount (O(n log n), not O(n²))
    # velocity_Xhr = how many txns this account has made up to this step in that window
    print("       (sorting by account + step...)")
    df = df.sort_values(["nameOrig", "step"]).reset_index(drop=True)

    # Cumulative txn count per origin account (proxy for lifetime velocity)
    df["velocity_cumcount"] = df.groupby("nameOrig").cumcount().astype("int32")

    # Txns by same account in the same 1-step window (hour)
    df["velocity_1hr"] = (
        df.groupby(["nameOrig", "step"])["step"]
        .transform("count")
        .astype("int32")
    )

    # Txns by same account in same 3-step bucket
    df["step_3hr_bucket"] = (df["step"] // 3).astype("int32")
    df["velocity_3hr"] = (
        df.groupby(["nameOrig", "step_3hr_bucket"])["step"]
        .transform("count")
        .astype("int32")
    )

    # Txns by same account in same 24-step bucket (day)
    df["step_24hr_bucket"] = (df["step"] // 24).astype("int32")
    df["velocity_24hr"] = (
        df.groupby(["nameOrig", "step_24hr_bucket"])["step"]
        .transform("count")
        .astype("int32")
    )
    t.tick(STEPS[6])

    # ── Step 8: Select columns & write parquet ────────────────────────────────
    feature_cols = [
        "step", "type_encoded", "amount", "log_amount",
        "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        "orig_balance_diff", "dest_balance_diff",
        "orig_zero_after", "dest_unchanged",
        "velocity_cumcount", "velocity_1hr", "velocity_3hr", "velocity_24hr",
        "isFraud",
    ]
    df = df[feature_cols]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    t.tick(STEPS[7])

    # ── Step 9: Sanity check ──────────────────────────────────────────────────
    check = pd.read_parquet(output_path)
    fraud_count = check["isFraud"].sum()
    total = len(check)
    t.tick(STEPS[8])

    t.done()

    print(f"\n  Output path  : {output_path}")
    print(f"  Rows         : {total:,}")
    print(f"  Fraud rows   : {fraud_count:,}")
    print(f"  Fraud rate   : {fraud_count/total*100:.4f}%")
    print(f"  Columns      : {list(check.columns)}")
    print()


if __name__ == "__main__":
    engineer_features()
