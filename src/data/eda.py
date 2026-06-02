"""
Day 2 — Dask EDA pipeline over full 6.3M rows.
Covers: class distribution, transaction types, balance anomalies.
Run after ingest.py confirms chunked ingestion works.
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

DATASET_PATH = Path("Dataset/paysim.csv")
OUTPUT_DIR = Path("notebooks/eda_output")

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


def load_dask(path: Path = DATASET_PATH) -> dd.DataFrame:
    print(f"Loading {path} with Dask...")
    df = dd.read_csv(path, dtype=DTYPES)
    print(f"Partitions: {df.npartitions}")
    return df


def class_distribution(df: dd.DataFrame) -> None:
    print("\n--- Class Distribution ---")
    counts = df["isFraud"].value_counts().compute()
    total = counts.sum()
    for label, count in counts.items():
        name = "Fraud" if label == 1 else "Legitimate"
        print(f"  {name:>12}: {count:>9,}  ({count/total*100:.4f}%)")


def transaction_type_breakdown(df: dd.DataFrame) -> None:
    print("\n--- Transaction Types ---")
    type_fraud = (
        df.groupby("type")["isFraud"]
        .agg(["sum", "count"])
        .compute()
        .rename(columns={"sum": "fraud_count", "count": "total"})
    )
    type_fraud["fraud_rate_%"] = (type_fraud["fraud_count"] / type_fraud["total"] * 100).round(4)
    type_fraud = type_fraud.sort_values("fraud_count", ascending=False)
    print(type_fraud.to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    type_fraud["fraud_rate_%"].plot(kind="bar", ax=ax, color="tomato", edgecolor="black")
    ax.set_title("Fraud Rate by Transaction Type")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "fraud_rate_by_type.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/fraud_rate_by_type.png")


def balance_anomalies(df: dd.DataFrame) -> None:
    """
    Flag transactions where orig balance drops to 0 AND dest balance doesn't change.
    Classic PaySim fraud signature.
    """
    print("\n--- Balance Anomaly Detection ---")

    # Filter to fraud-relevant types only
    relevant = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]

    # Anomaly: orig balance wiped out, dest balance unchanged
    anomaly = relevant[
        (relevant["newbalanceOrig"] == 0) &
        (relevant["oldbalanceOrg"] > 0) &
        (relevant["newbalanceDest"] == relevant["oldbalanceDest"])
    ]

    anomaly_count = anomaly.shape[0].compute()
    relevant_count = relevant.shape[0].compute()
    anomaly_fraud = anomaly["isFraud"].sum().compute()

    print(f"  TRANSFER + CASH_OUT rows : {relevant_count:,}")
    print(f"  Balance anomaly rows     : {anomaly_count:,}")
    print(f"  Anomaly rows = fraud     : {anomaly_fraud:,}")
    print(f"  Precision of anomaly flag: {anomaly_fraud/anomaly_count*100:.2f}%")


def amount_distribution(df: dd.DataFrame) -> None:
    print("\n--- Amount Distribution (fraud vs legit) ---")

    fraud_amounts = df[df["isFraud"] == 1]["amount"].compute()
    legit_sample = df[df["isFraud"] == 0]["amount"].sample(frac=0.01, random_state=42).compute()

    print(f"  Fraud amount — mean: ${fraud_amounts.mean():,.2f}  median: ${fraud_amounts.median():,.2f}  max: ${fraud_amounts.max():,.2f}")
    print(f"  Legit amount — mean: ${legit_sample.mean():,.2f}  median: ${legit_sample.median():,.2f}  max: ${legit_sample.max():,.2f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(legit_sample.clip(upper=1_000_000), bins=60, alpha=0.6, label="Legitimate (1% sample)", color="steelblue")
    ax.hist(fraud_amounts.clip(upper=1_000_000), bins=60, alpha=0.8, label="Fraud", color="tomato")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.set_title("Amount Distribution — Fraud vs Legitimate (capped at $1M)")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "amount_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/amount_distribution.png")


def run_eda() -> None:
    df = load_dask()

    class_distribution(df)
    transaction_type_breakdown(df)
    balance_anomalies(df)
    amount_distribution(df)

    print("\nEDA complete. Charts saved to notebooks/eda_output/")


if __name__ == "__main__":
    run_eda()
