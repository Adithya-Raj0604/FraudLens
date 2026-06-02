"""
Day 1 — Chunked ingestion with memory profiling.
Reads paysim.csv in 500k-row chunks, profiles peak memory, and prints a summary.
"""

import pandas as pd
import numpy as np
import os
import time
import tracemalloc
from pathlib import Path

DATASET_PATH = Path("Dataset/paysim.csv")
CHUNK_SIZE = 500_000


def profile_chunk(chunk: pd.DataFrame, chunk_num: int, elapsed: float, peak_mb: float) -> dict:
    fraud_count = chunk["isFraud"].sum()
    return {
        "chunk": chunk_num,
        "rows": len(chunk),
        "fraud_rows": int(fraud_count),
        "fraud_pct": round(fraud_count / len(chunk) * 100, 4),
        "peak_memory_mb": round(peak_mb, 2),
        "elapsed_s": round(elapsed, 2),
    }


def ingest_chunked(path: Path = DATASET_PATH, chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    print(f"Ingesting {path} in {chunk_size:,}-row chunks...\n")

    stats = []
    total_rows = 0
    total_fraud = 0

    for chunk_num, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size), start=1):
        tracemalloc.start()
        t0 = time.time()

        # Basic validation
        assert "isFraud" in chunk.columns, "isFraud column missing"

        elapsed = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        row_stat = profile_chunk(chunk, chunk_num, elapsed, peak_mb)
        stats.append(row_stat)

        total_rows += len(chunk)
        total_fraud += int(chunk["isFraud"].sum())

        print(
            f"  Chunk {chunk_num:>2}: {len(chunk):>7,} rows | "
            f"fraud={row_stat['fraud_rows']:>4} ({row_stat['fraud_pct']:.4f}%) | "
            f"peak={peak_mb:.1f}MB | {elapsed:.2f}s"
        )

    print(f"\nTotal rows ingested : {total_rows:,}")
    print(f"Total fraud rows    : {total_fraud:,}")
    print(f"Overall fraud rate  : {total_fraud / total_rows * 100:.4f}%")
    print(f"Chunks processed    : {len(stats)}")

    return pd.DataFrame(stats)


def print_schema(path: Path = DATASET_PATH) -> None:
    sample = pd.read_csv(path, nrows=5)
    print("\n--- Schema (first 5 rows) ---")
    print(sample.dtypes.to_string())
    print("\nSample:")
    print(sample.to_string())


if __name__ == "__main__":
    print_schema()
    print("\n--- Chunked Ingestion Memory Profile ---\n")
    stats_df = ingest_chunked()
    print("\n--- Per-Chunk Stats ---")
    print(stats_df.to_string(index=False))
