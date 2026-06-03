"""
Day 5 — SHAP TreeExplainer for per-transaction attribution.
Loads the realistic XGBoost model, computes SHAP values on the test set,
generates waterfall plots for sample fraud transactions, and validates on
10 high-confidence fraud cases.

Expected runtime: ~2-4 minutes.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import xgboost as xgb
from pathlib import Path

FEATURES_PATH  = Path("Dataset/features.parquet")
MODEL_PATH     = Path("models/xgboost_fraud_realistic.json")
THRESHOLD_PATH = Path("models/threshold_realistic.txt")
OUTPUT_DIR     = Path("notebooks/eda_output/shap")

FEATURE_COLS = [
    "step", "type_encoded", "amount", "log_amount",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "velocity_cumcount", "velocity_1hr", "velocity_3hr", "velocity_24hr",
]

FEATURE_LABELS = {
    "step":              "Hour of transaction",
    "type_encoded":      "Transaction type",
    "amount":            "Amount ($)",
    "log_amount":        "Log amount",
    "oldbalanceOrg":     "Origin balance (before)",
    "newbalanceOrig":    "Origin balance (after)",
    "oldbalanceDest":    "Dest balance (before)",
    "newbalanceDest":    "Dest balance (after)",
    "velocity_cumcount": "Cumulative txn count",
    "velocity_1hr":      "Txns same hour",
    "velocity_3hr":      "Txns same 3hr window",
    "velocity_24hr":     "Txns same day",
}


def print_section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ── Load model + data ─────────────────────────────────────────────────────────

def load_artifacts():
    print_section("1 / 4  —  Load model + data")
    t = time.time()

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    threshold = float(THRESHOLD_PATH.read_text().strip())

    df = pd.read_parquet(FEATURES_PATH)
    X  = df[FEATURE_COLS].values.astype("float32")
    y  = df["isFraud"].values

    print(f"  Model     : {MODEL_PATH}")
    print(f"  Threshold : {threshold:.4f}")
    print(f"  Rows      : {len(df):,}  fraud={y.sum():,}")
    print(f"  Loaded in {time.time()-t:.1f}s")

    return model, threshold, df, X, y


# ── SHAP values ───────────────────────────────────────────────────────────────

def compute_shap(model: xgb.XGBClassifier, X: np.ndarray, sample_size: int = 5000):
    """
    Compute SHAP values using TreeExplainer (exact, not approximate).
    Uses a stratified sample for the summary plot — full dataset would take too long to plot.
    """
    print_section("2 / 4  —  SHAP TreeExplainer")
    t = time.time()

    explainer = shap.TreeExplainer(model)
    print(f"  Explainer created in {time.time()-t:.1f}s")
    print(f"  Computing SHAP values on {sample_size:,}-row sample...")

    t2 = time.time()
    shap_values = explainer.shap_values(X[:sample_size])
    print(f"  SHAP values computed in {time.time()-t2:.1f}s")
    print(f"  Expected value (base rate): {explainer.expected_value:.4f}")

    return explainer, shap_values


# ── Summary bar plot ──────────────────────────────────────────────────────────

def plot_summary(shap_values: np.ndarray, X_sample: np.ndarray):
    print_section("3 / 4  —  Global feature importance (summary plot)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = list(FEATURE_LABELS.values())

    # Mean absolute SHAP per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_shap,
    }).sort_values("mean_abs_shap", ascending=True)

    print("  Feature importance (mean |SHAP|):")
    for _, row in importance_df.sort_values("mean_abs_shap", ascending=False).iterrows():
        bar = "█" * int(row["mean_abs_shap"] * 200)
        print(f"    {row['feature']:<30} {row['mean_abs_shap']:.4f}  {bar}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tomato" if v > importance_df["mean_abs_shap"].median() else "steelblue"
              for v in importance_df["mean_abs_shap"]]
    ax.barh(importance_df["feature"], importance_df["mean_abs_shap"],
            color=colors, edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance — Fraud Model (Realistic)")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    out = OUTPUT_DIR / "shap_feature_importance.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Saved: {out}")

    # SHAP beeswarm summary
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False, plot_size=None,
    )
    out2 = OUTPUT_DIR / "shap_beeswarm.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2}")


# ── Waterfall plots for fraud samples ─────────────────────────────────────────

def plot_waterfalls(
    explainer,
    model: xgb.XGBClassifier,
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_samples: int = 10,
):
    print_section("4 / 4  —  Waterfall plots — 10 fraud samples")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_proba = model.predict_proba(X)[:, 1]
    feature_names = list(FEATURE_LABELS.values())

    # Get high-confidence fraud predictions
    fraud_mask    = (y == 1) & (y_proba >= threshold)
    fraud_indices = np.where(fraud_mask)[0]

    print(f"  Fraud txns in dataset          : {(y==1).sum():,}")
    print(f"  Correctly predicted (TP)       : {fraud_mask.sum():,}")
    print(f"  Sampling {n_samples} for waterfall plots...")

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(fraud_indices, size=min(n_samples, len(fraud_indices)), replace=False)

    for i, idx in enumerate(sample_idx):
        sv = explainer.shap_values(X[idx:idx+1])[0]
        base = explainer.expected_value
        row  = df.iloc[idx]

        # Sort by absolute impact
        order     = np.argsort(np.abs(sv))[::-1]
        top_n     = 8
        top_idx   = order[:top_n]
        top_names = [feature_names[j] for j in top_idx]
        top_shap  = sv[top_idx]
        top_vals  = X[idx][top_idx]

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["tomato" if v > 0 else "steelblue" for v in top_shap]
        bars = ax.barh(range(top_n), top_shap[::-1], color=colors[::-1], edgecolor="white")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(
            [f"{top_names[::-1][j]}  = {top_vals[::-1][j]:.2f}" for j in range(top_n)],
            fontsize=9,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on fraud probability)")
        ax.set_title(
            f"Fraud Sample {i+1}  |  Risk score: {y_proba[idx]:.4f}  |  Amount: ${row['amount']:,.0f}",
            fontsize=10,
        )
        red_patch  = mpatches.Patch(color="tomato",    label="Increases fraud probability")
        blue_patch = mpatches.Patch(color="steelblue", label="Decreases fraud probability")
        ax.legend(handles=[red_patch, blue_patch], fontsize=8, loc="lower right")
        plt.tight_layout()
        out = OUTPUT_DIR / f"waterfall_fraud_{i+1:02d}.png"
        plt.savefig(out, dpi=150)
        plt.close()

        # Print summary to terminal
        print(f"\n  Sample {i+1:>2} | score={y_proba[idx]:.4f} | amount=${row['amount']:>12,.0f}")
        for j in range(min(4, top_n)):
            direction = "▲ fraud" if top_shap[j] > 0 else "▼ legit"
            print(f"    {direction}  {top_names[j]:<30} SHAP={top_shap[j]:+.4f}  val={top_vals[j]:.2f}")

    print(f"\n  Waterfall plots saved to {OUTPUT_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "="*55)
    print("  FraudLens — SHAP Explainability Pipeline")
    print("="*55)

    model, threshold, df, X, y = load_artifacts()

    explainer, shap_values = compute_shap(model, X, sample_size=5000)

    plot_summary(shap_values, X[:5000])

    plot_waterfalls(explainer, model, df, X, y, threshold, n_samples=10)

    print("\n" + "="*55)
    print("  SHAP complete.")
    print(f"  Charts: {OUTPUT_DIR}/")
    print("  Next  : Day 6 — FastAPI /predict + /explain endpoints")
    print("="*55 + "\n")


if __name__ == "__main__":
    run()
