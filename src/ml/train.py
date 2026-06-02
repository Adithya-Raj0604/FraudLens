"""
Day 4 — XGBoost training with SMOTE on the engineered feature set.
Reads Dataset/features.parquet, applies SMOTE, trains XGBoost,
tunes threshold on precision-recall curve, saves model to models/.

Expected runtime: ~5-10 minutes (SMOTE on ~2.7M rows is the bottleneck).
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

FEATURES_PATH = Path("Dataset/features.parquet")
MODEL_DIR     = Path("models")
OUTPUT_DIR    = Path("notebooks/eda_output")

TARGET_COL = "isFraud"

# Full feature set (includes PaySim balance leakage — near-perfect but unrealistic)
FEATURE_COLS_FULL = [
    "step", "type_encoded", "amount", "log_amount",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "orig_balance_diff", "dest_balance_diff",
    "orig_zero_after", "dest_unchanged",
    "velocity_cumcount", "velocity_1hr", "velocity_3hr", "velocity_24hr",
]

# Realistic feature set — excludes balance discrepancy flags that leak the label
# In production you wouldn't know these were fraudulently manipulated at tx time
FEATURE_COLS_REALISTIC = [
    "step", "type_encoded", "amount", "log_amount",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "velocity_cumcount", "velocity_1hr", "velocity_3hr", "velocity_24hr",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self.start = time.time()
        self.step_start = time.time()

    def tick(self, label: str):
        elapsed_step  = time.time() - self.step_start
        elapsed_total = time.time() - self.start
        print(f"  ✓ {label:<45} {elapsed_step:>6.1f}s  (total {elapsed_total:.0f}s)")
        self.step_start = time.time()

    def done(self):
        print(f"\n  Total time: {(time.time()-self.start)/60:.1f} min\n")


def print_section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_data(path: Path, feature_cols: list) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    print_section("1 / 6  —  Load features")
    t = time.time()
    df = pd.read_parquet(path)
    X = df[feature_cols].values.astype("float32")
    y = df[TARGET_COL].values.astype("int8")
    print(f"  Rows   : {len(df):,}")
    print(f"  Fraud  : {y.sum():,}  ({y.mean()*100:.4f}%)")
    print(f"  Loaded in {time.time()-t:.1f}s")
    return X, y, df


def split_data(X, y):
    print_section("2 / 6  —  Train / test split  (80/20, stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train : {len(X_train):,}  fraud={y_train.sum():,}")
    print(f"  Test  : {len(X_test):,}   fraud={y_test.sum():,}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train) -> tuple[np.ndarray, np.ndarray]:
    print_section("3 / 6  —  SMOTE oversampling")
    print(f"  Before: {len(X_train):,} rows  fraud={y_train.sum():,}  ({y_train.mean()*100:.4f}%)")
    print("  Running SMOTE (this takes 3-7 min on ~2.1M rows)...")
    t = time.time()
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  After : {len(X_res):,} rows  fraud={y_res.sum():,}  ({y_res.mean()*100:.2f}%)")
    print(f"  SMOTE done in {(time.time()-t)/60:.1f} min")
    return X_res, y_res


def train_model(X_res, y_res) -> xgb.XGBClassifier:
    print_section("4 / 6  —  Train XGBoost")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="aucpr",        # area under precision-recall curve
        tree_method="hist",         # fast histogram method
        n_jobs=-1,
        random_state=42,
    )
    t = time.time()
    print("  Training XGBoost (300 trees, hist method)...")
    model.fit(X_res, y_res, verbose=False)
    print(f"  Training done in {time.time()-t:.1f}s")
    return model


def tune_threshold(model: xgb.XGBClassifier, X_test, y_test) -> float:
    """
    Find the threshold that maximises F1 on the fraud class.
    Default 0.5 is rarely optimal on imbalanced data.
    """
    print_section("5 / 6  —  Threshold tuning + evaluation")

    y_proba = model.predict_proba(X_test)[:, 1]

    # ROC-AUC and PR-AUC
    roc  = roc_auc_score(y_test, y_proba)
    prauc = average_precision_score(y_test, y_proba)
    print(f"  ROC-AUC  : {roc:.4f}")
    print(f"  PR-AUC   : {prauc:.4f}")

    # Precision-recall curve — find best F1 threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx   = np.argmax(f1_scores)
    best_thresh = float(thresholds[best_idx])
    best_f1    = float(f1_scores[best_idx])

    print(f"\n  Best threshold : {best_thresh:.4f}")
    print(f"  Best F1        : {best_f1:.4f}")
    print(f"  Precision      : {precision[best_idx]:.4f}")
    print(f"  Recall         : {recall[best_idx]:.4f}")

    # Classification report at best threshold
    y_pred = (y_proba >= best_thresh).astype(int)
    print(f"\n  Classification report (threshold={best_thresh:.3f}):")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion matrix:")
    print(f"    TN={tn:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TP={tp:,}")

    # Also show at default 0.5 for comparison
    y_pred_05 = (y_proba >= 0.5).astype(int)
    r05 = classification_report(y_test, y_pred_05, output_dict=True)
    print(f"\n  At default threshold=0.50:")
    print(f"    Fraud recall={r05['1']['recall']:.4f}  precision={r05['1']['precision']:.4f}  F1={r05['1']['f1-score']:.4f}")

    # Save precision-recall curve
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="steelblue", lw=2, label=f"PR curve (AUC={prauc:.3f})")
    ax.scatter(recall[best_idx], precision[best_idx], color="tomato", zorder=5,
               s=80, label=f"Best F1 threshold={best_thresh:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Fraud Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=150)
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR}/precision_recall_curve.png")

    return best_thresh


def save_artifacts(model: xgb.XGBClassifier, threshold: float, suffix: str = ""):
    print_section("6 / 6  —  Save model artifacts")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path     = MODEL_DIR / f"xgboost_fraud{suffix}.json"
    threshold_path = MODEL_DIR / f"threshold{suffix}.txt"

    model.save_model(str(model_path))
    threshold_path.write_text(str(threshold))

    print(f"  Model     : {model_path}")
    print(f"  Threshold : {threshold_path}  (value={threshold:.4f})")


def run_pipeline(feature_cols: list, label: str, suffix: str):
    print(f"\n  Feature set : {label}  ({len(feature_cols)} features)")
    X, y, _          = load_data(FEATURES_PATH, feature_cols)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_res, y_res     = apply_smote(X_train, y_train)
    model            = train_model(X_res, y_res)
    best_threshold   = tune_threshold(model, X_test, y_test)
    save_artifacts(model, best_threshold, suffix=suffix)
    return model, best_threshold


def run():
    print("\n" + "="*55)
    print("  FraudLens — XGBoost Training Pipeline")
    print("="*55)

    # ── Run 1: Full features (includes leaky balance flags) ───────────────────
    print("\n\n>>> RUN 1: FULL feature set (includes PaySim balance leakage)")
    print("    Expect near-perfect metrics — this is a known PaySim artifact.")
    run_pipeline(FEATURE_COLS_FULL, "Full (with leakage)", suffix="")

    # ── Run 2: Realistic features (no leaky flags) ────────────────────────────
    print("\n\n>>> RUN 2: REALISTIC feature set (balance leakage features removed)")
    print("    These metrics represent real-world difficulty more honestly.")
    run_pipeline(FEATURE_COLS_REALISTIC, "Realistic (no leakage)", suffix="_realistic")

    print("\n" + "="*55)
    print("  Both models saved. Next: Day 5 — SHAP explainability")
    print("  Use xgboost_fraud_realistic.json for the live API.")
    print("="*55 + "\n")


if __name__ == "__main__":
    run()
