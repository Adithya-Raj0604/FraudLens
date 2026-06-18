"""
Domain inference layer — the single source of truth for the fraud model.

Owns the feature schema, the loaded model/explainer state, and the core
predict + SHAP-explain logic. Both the FastAPI routes (src/api/main.py) and the
LangGraph agent tools (src/agent/tools.py) depend on THIS module, not on each
other. Keep this free of FastAPI/web concerns so the dependency direction stays
web → domain (never the reverse).
"""

from pathlib import Path

import numpy as np
import shap
import xgboost as xgb

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_PATH     = Path("models/xgboost_fraud_realistic.json")
THRESHOLD_PATH = Path("models/threshold_realistic.txt")

# ── Feature schema ────────────────────────────────────────────────────────────

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

TYPE_MAP = {"TRANSFER": 1, "CASH_OUT": 2}

# Agent escalates to explain_prediction at or above this risk score.
EXPLAIN_THRESHOLD = 0.4


# ── Model state (loaded once at startup) ──────────────────────────────────────

class ModelState:
    model:     xgb.XGBClassifier = None
    explainer: shap.TreeExplainer = None
    threshold: float = 0.5


state = ModelState()


def load_model_state() -> None:
    """Load the XGBoost model, SHAP explainer, and threshold into `state`."""
    state.model = xgb.XGBClassifier()
    state.model.load_model(str(MODEL_PATH))
    state.explainer = shap.TreeExplainer(state.model)
    state.threshold = float(THRESHOLD_PATH.read_text().strip())


# ── Feature engineering ───────────────────────────────────────────────────────

def build_feature_vector(tx: dict) -> np.ndarray:
    """
    Convert a transaction dict into the model's feature vector.
    Raises ValueError on an unsupported transaction type (callers translate this
    to whatever error their layer needs — e.g. an HTTP 422).
    """
    tx_type = tx["type"]
    if tx_type not in TYPE_MAP:
        raise ValueError(f"type must be TRANSFER or CASH_OUT, got '{tx_type}'")

    type_encoded = TYPE_MAP[tx_type]
    log_amount   = float(np.log1p(tx["amount"]))

    values = [
        tx["step"], type_encoded, tx["amount"], log_amount,
        tx["oldbalanceOrg"], tx["newbalanceOrig"],
        tx["oldbalanceDest"], tx["newbalanceDest"],
        tx["velocity_cumcount"], tx["velocity_1hr"],
        tx["velocity_3hr"], tx["velocity_24hr"],
    ]
    return np.array(values, dtype="float32").reshape(1, -1)


def risk_label(score: float, threshold: float) -> str:
    if score >= threshold:
        return "HIGH"
    if score >= threshold * 0.6:
        return "MEDIUM"
    return "LOW"


def flagged_features(shap_vals: np.ndarray, top_n: int = 3) -> list[str]:
    """Return the top N feature labels driving fraud risk upward."""
    positive = [(FEATURE_LABELS[FEATURE_COLS[i]], shap_vals[i])
                for i in range(len(FEATURE_COLS)) if shap_vals[i] > 0]
    positive.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in positive[:top_n]]


def _summary(top_label: str, top_shap: float, risk_score: float) -> str:
    direction = "increases" if top_shap > 0 else "decreases"
    return (
        f"Primary driver: '{top_label}' {direction} fraud probability "
        f"by {abs(top_shap):.4f} SHAP units. "
        f"Overall risk score: {risk_score:.4f}."
    )


# ── Core operations (used by both the API and the agent tools) ────────────────

def predict(tx: dict) -> dict:
    """Run the fraud model on a transaction dict and return scores + flags."""
    X          = build_feature_vector(tx)
    risk_score = float(state.model.predict_proba(X)[0][1])
    sv         = state.explainer.shap_values(X)[0]

    return {
        "risk_score":       round(risk_score, 4),
        "predicted_class":  int(risk_score >= state.threshold),
        "risk_label":       risk_label(risk_score, state.threshold),
        "threshold":        state.threshold,
        "flagged_features": flagged_features(sv),
    }


def explain(tx: dict) -> dict:
    """Run SHAP attribution on a transaction dict, sorted by absolute impact."""
    X          = build_feature_vector(tx)
    risk_score = float(state.model.predict_proba(X)[0][1])
    sv         = state.explainer.shap_values(X)[0]
    base_value = float(state.explainer.expected_value)

    features = sorted(
        [
            {
                "feature":    col,
                "label":      FEATURE_LABELS[col],
                "shap_value": round(float(val), 6),
                "raw_value":  round(float(X[0][i]), 4),
                "direction":  "increases_fraud" if val > 0 else "decreases_fraud",
            }
            for i, (col, val) in enumerate(zip(FEATURE_COLS, sv))
        ],
        key=lambda f: abs(f["shap_value"]),
        reverse=True,
    )

    top = features[0]
    return {
        "risk_score": round(risk_score, 4),
        "base_value": round(base_value, 6),
        "top_driver": top["label"],
        "summary":    _summary(top["label"], top["shap_value"], risk_score),
        "features":   features,
    }
