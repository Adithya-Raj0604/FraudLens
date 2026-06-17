"""
Agent tools — four callable functions the LangGraph agent can invoke.
Each function takes plain Python types and returns a plain dict or string.
"""

import sys
import numpy as np
from pathlib import Path

# Makes `src` a reachable package root regardless of where Python is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.main import build_feature_vector, flagged_features, risk_label, state, FEATURE_COLS, FEATURE_LABELS
from src.rag.pipeline import build_index, query_index

# ── RAG index — built once at module load ─────────────────────────────────────
_DOCS_DIR = Path(__file__).resolve().parents[2] / "src" / "rag" / "docs"
_index, _chunks = build_index(str(_DOCS_DIR))


# ── Tools ─────────────────────────────────────────────────────────────────────

def run_fraud_model(transaction: dict) -> dict:
    """
    Run the XGBoost fraud model on a transaction dict.
    Always the first tool the agent calls.
    """
    from src.api.main import TransactionInput

    tx         = TransactionInput(**transaction)
    X          = build_feature_vector(tx)
    risk_score = float(state.model.predict_proba(X)[0][1])
    predicted  = int(risk_score >= state.threshold)
    sv         = state.explainer.shap_values(X)[0]

    return {
        "risk_score":       round(risk_score, 4),
        "predicted_class":  predicted,
        "risk_label":       risk_label(risk_score, state.threshold),
        "flagged_features": flagged_features(X, sv),
    }


def explain_prediction(transaction: dict) -> dict:
    """
    Run SHAP explainability on a transaction dict.
    Agent calls this when risk_score is above threshold.
    """
    from src.api.main import TransactionInput

    tx         = TransactionInput(**transaction)
    X          = build_feature_vector(tx)
    risk_score = float(state.model.predict_proba(X)[0][1])
    sv         = state.explainer.shap_values(X)[0]

    # Pair each feature name with its SHAP value, sort by absolute impact
    features = sorted(
        [
            {
                "feature":    FEATURE_LABELS[col],
                "shap_value": round(float(val), 6),
                "direction":  "increases_fraud" if val > 0 else "decreases_fraud",
            }
            for col, val in zip(FEATURE_COLS, sv)
        ],
        key=lambda f: abs(f["shap_value"]),
        reverse=True,
    )

    top        = features[0]
    direction  = "increases" if top["shap_value"] > 0 else "decreases"
    summary    = (
        f"Primary driver: '{top['feature']}' {direction} fraud probability "
        f"by {abs(top['shap_value']):.4f} SHAP units. "
        f"Overall risk score: {risk_score:.4f}."
    )

    return {
        "risk_score":   round(risk_score, 4),
        "top_driver":   top["feature"],
        "summary":      summary,
        "top_features": features[:3],
    }


def check_account_velocity(transaction: dict) -> dict:
    """
    Return velocity stats for the originating account.
    In production this would query a rolling transaction-history store keyed by
    account_id; in this demo it surfaces the velocity features already on the
    transaction so the agent can reason over behavioural frequency (and the
    numbers stay consistent with the transaction under investigation).
    """
    v1  = int(transaction.get("velocity_1hr", 0))
    v3  = int(transaction.get("velocity_3hr", 0))
    v24 = int(transaction.get("velocity_24hr", 0))

    return {
        "velocity_1hr":      v1,
        "velocity_3hr":      v3,
        "velocity_24hr":     v24,
        "velocity_cumcount": int(transaction.get("velocity_cumcount", 0)),
        "risk_flag":         v24 > 5,
        "note": "risk_flag=True when velocity_24hr > 5 (Customer Risk Rating Matrix).",
    }


def retrieve_regulations(query: str, top_k: int = 2, max_chars: int = 500) -> str:
    """
    Semantic search over OSFI/FINTRAC compliance documents.
    Returns the top matching chunks, each truncated, as a single string.
    Trimmed to keep the agent's token usage low — full chunks are large.
    """
    results = query_index(_index, _chunks, query, top_k=top_k)
    trimmed = [
        (r[:max_chars].rsplit(" ", 1)[0] + " …") if len(r) > max_chars else r
        for r in results
    ]
    return "\n\n---\n\n".join(trimmed)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import xgboost as xgb
    import shap
    from pathlib import Path

    # Manually load model so state is populated (normally FastAPI lifespan does this)
    state.model = xgb.XGBClassifier()
    state.model.load_model("models/xgboost_fraud_realistic.json")
    state.explainer = shap.TreeExplainer(state.model)
    state.threshold = float(Path("models/threshold_realistic.txt").read_text().strip())

    sample_tx = {
        "step": 1, "type": "TRANSFER", "amount": 9000.0,
        "oldbalanceOrg": 9000.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,   "newbalanceDest": 9000.0,
        "velocity_cumcount": 7,  "velocity_1hr": 2,
        "velocity_3hr": 4,       "velocity_24hr": 7,
    }

    print("=== run_fraud_model ===")
    print(run_fraud_model(sample_tx))

    print("\n=== explain_prediction ===")
    result = explain_prediction(sample_tx)
    print(result["summary"])
    for f in result["top_features"]:
        print(f)

    print("\n=== check_account_velocity ===")
    print(check_account_velocity(sample_tx))

    print("\n=== retrieve_regulations ===")
    print(retrieve_regulations("account balance drained fraud escalation")[:500])
