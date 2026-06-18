"""
Agent tools — four callable functions the LangGraph agent can invoke.
Each function takes plain Python types and returns a plain dict or string.
"""

import sys
import numpy as np
from pathlib import Path

# Makes `src` a reachable package root regardless of where Python is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ml import inference
from src.rag.pipeline import build_index, load_index, query_index

# ── RAG index ─────────────────────────────────────────────────────────────────
# Prefer a pre-built index (baked into the Docker image at build time) so cold
# starts don't re-encode the docs. Fall back to building it for local dev.
_RAG_DIR   = Path(__file__).resolve().parents[2] / "src" / "rag"
_DOCS_DIR  = _RAG_DIR / "docs"
_INDEX_DIR = _RAG_DIR / "index"

if (_INDEX_DIR / "faiss.index").exists():
    _index, _chunks = load_index(str(_INDEX_DIR))
    print(f"Loaded pre-built RAG index: {_index.ntotal} vectors")
else:
    _index, _chunks = build_index(str(_DOCS_DIR))


# ── Tools ─────────────────────────────────────────────────────────────────────

def run_fraud_model(transaction: dict) -> dict:
    """
    Run the XGBoost fraud model on a transaction dict.
    Always the first tool the agent calls.
    """
    result = inference.predict(transaction)
    return {
        "risk_score":       result["risk_score"],
        "predicted_class":  result["predicted_class"],
        "risk_label":       result["risk_label"],
        "flagged_features": result["flagged_features"],
    }


def explain_prediction(transaction: dict) -> dict:
    """
    Run SHAP explainability on a transaction dict.
    Agent calls this when risk_score is at or above the explain threshold.
    """
    result = inference.explain(transaction)

    # Trim to the top 3 drivers and the keys the agent needs (keeps tokens low).
    top_features = [
        {
            "feature":    f["label"],
            "shap_value": f["shap_value"],
            "direction":  f["direction"],
        }
        for f in result["features"][:3]
    ]

    return {
        "risk_score":   result["risk_score"],
        "top_driver":   result["top_driver"],
        "summary":      result["summary"],
        "top_features": top_features,
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
    # Populate model state (normally FastAPI lifespan does this).
    inference.load_model_state()

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
