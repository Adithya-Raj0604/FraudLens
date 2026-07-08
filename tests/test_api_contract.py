"""
API contract tests — JSON schema validation and response-time budgets for the
FraudLens FastAPI endpoints.

Complements tests/test_api.py, which covers business-logic assertions (score
ranges, labels, 422 rejection). This file checks the *shape* of each response
against an explicit JSON Schema (independent of the Pydantic models used to
build it) and asserts each call completes within a latency budget.

Run: pytest tests/test_api_contract.py -v
"""

import time

import pytest
from fastapi.testclient import TestClient
from jsonschema import validate

from src.api.main import app

# Generous budgets for CI runners (in-process TestClient, no real network hop)
# — loose enough to avoid flakiness, tight enough to catch a real regression.
RESPONSE_TIME_BUDGETS = {
    "health": 0.5,
    "predict": 1.5,
    "explain": 2.0,
    "model_info": 0.5,
}


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


SAMPLE_TX = {
    "step": 1, "type": "TRANSFER", "amount": 181.0,
    "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    "velocity_cumcount": 1, "velocity_1hr": 1,
    "velocity_3hr": 1, "velocity_24hr": 1,
}

# ── JSON Schemas ────────────────────────────────────────────────────────────

HEALTH_SCHEMA = {
    "type": "object",
    "required": ["status", "model_loaded", "threshold"],
    "properties": {
        "status": {"type": "string"},
        "model_loaded": {"type": "boolean"},
        "threshold": {"type": "number"},
    },
}

PREDICT_SCHEMA = {
    "type": "object",
    "required": ["risk_score", "predicted_class", "risk_label", "threshold_used", "flagged_features"],
    "properties": {
        "risk_score": {"type": "number", "minimum": 0, "maximum": 1},
        "predicted_class": {"type": "integer", "enum": [0, 1]},
        "risk_label": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
        "threshold_used": {"type": "number"},
        "flagged_features": {"type": "array", "items": {"type": "string"}},
    },
}

SHAP_FEATURE_SCHEMA = {
    "type": "object",
    "required": ["feature", "label", "shap_value", "raw_value", "direction"],
    "properties": {
        "feature": {"type": "string"},
        "label": {"type": "string"},
        "shap_value": {"type": "number"},
        "raw_value": {"type": "number"},
        "direction": {"type": "string", "enum": ["increases_fraud", "decreases_fraud"]},
    },
}

EXPLAIN_SCHEMA = {
    "type": "object",
    "required": ["risk_score", "base_value", "features", "top_driver", "summary"],
    "properties": {
        "risk_score": {"type": "number", "minimum": 0, "maximum": 1},
        "base_value": {"type": "number"},
        "features": {"type": "array", "items": SHAP_FEATURE_SCHEMA, "minItems": 1},
        "top_driver": {"type": "string"},
        "summary": {"type": "string"},
    },
}

MODEL_INFO_SCHEMA = {
    "type": "object",
    "required": ["model_type", "features", "feature_labels", "threshold", "metrics"],
    "properties": {
        "model_type": {"type": "string"},
        "features": {"type": "array", "items": {"type": "string"}},
        "feature_labels": {"type": "object"},
        "threshold": {"type": "number"},
        "metrics": {
            "type": "object",
            "required": ["recall", "precision", "f1", "roc_auc", "pr_auc"],
            "properties": {k: {"type": "number"} for k in ("recall", "precision", "f1", "roc_auc", "pr_auc")},
        },
    },
}

# FastAPI's 422 body shape (list of pydantic error objects under "detail")
VALIDATION_ERROR_SCHEMA = {
    "type": "object",
    "required": ["detail"],
}


def _timed(fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_schema_and_latency(client):
    resp, elapsed = _timed(lambda: client.get("/health"))
    assert resp.status_code == 200
    validate(instance=resp.json(), schema=HEALTH_SCHEMA)
    assert elapsed < RESPONSE_TIME_BUDGETS["health"], f"/health took {elapsed:.3f}s"


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_schema_and_latency(client):
    resp, elapsed = _timed(lambda: client.post("/predict", json=SAMPLE_TX))
    assert resp.status_code == 200
    validate(instance=resp.json(), schema=PREDICT_SCHEMA)
    assert elapsed < RESPONSE_TIME_BUDGETS["predict"], f"/predict took {elapsed:.3f}s"


def test_predict_validation_error_schema(client):
    bad_tx = {**SAMPLE_TX, "amount": -1}
    resp = client.post("/predict", json=bad_tx)
    assert resp.status_code == 422
    validate(instance=resp.json(), schema=VALIDATION_ERROR_SCHEMA)


# ── /explain ──────────────────────────────────────────────────────────────────

def test_explain_schema_and_latency(client):
    resp, elapsed = _timed(lambda: client.post("/explain", json=SAMPLE_TX))
    assert resp.status_code == 200
    validate(instance=resp.json(), schema=EXPLAIN_SCHEMA)
    assert elapsed < RESPONSE_TIME_BUDGETS["explain"], f"/explain took {elapsed:.3f}s"


# ── /model-info ───────────────────────────────────────────────────────────────

def test_model_info_schema_and_latency(client):
    resp, elapsed = _timed(lambda: client.get("/model-info"))
    assert resp.status_code == 200
    validate(instance=resp.json(), schema=MODEL_INFO_SCHEMA)
    assert elapsed < RESPONSE_TIME_BUDGETS["model_info"], f"/model-info took {elapsed:.3f}s"


# ── Cross-endpoint consistency ──────────────────────────────────────────────

def test_predict_and_explain_agree_on_risk_score(client):
    """/predict and /explain both score the same transaction — results must match."""
    predict_score = client.post("/predict", json=SAMPLE_TX).json()["risk_score"]
    explain_score = client.post("/explain", json=SAMPLE_TX).json()["risk_score"]
    assert predict_score == explain_score
