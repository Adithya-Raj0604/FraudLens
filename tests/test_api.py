"""
Basic API tests for /health, /predict, /explain, /model-info.
Run: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


# ── Sample payloads ───────────────────────────────────────────────────────────

FRAUD_TX = {
    "step": 1, "type": "TRANSFER", "amount": 181.0,
    "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,  "newbalanceDest": 0.0,
    "velocity_cumcount": 1, "velocity_1hr": 1,
    "velocity_3hr": 1,      "velocity_24hr": 1,
}

LEGIT_TX = {
    "step": 200, "type": "CASH_OUT", "amount": 5000.0,
    "oldbalanceOrg": 50000.0, "newbalanceOrig": 45000.0,
    "oldbalanceDest": 10000.0, "newbalanceDest": 15000.0,
    "velocity_cumcount": 10, "velocity_1hr": 1,
    "velocity_3hr": 2,       "velocity_24hr": 3,
}

INVALID_TYPE_TX = {**FRAUD_TX, "type": "PAYMENT"}
INVALID_AMOUNT_TX = {**FRAUD_TX, "amount": -100.0}


# ── Health ────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert 0 < data["threshold"] < 1


# ── Predict ───────────────────────────────────────────────────────────────────

def test_predict_returns_valid_schema(client):
    r = client.post("/predict", json=FRAUD_TX)
    assert r.status_code == 200
    data = r.json()
    assert "risk_score" in data
    assert "predicted_class" in data
    assert "risk_label" in data
    assert "threshold_used" in data
    assert "flagged_features" in data


def test_predict_risk_score_in_range(client):
    r = client.post("/predict", json=FRAUD_TX)
    assert 0.0 <= r.json()["risk_score"] <= 1.0


def test_predict_risk_label_valid(client):
    r = client.post("/predict", json=FRAUD_TX)
    assert r.json()["risk_label"] in {"HIGH", "MEDIUM", "LOW"}


def test_predict_legit_transaction(client):
    r = client.post("/predict", json=LEGIT_TX)
    assert r.status_code == 200
    assert r.json()["risk_score"] < 0.9


def test_predict_invalid_type_rejected(client):
    r = client.post("/predict", json=INVALID_TYPE_TX)
    assert r.status_code == 422


def test_predict_negative_amount_rejected(client):
    r = client.post("/predict", json=INVALID_AMOUNT_TX)
    assert r.status_code == 422


# ── Explain ───────────────────────────────────────────────────────────────────

def test_explain_returns_valid_schema(client):
    r = client.post("/explain", json=FRAUD_TX)
    assert r.status_code == 200
    data = r.json()
    assert "risk_score" in data
    assert "base_value" in data
    assert "features" in data
    assert "top_driver" in data
    assert "summary" in data


def test_explain_feature_count(client):
    r = client.post("/explain", json=FRAUD_TX)
    features = r.json()["features"]
    assert len(features) == 12  # FEATURE_COLS count


def test_explain_features_have_direction(client):
    r = client.post("/explain", json=FRAUD_TX)
    for f in r.json()["features"]:
        assert f["direction"] in {"increases_fraud", "decreases_fraud"}


def test_explain_sorted_by_abs_shap(client):
    r = client.post("/explain", json=FRAUD_TX)
    shap_vals = [abs(f["shap_value"]) for f in r.json()["features"]]
    assert shap_vals == sorted(shap_vals, reverse=True)


def test_explain_summary_is_string(client):
    r = client.post("/explain", json=FRAUD_TX)
    assert isinstance(r.json()["summary"], str)
    assert len(r.json()["summary"]) > 20


# ── Model info ────────────────────────────────────────────────────────────────

def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert "model_type" in data
    assert "features" in data
    assert "metrics" in data
    assert data["metrics"]["recall"] > 0.8
