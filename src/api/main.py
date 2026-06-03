"""
Day 6 — FastAPI skeleton with /predict and /explain endpoints.
Loads the realistic XGBoost model + SHAP explainer at startup.
All endpoints are designed to be wrapped as LangGraph agent tools in Phase 3.
"""

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_PATH     = Path("models/xgboost_fraud_realistic.json")
THRESHOLD_PATH = Path("models/threshold_realistic.txt")

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

# ── Global model state (loaded once at startup) ───────────────────────────────

class ModelState:
    model:      xgb.XGBClassifier = None
    explainer:  shap.TreeExplainer = None
    threshold:  float = 0.5


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + SHAP explainer once at startup."""
    print("Loading XGBoost model...")
    state.model = xgb.XGBClassifier()
    state.model.load_model(str(MODEL_PATH))

    print("Building SHAP TreeExplainer...")
    state.explainer = shap.TreeExplainer(state.model)

    print(f"Loading threshold...")
    state.threshold = float(THRESHOLD_PATH.read_text().strip())

    print(f"Ready — threshold={state.threshold:.4f}")
    yield
    print("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FraudLens API",
    description="Agentic fraud investigation system — ML + SHAP + velocity endpoints",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class TransactionInput(BaseModel):
    """Matches PaySim schema fields available at transaction time."""
    step:           int   = Field(..., ge=1, le=743,  description="Hour of simulation (1–743)")
    type:           str   = Field(...,                 description="TRANSFER or CASH_OUT")
    amount:         float = Field(..., gt=0,           description="Transaction amount")
    oldbalanceOrg:  float = Field(..., ge=0,           description="Origin balance before")
    newbalanceOrig: float = Field(..., ge=0,           description="Origin balance after")
    oldbalanceDest: float = Field(..., ge=0,           description="Destination balance before")
    newbalanceDest: float = Field(..., ge=0,           description="Destination balance after")
    # Velocity — caller computes these from account history
    velocity_cumcount: int = Field(default=1, ge=0,   description="Cumulative txn count for this account")
    velocity_1hr:      int = Field(default=1, ge=0,   description="Txns by this account in same hour")
    velocity_3hr:      int = Field(default=1, ge=0,   description="Txns by this account in same 3hr window")
    velocity_24hr:     int = Field(default=1, ge=0,   description="Txns by this account in same day")

    model_config = {"json_schema_extra": {
        "examples": [{
            "step": 1, "type": "TRANSFER", "amount": 181.0,
            "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,  "newbalanceDest": 0.0,
            "velocity_cumcount": 1, "velocity_1hr": 1,
            "velocity_3hr": 1,      "velocity_24hr": 1,
        }]
    }}


class PredictResponse(BaseModel):
    risk_score:       float
    predicted_class:  int
    risk_label:       str
    threshold_used:   float
    flagged_features: list[str]


class SHAPFeature(BaseModel):
    feature:     str
    label:       str
    shap_value:  float
    raw_value:   float
    direction:   str   # "increases_fraud" | "decreases_fraud"


class ExplainResponse(BaseModel):
    risk_score:     float
    base_value:     float
    features:       list[SHAPFeature]
    top_driver:     str
    summary:        str


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_feature_vector(tx: TransactionInput) -> np.ndarray:
    """Convert TransactionInput into the model's feature vector."""
    if tx.type not in TYPE_MAP:
        raise HTTPException(status_code=422, detail=f"type must be TRANSFER or CASH_OUT, got '{tx.type}'")

    type_encoded = TYPE_MAP[tx.type]
    log_amount   = float(np.log1p(tx.amount))

    values = [
        tx.step, type_encoded, tx.amount, log_amount,
        tx.oldbalanceOrg, tx.newbalanceOrig,
        tx.oldbalanceDest, tx.newbalanceDest,
        tx.velocity_cumcount, tx.velocity_1hr,
        tx.velocity_3hr, tx.velocity_24hr,
    ]
    return np.array(values, dtype="float32").reshape(1, -1)


def risk_label(score: float, threshold: float) -> str:
    if score >= threshold:
        return "HIGH"
    elif score >= threshold * 0.6:
        return "MEDIUM"
    return "LOW"


def flagged_features(X: np.ndarray, shap_vals: np.ndarray, top_n: int = 3) -> list[str]:
    """Return top N feature names driving fraud risk upward."""
    positive_shap = [(FEATURE_LABELS[FEATURE_COLS[i]], shap_vals[i])
                     for i in range(len(FEATURE_COLS)) if shap_vals[i] > 0]
    positive_shap.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in positive_shap[:top_n]]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "threshold": state.threshold,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(tx: TransactionInput):
    """
    Run fraud model on a transaction.
    Agent tool: run_fraud_model(transaction) — always called first.
    Returns risk score 0–1 and predicted class.
    """
    X = build_feature_vector(tx)

    risk_score = float(state.model.predict_proba(X)[0][1])
    predicted  = int(risk_score >= state.threshold)

    # Compute SHAP for flagged features (lightweight — single row)
    sv = state.explainer.shap_values(X)[0]

    return PredictResponse(
        risk_score=round(risk_score, 4),
        predicted_class=predicted,
        risk_label=risk_label(risk_score, state.threshold),
        threshold_used=state.threshold,
        flagged_features=flagged_features(X, sv),
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(tx: TransactionInput):
    """
    SHAP attribution for a single transaction.
    Agent tool: explain_prediction(transaction) — called when risk_score > 0.5.
    Returns per-feature SHAP values with directional impact.
    """
    X = build_feature_vector(tx)

    risk_score = float(state.model.predict_proba(X)[0][1])
    sv         = state.explainer.shap_values(X)[0]
    base_value = float(state.explainer.expected_value)

    features = []
    for i, col in enumerate(FEATURE_COLS):
        features.append(SHAPFeature(
            feature=col,
            label=FEATURE_LABELS[col],
            shap_value=round(float(sv[i]), 6),
            raw_value=round(float(X[0][i]), 4),
            direction="increases_fraud" if sv[i] > 0 else "decreases_fraud",
        ))

    # Sort by absolute SHAP impact
    features.sort(key=lambda f: abs(f.shap_value), reverse=True)

    top_driver = features[0].label
    top_shap   = features[0].shap_value
    direction  = "increases" if top_shap > 0 else "decreases"

    summary = (
        f"Primary driver: '{top_driver}' {direction} fraud probability "
        f"by {abs(top_shap):.4f} SHAP units. "
        f"Overall risk score: {risk_score:.4f}."
    )

    return ExplainResponse(
        risk_score=round(risk_score, 4),
        base_value=round(base_value, 6),
        features=features,
        top_driver=top_driver,
        summary=summary,
    )


@app.get("/model-info")
def model_info():
    """Returns model metadata — useful for the frontend architecture tab."""
    return {
        "model_type": "XGBoostClassifier",
        "feature_set": "realistic (no balance leakage flags)",
        "features": FEATURE_COLS,
        "feature_labels": FEATURE_LABELS,
        "threshold": state.threshold,
        "n_estimators": state.model.n_estimators,
        "trained_on": "PaySim 6.3M transactions (TRANSFER + CASH_OUT only)",
        "metrics": {
            "recall": 0.8990,
            "precision": 0.9438,
            "f1": 0.9208,
            "roc_auc": 0.9984,
            "pr_auc": 0.9709,
        }
    }
