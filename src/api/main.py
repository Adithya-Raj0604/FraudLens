"""
FastAPI app — /predict, /explain, /investigate (SSE), /model-info, /health.

The web layer only. All model state and ML logic live in src/ml/inference.py;
these routes validate input, call into that domain module, and shape responses.
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# slowapi must stay pinned in requirements.txt — a missing/mismatched pin here
# crashes uvicorn on import and fails the CI "Start backend" step silently
# (wait-on just times out waiting for /health).
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

from src.ml import inference
from src.ml.inference import FEATURE_COLS, FEATURE_LABELS, state


def load_anthropic_key_from_ssm() -> None:
    """
    In AWS, fetch ANTHROPIC_API_KEY from SSM Parameter Store (SecureString) and
    place it in the environment before the agent is built.

    Local dev is unaffected: .env already provides the key, so this is skipped.
    To enable in Lambda, set env var ANTHROPIC_API_KEY_SSM_PARAM to the parameter
    name (e.g. /fraudlens/anthropic-api-key) and grant the role ssm:GetParameter.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return  # already provided (local .env or a plain Lambda env var)

    param_name = os.environ.get("ANTHROPIC_API_KEY_SSM_PARAM")
    if not param_name:
        return  # nothing configured — nothing to do

    import boto3
    ssm = boto3.client("ssm")
    resp = ssm.get_parameter(Name=param_name, WithDecryption=True)
    os.environ["ANTHROPIC_API_KEY"] = resp["Parameter"]["Value"]
    print(f"Loaded ANTHROPIC_API_KEY from SSM parameter '{param_name}'")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + SHAP explainer once at startup."""
    load_anthropic_key_from_ssm()

    print("Loading model + SHAP explainer...")
    inference.load_model_state()

    print(f"Ready — threshold={state.threshold:.4f}")
    yield
    print("Shutting down.")


# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── Allowed origins ───────────────────────────────────────────────────────────
# In dev: localhost React. In prod: set FRONTEND_ORIGIN env var to your domain.

_FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FraudLens API",
    description="Agentic fraud investigation system — ML + SHAP + velocity endpoints",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_FRONTEND_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "threshold": state.threshold,
    }


@app.post("/predict", response_model=PredictResponse)
@limiter.limit("30/minute")
def predict(request: Request, tx: TransactionInput):
    """
    Run fraud model on a transaction.
    Agent tool: run_fraud_model(transaction) — always called first.
    Returns risk score 0–1 and predicted class.
    """
    try:
        result = inference.predict(tx.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictResponse(
        risk_score=result["risk_score"],
        predicted_class=result["predicted_class"],
        risk_label=result["risk_label"],
        threshold_used=result["threshold"],
        flagged_features=result["flagged_features"],
    )


@app.post("/explain", response_model=ExplainResponse)
@limiter.limit("30/minute")
def explain(request: Request, tx: TransactionInput):
    """
    SHAP attribution for a single transaction.
    Agent tool: explain_prediction(transaction) — called when risk_score >= 0.4.
    Returns per-feature SHAP values with directional impact.
    """
    try:
        result = inference.explain(tx.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return ExplainResponse(
        risk_score=result["risk_score"],
        base_value=result["base_value"],
        features=[SHAPFeature(**f) for f in result["features"]],
        top_driver=result["top_driver"],
        summary=result["summary"],
    )


@app.post("/investigate")
@limiter.limit("10/minute")
async def investigate(request: Request, tx: TransactionInput):
    """
    Agentic fraud investigation with SSE streaming.
    Runs the LangGraph ReAct agent and streams each step as it happens.
    Client receives: tool_call → tool_result → tool_call → ... → report
    """
    from src.agent.graph import stream as agent_stream

    tx_dict = tx.model_dump()
    # TransactionInput uses 'type' field; tools expect it as-is
    tx_dict["type"] = tx.type

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            for event in agent_stream(tx_dict):
                payload = json.dumps(event)
                yield f"data: {payload}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
