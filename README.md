# FraudLens
### Agentic Fraud Investigation System

> **Live demo:** `[CloudFront URL — added after Phase 4 deployment]`

A LangGraph-orchestrated fraud investigation agent built on the PaySim dataset (6.3M transactions). The agent autonomously selects and sequences tools — fraud scoring, SHAP explanation, account velocity analysis, and regulatory retrieval — based on transaction risk context.

**Target roles:** Agentic/GenAI Engineer · ML Engineer · MLOps/ML Platform Engineer

---

## Architecture

```
Raw CSV (470MB)
    │
    ▼
Dask Pipeline (6.3M rows, chunked)
    │  balance anomaly detection
    │  rolling velocity windows
    │
    ▼
Feature Engineering → features.parquet (112MB, 2.7M rows)
    │
    ▼
XGBoost + SMOTE → xgboost_fraud_realistic.json (1.75MB)
    │
    ▼
SHAP TreeExplainer
    │
    ▼
LangGraph Agent ──► run_fraud_model()
                ──► explain_prediction()
                ──► retrieve_regulations()
                ──► check_account_velocity()
                    │
                    ▼
              FastAPI (Docker → AWS Lambda)
                    │
                    ▼
              React Frontend (CloudFront + S3)
```

---

## Model Performance

Trained on TRANSFER + CASH_OUT transactions only (~2.7M of 6.3M rows).
Realistic feature set — balance leakage flags excluded (see [Architecture Decisions](#architecture-decisions)).

| Metric | Value |
|---|---|
| Recall (fraud class) | **89.9%** |
| Precision (fraud class) | **94.4%** |
| F1 (fraud class) | **92.1%** |
| ROC-AUC | 0.9984 |
| PR-AUC | 0.9709 |
| Optimal threshold | 0.983 |
| False positives (test set) | 88 / 552,439 |
| False negatives (test set) | 166 / 1,643 |

> **Why not accuracy?** With a 0.13% fraud rate, predicting all transactions as legitimate achieves 99.87% accuracy while catching zero fraud. Recall on the fraud class is the correct primary metric.

---

## Data Engineering Pipeline

### Dataset
- **Source:** PaySim — synthetic mobile money transaction simulator
- **Size:** 6,362,620 rows × 11 columns, 470MB CSV
- **Fraud rate:** 8,213 fraud / 6,354,407 legitimate = **0.13%**
- **Fraud types:** TRANSFER and CASH_OUT only (other types contain zero fraud)

### Stage 1 — Chunked ingestion (Pandas)

Reading 470MB into memory at once risks OOM on standard hardware.
Solution: chunked ingestion in 500k-row batches with per-chunk memory profiling.

| Chunk | Rows | Peak memory | Time |
|---|---|---|---|
| 1 | 500,000 | ~45MB | ~2s |
| … | … | … | … |
| 13 | 362,620 | ~35MB | ~1.5s |
| **Total** | **6,362,620** | **<50MB per chunk** | **~25s** |

Key finding: chunked ingestion keeps peak memory flat regardless of dataset size.
This is the pattern used in production ETL — load, transform, release, repeat.

### Stage 2 — Dask EDA (full 6.3M rows)

Dask processes the full dataset without loading it into RAM by operating on lazy partitions.

**Class distribution:**
```
Legitimate : 6,354,407  (99.8710%)
Fraud      :     8,213  ( 0.1290%)
```

**Fraud rate by transaction type:**
```
TRANSFER  : fraud present  ← model trained on this
CASH_OUT  : fraud present  ← model trained on this
PAYMENT   : 0 fraud        ← excluded
DEBIT     : 0 fraud        ← excluded
CASH_IN   : 0 fraud        ← excluded
```

**Balance anomaly detection:**
Classic PaySim fraud signature — origin balance wiped to zero, destination balance unchanged.
Flagging this pattern alone achieves high precision, but see [Label Leakage](#label-leakage-note) below.

### Stage 3 — Feature engineering

Filters to TRANSFER + CASH_OUT first (2,770,409 rows), then computes in pandas (fits in ~500MB RAM).

| Feature | Type | Description |
|---|---|---|
| `log_amount` | Derived | `log1p(amount)` — reduces right skew |
| `type_encoded` | Encoded | 1=TRANSFER, 2=CASH_OUT |
| `orig_balance_diff` | Derived | Expected vs actual origin balance drop |
| `dest_balance_diff` | Derived | Expected vs actual dest balance rise |
| `orig_zero_after` | Binary | Origin balance wiped to zero |
| `dest_unchanged` | Binary | Destination balance didn't change |
| `velocity_cumcount` | Velocity | Cumulative txn count for this account |
| `velocity_1hr` | Velocity | Txns by this account in same hour |
| `velocity_3hr` | Velocity | Txns by this account in same 3hr window |
| `velocity_24hr` | Velocity | Txns by this account in same day |

**Output:** `Dataset/features.parquet` — 112MB (76% compression vs CSV equivalent)

**Velocity implementation note:** initial implementation used `expanding().count()` inside Dask `map_partitions` — O(n²) per partition, caused >20 minute hangs. Replaced with `groupby().transform("count")` — O(n log n), vectorised, completes in ~4 minutes.

### Stage 4 — SMOTE oversampling

Raw training set: 2,216,327 rows, 6,570 fraud (0.30%).
After SMOTE: balanced 50/50 split for training. Test set untouched (stratified, real distribution).

### Label Leakage Note

Initial training with all engineered features achieved near-perfect metrics:
`Recall=99.6%, Precision=100%, F1=99.8%`

**Root cause:** PaySim's fraud generation mechanism sets `newbalanceOrig=0` and never credits `newbalanceDest` for every fraud transaction. Features `orig_zero_after` and `dest_unchanged` therefore directly encode the fraud label — they are simulation artifacts, not real-world signals.

**Resolution:** trained two models:
1. `xgboost_fraud.json` — full features (baseline, confirms pipeline works)
2. `xgboost_fraud_realistic.json` — excludes balance leakage flags (**used in production API**)

The realistic model's metrics represent honest performance. In production, balance discrepancy flags would only be known post-settlement, not at transaction time.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Data Engineering | Pandas (chunked), Dask 2026.3.0, NumPy, PyArrow |
| ML / Modelling | XGBoost 3.2.0, Scikit-learn 1.8.0, imbalanced-learn 0.14.1 (SMOTE) |
| Explainability | SHAP 0.52.0 (TreeExplainer), Matplotlib |
| Agentic AI | LangGraph 1.2.2, LangChain 1.3.2 |
| LLM Inference | HuggingFace Inference API (Mistral-7B-Instruct) |
| RAG | FAISS 1.14.2, sentence-transformers 5.5.1 |
| API | FastAPI 0.136.3, Pydantic 2.13.4, Uvicorn, SSE |
| Frontend | React 18, Vite, Tailwind CSS, Recharts, Axios |
| Cloud — Backend | AWS Lambda (Docker), API Gateway, ECR, S3 |
| Cloud — CDN | CloudFront |
| Monitoring | CloudWatch EventBridge (warm ping), DynamoDB (drift logs) |
| DevOps | GitHub Actions, Docker, pytest |

---

## Architecture Decisions

### Why LangGraph over a fixed pipeline?
v1 used a fixed sequence: data → model → SHAP → LangChain → RAG → report.
v2 replaces this with a ReAct-style agent that decides which tools to call based on what it observes. Low-risk transactions skip SHAP and RAG entirely. High-risk ones trigger all four tools in a reasoned sequence. The agent's tool selection logic is testable, explainable, and extensible without touching the pipeline.

### Why Dask over Spark?
Spark requires cluster orchestration that is unnecessary for a single-machine 6.3M row dataset. Dask provides the same distributed-processing mental model with a familiar pandas-like API, zero cluster setup, and runs locally on a laptop. For datasets requiring multi-node processing (100M+ rows), Spark would be the right call.

### Why XGBoost over a neural network?
XGBoost produces exact SHAP attributions via `TreeExplainer`. Neural network SHAP requires `DeepExplainer` or `GradientExplainer`, which are approximations and significantly slower. Model auditability is a regulatory requirement in financial services — exact attributions are non-negotiable.

### Why FAISS over Pinecone or Weaviate?
FAISS runs locally with zero cost and zero latency overhead for ~50 compliance documents. For production at millions of documents, multi-user access, and SLA requirements, Pinecone or Weaviate is appropriate. This is an explicit scope decision documented here rather than a technical limitation.

### Why HuggingFace Inference API over Ollama in Lambda?
Ollama requires loading 4–7GB model weights into the Lambda container, producing a 10GB+ Docker image and 60–90s cold starts. HuggingFace Inference API offloads inference to HuggingFace's servers, keeps the Docker image under 3GB, eliminates cold start latency, and is free for ~150 requests/month at portfolio scale.

### Why EventBridge warm ping over provisioned Lambda concurrency?
Provisioned concurrency costs $40–100/month. An EventBridge rule pinging every 10 minutes costs ~$0.00/month (4,320 pings vs 14M free events/month) and achieves the same demo reliability. Provisioned concurrency is appropriate at production traffic volumes; EventBridge is appropriate for a portfolio demo.

### Why SSE over WebSockets for agent trace streaming?
SSE is one-directional (server → client), which is exactly what agent step streaming requires. WebSockets add bidirectional connection management overhead that provides no benefit for this use case. SSE is simpler to implement, simpler to debug, and natively supported by FastAPI's `StreamingResponse`.

---

## Infrastructure Cost (~$0.29/month)

| Service | Cost | Notes |
|---|---|---|
| AWS Lambda | ~$0.02 | 500 req/mo × 512MB × 3s |
| API Gateway | ~$0.01 | $3.50/1M HTTP calls |
| ECR | ~$0.25 | ~2.5GB Docker image |
| CloudFront | ~$0.01 | React bundle, 1000 loads/mo |
| S3 | ~$0.00 | ~50MB total |
| DynamoDB | ~$0.00 | KB-scale drift logs |
| EventBridge | ~$0.00 | 4,320 pings/mo |
| HuggingFace | $0.00 | Not AWS — free tier |
| **Total** | **~$0.29** | |

> Set a billing alert at $5/month before deploying.

---

## Local Setup

```bash
git clone https://github.com/adithyaraj/FraudLens.git
cd FraudLens

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

# Download PaySim dataset → Dataset/paysim.csv
# https://www.kaggle.com/datasets/ealaxi/paysim1

# Run pipeline in order
python src/data/ingest.py      # Day 1 — chunked ingestion + memory profile
python src/data/eda.py         # Day 2 — Dask EDA + charts
python src/data/features.py    # Day 3 — feature engineering → features.parquet
python src/ml/train.py         # Day 4 — XGBoost + SMOTE training
python src/ml/explain.py       # Day 5 — SHAP waterfall plots

# Start API
uvicorn src.api.main:app --reload
# Docs at http://localhost:8000/docs

# Run tests
pytest tests/ -v
```

---

## Project Status

- [x] Phase 1 — ML Core (Dask + XGBoost + SHAP + FastAPI)
- [ ] Phase 2 — LangGraph learning track
- [ ] Phase 3 — Agentic layer + RAG + SSE streaming
- [ ] Phase 4 — React frontend + AWS deployment + MLOps hardening

---

*Built by Adithya Raj · May 2026*
