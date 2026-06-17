# FraudLens API — AWS Lambda container image (via Lambda Web Adapter).
#
# Runs the FastAPI app as an ordinary uvicorn server. The Lambda Web Adapter
# (LWA) bridges Lambda invocations <-> HTTP and — when the Lambda Function URL
# uses InvokeMode=RESPONSE_STREAM — enables the /investigate SSE stream.
#
# Build:  docker build -t fraudlens-api .
# Run:    docker run -p 8080:8080 --env-file .env fraudlens-api   (local smoke test)

FROM public.ecr.aws/docker/library/python:3.12-slim

# ── Lambda Web Adapter ────────────────────────────────────────────────────────
# Bump the tag to the latest from:
#   https://github.com/awslabs/aws-lambda-web-adapter/releases
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Install CPU-only torch FIRST so sentence-transformers doesn't drag in the
# ~2GB CUDA build (keeps the image well under the Lambda 10GB limit).
RUN pip install --no-cache-dir torch==2.12.0+cpu --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Bake the embedding model into the image (no cold-start download) ───────────
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── App code + model artifacts ────────────────────────────────────────────────
COPY src ./src
COPY models ./models

# ── Pre-build the FAISS index at build time (baked in, not rebuilt per cold start)
RUN python -m src.rag.build_index

# ── Runtime configuration ─────────────────────────────────────────────────────
ENV PORT=8080
ENV AWS_LWA_INVOKE_MODE=response_stream
ENV AWS_LWA_READINESS_CHECK_PATH=/health
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV MPLCONFIGDIR=/tmp/matplotlib

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
