# FraudLens — AWS Deployment

Deploys the FastAPI agent as a containerized **AWS Lambda** function behind a
**Function URL** with response streaming (so the `/investigate` SSE live-trace
works). The Anthropic API key is read at runtime from **SSM Parameter Store**.

| Resource | Name |
|---|---|
| Region / Account | `us-east-1` / *(your AWS account — auto-detected from credentials)* |
| ECR repo | `fraudlens-api` |
| Lambda function | `fraudlens-api` |
| IAM role | `fraudlens-lambda-role` |
| SSM SecureString | `/fraudlens/anthropic-api-key` |

## Files

- `lambda-trust-policy.json` — lets Lambda assume the execution role
- `lambda-ssm-policy.json` — `ssm:GetParameter` + `kms:Decrypt` scoped to the one parameter
- `lambda-env.json` — Lambda env vars (the SSM param **name**, not the key; `FRONTEND_ORIGIN`)
- `deploy.ps1` — idempotent build → push → IAM → Lambda → Function URL

## Steps

**1. Store the secret (run once, with your real key):**

```powershell
aws ssm put-parameter --name /fraudlens/anthropic-api-key `
    --type SecureString --value "sk-ant-..." --region us-east-1
```

**2. Deploy (from the repo root):**

```powershell
./deploy/deploy.ps1
```

The script prints the **Function URL** at the end. Test it:

```powershell
curl https://<id>.lambda-url.us-east-1.on.aws/health
```

## Notes

- **Secrets never enter the image** — `.env` is in `.dockerignore`; the key lives
  only in SSM and is fetched at startup via the Lambda role.
- **Image manifest** — `deploy.ps1` builds with `--provenance=false --sbom=false`
  because Lambda rejects the multi-manifest/attestation images that BuildKit
  produces by default.
- **Cold start** — the first invoke after deploy is slow while Lambda optimizes
  the ~4.4 GB image; add an EventBridge warm-ping rule (every 5 min) for demos.
- **Public endpoint** — Function URL auth is `NONE`. Rate limiting (slowapi) is
  on, but also set an Anthropic billing cap since the endpoint calls a paid LLM.
- **Updating** — re-running `deploy.ps1` rebuilds, pushes, and calls
  `update-function-code` (idempotent).
