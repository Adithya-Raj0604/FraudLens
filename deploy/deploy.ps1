# FraudLens - AWS Lambda deployment  (region us-east-1; AWS account auto-detected from credentials)
#
# Run from the REPO ROOT so the file:// paths resolve:
#     ./deploy/deploy.ps1
#
# One-time prerequisite - store your Anthropic key as a SecureString:
#     aws ssm put-parameter --name /fraudlens/anthropic-api-key `
#         --type SecureString --value "sk-ant-..." --region us-east-1

# NOTE: deliberately NOT "Stop". In Windows PowerShell 5.1, $ErrorActionPreference="Stop"
# turns any native command writing to stderr (e.g. `aws ... describe` on a not-yet-existing
# resource - which is normal on first deploy) into a terminating error. We detect failures
# explicitly via $LASTEXITCODE + Assert-LastExit instead.
$ErrorActionPreference = "Continue"

# Account is resolved from your current AWS credentials - no ID hard-coded in the
# repo. Override by setting $env:AWS_ACCOUNT_ID before running.
$ACCOUNT = if ($env:AWS_ACCOUNT_ID) { $env:AWS_ACCOUNT_ID } else { (aws sts get-caller-identity --query Account --output text) }
$REGION  = "us-east-1"
$REPO    = "fraudlens-api"
$FUNC    = "fraudlens-api"
$ROLE    = "fraudlens-lambda-role"
$ECR     = "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
$IMAGE   = "$ECR/${REPO}:latest"

function Assert-LastExit($what) {
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what (exit $LASTEXITCODE)" }
}

Write-Host "==> 1/6  Build image (single-platform, no attestations - Lambda requires this)"
docker build --provenance=false --sbom=false --platform linux/amd64 -t $REPO .
Assert-LastExit "docker build"

Write-Host "==> 2/6  Ensure ECR repository"
aws ecr describe-repositories --repository-names $REPO --region $REGION 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    aws ecr create-repository --repository-name $REPO --region $REGION | Out-Null
    Assert-LastExit "ecr create-repository"
}

Write-Host "==> 3/6  Push image to ECR"
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR
Assert-LastExit "ecr login + docker login"
docker tag "${REPO}:latest" $IMAGE
docker push $IMAGE
Assert-LastExit "docker push"

Write-Host "==> 4/6  Ensure IAM execution role"
$roleArn = aws iam get-role --role-name $ROLE --query "Role.Arn" --output text 2>$null
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($roleArn)) {
    aws iam create-role --role-name $ROLE `
        --assume-role-policy-document file://deploy/lambda-trust-policy.json | Out-Null
    Assert-LastExit "iam create-role"
    aws iam attach-role-policy --role-name $ROLE `
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole | Out-Null
    Assert-LastExit "iam attach-role-policy"
    # Substitute the resolved account ID into the policy template (the committed
    # file keeps a __ACCOUNT_ID__ placeholder so no real ID lives in the repo).
    $ssmPolicy = Join-Path $env:TEMP "fraudlens-ssm-policy.json"
    (Get-Content deploy/lambda-ssm-policy.json -Raw).Replace("__ACCOUNT_ID__", $ACCOUNT) |
        Set-Content -Path $ssmPolicy -Encoding utf8
    aws iam put-role-policy --role-name $ROLE --policy-name fraudlens-ssm-read `
        --policy-document file://$ssmPolicy | Out-Null
    Assert-LastExit "iam put-role-policy"
    Write-Host "    waiting 15s for IAM role propagation..."
    Start-Sleep -Seconds 15
    $roleArn = aws iam get-role --role-name $ROLE --query "Role.Arn" --output text
}
Write-Host "    role: $roleArn"

Write-Host "==> 5/6  Create or update the Lambda function"
aws lambda get-function --function-name $FUNC --region $REGION 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    aws lambda update-function-code --function-name $FUNC --image-uri $IMAGE --region $REGION | Out-Null
    Assert-LastExit "lambda update-function-code"
} else {
    aws lambda create-function --function-name $FUNC --package-type Image `
        --code ImageUri=$IMAGE --role $roleArn `
        --memory-size 3008 --timeout 120 `
        --environment file://deploy/lambda-env.json `
        --region $REGION | Out-Null
    Assert-LastExit "lambda create-function"
}

Write-Host "==> 6/7  Cap reserved concurrency (limits max AWS burn from the public URL)"
# Best-effort: on brand-new accounts the TOTAL concurrency limit is only 10, and
# AWS refuses any reservation that drops the unreserved pool below its floor of 10.
# That account-wide cap of 10 already bounds burn, so we just warn and continue.
aws lambda put-function-concurrency --function-name $FUNC `
    --reserved-concurrent-executions 2 --region $REGION | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Could not set reserved concurrency (likely the account's low concurrency quota). Skipping - the account-wide limit still caps burn."
}

Write-Host "==> 7/7  Ensure Function URL (RESPONSE_STREAM = SSE streaming)"
aws lambda get-function-url-config --function-name $FUNC --region $REGION 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    aws lambda create-function-url-config --function-name $FUNC `
        --auth-type NONE --invoke-mode RESPONSE_STREAM --region $REGION | Out-Null
    Assert-LastExit "create-function-url-config"
    # Function URLs created after Oct 2025 need BOTH lambda:InvokeFunctionUrl AND
    # lambda:InvokeFunction granted publicly, or a NONE URL still returns 403.
    aws lambda add-permission --function-name $FUNC `
        --action lambda:InvokeFunctionUrl --principal "*" `
        --function-url-auth-type NONE --statement-id fnurl-public --region $REGION | Out-Null
    Assert-LastExit "add-permission InvokeFunctionUrl"
    aws lambda add-permission --function-name $FUNC `
        --action lambda:InvokeFunction --principal "*" `
        --invoked-via-function-url --statement-id fnurl-public-invoke --region $REGION | Out-Null
    Assert-LastExit "add-permission InvokeFunction"
}

$url = aws lambda get-function-url-config --function-name $FUNC --query "FunctionUrl" --output text --region $REGION
Write-Host ""
Write-Host "Deployed. Function URL: $url"
Write-Host "Health check:  curl ${url}health"
Write-Host "Note: the first invoke may be slow while Lambda optimizes the image."
