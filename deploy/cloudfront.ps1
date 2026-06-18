# FraudLens - CloudFront + S3 for the frontend, fronting the IAM-protected Lambda API.
#
#   Browser -> CloudFront (public) -> OAC SigV4-signs -> { S3 (React app) | Lambda API }
#
# Run from the REPO ROOT, AFTER deploy/deploy.ps1 has created the Lambda + Function URL:
#     ./deploy/cloudfront.ps1
#
# Idempotent: re-running reuses the bucket/OACs/distribution and just rebuilds +
# re-syncs the frontend. CloudFront edits take ~10-15 min to propagate.

$ErrorActionPreference = "Continue"   # native stderr must not abort (see deploy.ps1)

$REGION = "us-east-1"
$FUNC   = "fraudlens-api"
$ACCOUNT = if ($env:AWS_ACCOUNT_ID) { $env:AWS_ACCOUNT_ID } else { (aws sts get-caller-identity --query Account --output text) }
$BUCKET = "fraudlens-frontend-$ACCOUNT"   # bucket names are global; the account id keeps it unique
$DIST_COMMENT = "fraudlens-cdn"

function Assert-LastExit($what) {
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what (exit $LASTEXITCODE)" }
}

# ── 1/7  Resolve the Lambda Function URL domain ───────────────────────────────
Write-Host "==> 1/7  Resolve Lambda Function URL"
$fnUrl = aws lambda get-function-url-config --function-name $FUNC --query FunctionUrl --output text 2>$null
if ([string]::IsNullOrWhiteSpace($fnUrl) -or $fnUrl -eq "None") {
    throw "No Function URL on $FUNC. Run deploy/deploy.ps1 first."
}
$LAMBDA_DOMAIN = $fnUrl -replace '^https://', '' -replace '/$', ''
Write-Host "    lambda origin: $LAMBDA_DOMAIN"

# ── 2/7  S3 bucket (private; reachable only via CloudFront OAC) ────────────────
Write-Host "==> 2/7  Ensure private S3 bucket  $BUCKET"
aws s3api head-bucket --bucket $BUCKET 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    # us-east-1 must NOT pass a LocationConstraint
    aws s3api create-bucket --bucket $BUCKET --region $REGION | Out-Null
    Assert-LastExit "create-bucket"
    aws s3api put-public-access-block --bucket $BUCKET `
        --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true" | Out-Null
    Assert-LastExit "put-public-access-block"
}

# ── 3/7  Origin Access Controls (one for S3, one for Lambda) ──────────────────
function Ensure-OAC($name, $originType, $signing) {
    $id = aws cloudfront list-origin-access-controls `
        --query "OriginAccessControlList.Items[?Name=='$name'].Id | [0]" --output text 2>$null
    if ([string]::IsNullOrWhiteSpace($id) -or $id -eq "None") {
        $cfg = Join-Path $env:TEMP "$name.json"
        @{
            Name                            = $name
            OriginAccessControlOriginType   = $originType
            SigningBehavior                 = $signing
            SigningProtocol                 = "sigv4"
        } | ConvertTo-Json | Set-Content -Path $cfg -Encoding ascii
        $id = aws cloudfront create-origin-access-control `
            --origin-access-control-config file://$cfg `
            --query "OriginAccessControl.Id" --output text
        Assert-LastExit "create-origin-access-control $name"
    }
    return $id
}
Write-Host "==> 3/7  Ensure Origin Access Controls"
# S3 OAC signs (S3 requires it). The Lambda OAC must NOT sign: OAC cannot sign
# POST request bodies, which breaks /predict + /investigate with InvalidSignature.
# The Function URL is NONE (public), so CloudFront forwards unsigned and Lambda's
# public resource policy allows it.
$S3_OAC_ID     = Ensure-OAC "fraudlens-s3-oac"     "s3"     "always"
$LAMBDA_OAC_ID = Ensure-OAC "fraudlens-lambda-oac" "lambda" "never"
Write-Host "    s3 OAC: $S3_OAC_ID   lambda OAC: $LAMBDA_OAC_ID"

# ── 4/7  CloudFront distribution ──────────────────────────────────────────────
Write-Host "==> 4/7  Ensure CloudFront distribution"
$DIST_ID = aws cloudfront list-distributions `
    --query "DistributionList.Items[?Comment=='$DIST_COMMENT'].Id | [0]" --output text 2>$null
if ([string]::IsNullOrWhiteSpace($DIST_ID) -or $DIST_ID -eq "None") {
    $cfg = Join-Path $env:TEMP "fraudlens-distribution.json"
    (Get-Content deploy/cloudfront-distribution.json -Raw).
        Replace("__CALLER_REF__",  [guid]::NewGuid().ToString()).
        Replace("__S3_DOMAIN__",   "$BUCKET.s3.$REGION.amazonaws.com").
        Replace("__S3_OAC_ID__",   $S3_OAC_ID).
        Replace("__LAMBDA_DOMAIN__", $LAMBDA_DOMAIN).
        Replace("__LAMBDA_OAC_ID__", $LAMBDA_OAC_ID) |
        Set-Content -Path $cfg -Encoding ascii
    $DIST_ID = aws cloudfront create-distribution --distribution-config file://$cfg `
        --query "Distribution.Id" --output text
    Assert-LastExit "create-distribution"
}
$DIST_ARN    = aws cloudfront get-distribution --id $DIST_ID --query "Distribution.ARN" --output text
$DIST_DOMAIN = aws cloudfront get-distribution --id $DIST_ID --query "Distribution.DomainName" --output text
Write-Host "    distribution: $DIST_ID  ($DIST_DOMAIN)"

# ── 5/7  Authorize CloudFront on both origins ─────────────────────────────────
Write-Host "==> 5/7  Authorize CloudFront on S3 + Lambda"
# S3 bucket policy (allow only this distribution via OAC)
$bp = Join-Path $env:TEMP "fraudlens-bucket-policy.json"
(Get-Content deploy/s3-bucket-policy.json -Raw).
    Replace("__BUCKET__", $BUCKET).
    Replace("__DIST_ARN__", $DIST_ARN) |
    Set-Content -Path $bp -Encoding ascii
aws s3api put-bucket-policy --bucket $BUCKET --policy file://$bp | Out-Null
Assert-LastExit "put-bucket-policy"

# Lambda resource policy: let CloudFront (this distribution) invoke the IAM URL.
# Function URLs created after Oct 2025 require BOTH lambda:InvokeFunctionUrl AND
# lambda:InvokeFunction - granting only the first returns 403. (See AWS docs:
# lambda/latest/dg/urls-auth.html.)
aws lambda remove-permission --function-name $FUNC --statement-id cloudfront-oac --region $REGION 2>$null | Out-Null
aws lambda add-permission --function-name $FUNC --statement-id cloudfront-oac `
    --action lambda:InvokeFunctionUrl --principal cloudfront.amazonaws.com `
    --source-arn $DIST_ARN --function-url-auth-type AWS_IAM --region $REGION | Out-Null
Assert-LastExit "lambda add-permission InvokeFunctionUrl (cloudfront)"

aws lambda remove-permission --function-name $FUNC --statement-id cloudfront-oac-invoke --region $REGION 2>$null | Out-Null
aws lambda add-permission --function-name $FUNC --statement-id cloudfront-oac-invoke `
    --action lambda:InvokeFunction --principal cloudfront.amazonaws.com `
    --source-arn $DIST_ARN --invoked-via-function-url --region $REGION | Out-Null
Assert-LastExit "lambda add-permission InvokeFunction (cloudfront)"

# ── 6/7  Build the frontend ───────────────────────────────────────────────────
Write-Host "==> 6/7  Build React frontend (npm run build)"
Push-Location frontend
npm install | Out-Null
Assert-LastExit "npm install"
npm run build | Out-Null
Assert-LastExit "npm run build"
Pop-Location

# ── 7/7  Upload to S3 + invalidate cache ──────────────────────────────────────
Write-Host "==> 7/7  Sync build to S3 + invalidate CloudFront"
aws s3 sync frontend/dist "s3://$BUCKET" --delete | Out-Null
Assert-LastExit "s3 sync"
aws cloudfront create-invalidation --distribution-id $DIST_ID --paths "/*" | Out-Null
Assert-LastExit "create-invalidation"

Write-Host ""
Write-Host "Deployed.  https://$DIST_DOMAIN"
Write-Host "  Frontend:  https://$DIST_DOMAIN/"
Write-Host "  API:       https://$DIST_DOMAIN/health"
Write-Host "Note: a new distribution takes ~10-15 min to finish deploying before it serves traffic."
