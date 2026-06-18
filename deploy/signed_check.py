"""
One-off: call the Lambda Function URL with a SigV4-signed request (AWS_IAM auth).
Proves the live function runs end-to-end using the local AWS credentials.

Usage:  python deploy/signed_check.py <function-url> [path]
"""
import sys
import urllib.request
import urllib.error

import botocore.session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

REGION = "us-east-1"

base = sys.argv[1].rstrip("/")
path = sys.argv[2] if len(sys.argv) > 2 else "/health"
url = base + path

creds = botocore.session.Session().get_credentials().get_frozen_credentials()
req = AWSRequest(method="GET", url=url)
SigV4Auth(creds, "lambda", REGION).add_auth(req)

http_req = urllib.request.Request(url, headers=dict(req.headers), method="GET")
try:
    with urllib.request.urlopen(http_req, timeout=180) as resp:
        print(f"HTTP {resp.status}")
        print(resp.read().decode())
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}")
    print(e.read().decode())
