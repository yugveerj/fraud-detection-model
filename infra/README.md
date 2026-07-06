# Infrastructure (`infra/`) — scoring API on AWS

All AWS resources for the scoring service, as Terraform. **The agent authors this and
runs `fmt` / `validate` / `plan` only; `apply` is owner- or CI-run** (GitHub Actions
OIDC), never an agent action (CLAUDE.md live-AWS boundary). Once applied, console-only
changes are prohibited — everything goes through a `terraform plan` in a PR.

## What it provisions (region `us-west-2`)

| Resource | Purpose |
| --- | --- |
| ECR repository | holds the Lambda container image (scan-on-push, lifecycle-pruned) |
| Lambda function (image) | `serving.handler.handler` — `/score` + `/healthz` |
| API Gateway HTTP API | public endpoint, CORS locked to the Pages origin, **throttled 5 rps / burst 10** |
| IAM: Lambda exec role | least-privilege execution + logs |
| IAM: OIDC provider + deploy role | GitHub Actions assumes this to deploy — no static keys |
| CloudWatch billing alarms (us-east-1) | estimated charges > **$10** and > **$25** → SNS email |
| CloudWatch Lambda-error alarm | operational alerting |
| Log groups (Lambda + API GW) | 14-day retention |

## Projected cost (steady state)

At demo traffic the service is effectively free:

| Component | Estimate |
| --- | --- |
| Lambda | scale-to-zero; a few requests → **$0** (free tier) |
| API Gateway HTTP API | $1.00 / million requests → **~$0** at demo volume |
| ECR storage | one small image → **pennies** |
| CloudWatch logs/alarms | 14-day retention, 3 alarms → **~$0** |
| **Total** | **≈ $0–3 / month** (billing alarms fire well before $10) |

## Apply order (owner / CI)

The Lambda references an image that must exist first, so the deploy is two-phase (the
`deploy.yml` workflow does this automatically):

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars   # fill in owner values
terraform init
terraform apply -target=aws_ecr_repository.this # 1. create ECR
# 2. build + push the image (needs the serving bundle):
#    uv run python -m serving.artifact && docker build -t <ecr_url>:latest . && docker push ...
terraform apply                                 # 3. Lambda + API + alarms
terraform output api_endpoint
```

Then set the repo variables/secrets the workflows use: `AWS_DEPLOY_ROLE_ARN`
(= `terraform output deploy_role_arn`), `AWS_REGION`, `API_BASE`, `DEMO_URL`.

## Prerequisites the owner sets up once
- Enable **Receive Billing Alerts** in the Billing console (required for the billing
  alarms; us-east-1).
- A remote state backend for a shared account (swap the block in `versions.tf`).
- The `production` GitHub Environment with a required reviewer → enforces the deploy gate.

## Teardown
`terraform destroy` removes everything (ECR has `force_delete = true`).
