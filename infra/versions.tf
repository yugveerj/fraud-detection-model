terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.70"
    }
  }

  # Local state by default (fine for a single-owner demo). For a shared/real account,
  # switch to an S3 + DynamoDB backend or Terraform Cloud (free tier) — a one-block
  # change here. No console-only changes once this is applied (SPEC Section 5).
}
