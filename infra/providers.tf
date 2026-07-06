provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "terraform"
      Purpose   = "fraud-detection-demo"
    }
  }
}

# CloudWatch billing metrics (AWS/Billing EstimatedCharges) are published only in
# us-east-1, so the billing alarms use this aliased provider.
provider "aws" {
  alias  = "billing"
  region = "us-east-1"

  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
