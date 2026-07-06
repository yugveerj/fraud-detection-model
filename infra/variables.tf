variable "project" {
  type        = string
  default     = "fraud-scoring"
  description = "Name prefix for all resources."
}

variable "region" {
  type        = string
  default     = "us-west-2"
  description = "AWS region (matches project 2's footprint, SPEC Section 5)."
}

variable "image_tag" {
  type        = string
  default     = "latest"
  description = "ECR image tag the Lambda runs. The deploy workflow pushes this tag."
}

variable "lambda_memory_mb" {
  type        = number
  default     = 2048
  description = "Lambda memory (MB). SHAP + XGBoost init benefit from headroom."
}

variable "lambda_timeout_s" {
  type        = number
  default     = 30
  description = "Lambda timeout (seconds)."
}

variable "throttle_rate" {
  type        = number
  default     = 5
  description = "API Gateway steady-state requests/sec (cost guard)."
}

variable "throttle_burst" {
  type        = number
  default     = 10
  description = "API Gateway burst capacity (cost guard)."
}

variable "pages_origin" {
  type        = string
  default     = "https://example.github.io"
  description = "GitHub Pages origin allowed by CORS (set to your Pages URL)."
}

variable "github_owner" {
  type        = string
  default     = ""
  description = "GitHub org/user for the OIDC deploy-role trust policy."
}

variable "github_repo" {
  type        = string
  default     = "fraud-detection-model"
  description = "GitHub repo name for the OIDC deploy-role trust policy."
}

variable "alert_email" {
  type        = string
  default     = ""
  description = "Email for billing-alarm notifications (empty = skip subscription)."
}

variable "billing_alarm_low_usd" {
  type        = number
  default     = 10
  description = "First estimated-charges alarm threshold (USD)."
}

variable "billing_alarm_high_usd" {
  type        = number
  default     = 25
  description = "Second estimated-charges alarm threshold (USD)."
}

variable "create_oidc_provider" {
  type        = bool
  default     = true
  description = "Create the GitHub OIDC provider (set false if it already exists in the account)."
}
