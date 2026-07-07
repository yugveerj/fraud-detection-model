resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.project}"
  retention_in_days = 14
}

resource "aws_lambda_function" "this" {
  function_name = var.project
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.this.repository_url}:${var.image_tag}"
  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout_s
  architectures = ["arm64"] # Graviton — matches the arm64 build + ~20% cheaper

  environment {
    variables = {
      ALLOWED_ORIGIN = var.pages_origin
      MODEL_DIR      = "/var/task/artifacts/serving"
      # Lambda cold-start hardening: writable caches + single-threaded numeric libs
      # (thread contention on a small container hurts more than it helps here).
      NUMBA_CACHE_DIR      = "/tmp"
      MPLCONFIGDIR         = "/tmp"
      OMP_NUM_THREADS      = "2"
      OPENBLAS_NUM_THREADS = "1"
      MKL_NUM_THREADS      = "1"
      NUMBA_NUM_THREADS    = "1"
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.lambda_logs,
    aws_cloudwatch_log_group.lambda,
  ]

  # The image is pushed by the deploy workflow; ignore drift on the tag digest so
  # `terraform apply` doesn't fight the CI image push.
  lifecycle {
    ignore_changes = [image_uri]
  }
}
