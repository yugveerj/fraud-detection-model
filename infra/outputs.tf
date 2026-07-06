output "api_endpoint" {
  value       = aws_apigatewayv2_stage.default.invoke_url
  description = "Base URL of the scoring API (append /healthz or /score)."
}

output "ecr_repository_url" {
  value       = aws_ecr_repository.this.repository_url
  description = "ECR repo the deploy workflow pushes the image to."
}

output "lambda_function_name" {
  value       = aws_lambda_function.this.function_name
  description = "Lambda function the deploy workflow updates."
}

output "deploy_role_arn" {
  value       = aws_iam_role.deploy.arn
  description = "Role GitHub Actions assumes via OIDC (set as a repo secret/var)."
}
