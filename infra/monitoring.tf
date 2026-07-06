# --- Billing alarms (cost guard, SPEC Section 5 + Section 9) --------------------
# CloudWatch billing metrics live in us-east-1; requires "Receive Billing Alerts"
# enabled once in the account (Billing console preferences).
resource "aws_sns_topic" "billing" {
  provider = aws.billing
  name     = "${var.project}-billing-alerts"
}

resource "aws_sns_topic_subscription" "billing_email" {
  count     = var.alert_email == "" ? 0 : 1
  provider  = aws.billing
  topic_arn = aws_sns_topic.billing.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_metric_alarm" "billing_low" {
  provider            = aws.billing
  alarm_name          = "${var.project}-estimated-charges-${var.billing_alarm_low_usd}usd"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = 21600 # 6h (billing metric cadence)
  statistic           = "Maximum"
  threshold           = var.billing_alarm_low_usd
  alarm_description   = "Estimated charges exceeded $${var.billing_alarm_low_usd}."
  dimensions          = { Currency = "USD" }
  alarm_actions       = [aws_sns_topic.billing.arn]
}

resource "aws_cloudwatch_metric_alarm" "billing_high" {
  provider            = aws.billing
  alarm_name          = "${var.project}-estimated-charges-${var.billing_alarm_high_usd}usd"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = 21600
  statistic           = "Maximum"
  threshold           = var.billing_alarm_high_usd
  alarm_description   = "Estimated charges exceeded $${var.billing_alarm_high_usd}."
  dimensions          = { Currency = "USD" }
  alarm_actions       = [aws_sns_topic.billing.arn]
}

# --- Operational alarm: Lambda errors ------------------------------------------
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${var.project}-lambda-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Scoring Lambda returned errors."
  dimensions          = { FunctionName = aws_lambda_function.this.function_name }
  treat_missing_data  = "notBreaching"
}
