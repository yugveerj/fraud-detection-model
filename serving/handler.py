"""AWS Lambda handler for the scoring API (SPEC Section 5).

Routes (behind an API Gateway HTTP API):
* ``GET /healthz`` — model version, run id, calibration, operating threshold.
* ``POST /score`` — pydantic-validated transaction → calibrated probability, decision
  at the frozen operating point, and top-5 SHAP contributions.

The bundle (calibrated + base model, threshold, meta) and the SHAP explainer are
loaded once on cold start and cached. Model artifacts live in ``MODEL_DIR``
(default ``artifacts/serving``), baked into the container image at build time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
from pydantic import ValidationError

from serving import score as score_mod
from serving.models import HealthResponse, ScoreRequest, ScoreResponse

MODEL_DIR = Path(
    os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "artifacts" / "serving")
)
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

_STATE: dict = {}


def _load():
    """Load the bundle + build the explainer once (cached across warm invocations)."""
    if "bundle" not in _STATE:
        bundle = joblib.load(MODEL_DIR / "bundle.joblib")
        _STATE["bundle"] = bundle
        _STATE["explainer"] = score_mod.build_explainer(bundle["base"])
    return _STATE["bundle"], _STATE["explainer"]


def _cors_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


def _response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": _cors_headers(), "body": json.dumps(body)}


def _method_and_path(event: dict) -> tuple[str, str]:
    ctx = event.get("requestContext", {})
    http = ctx.get("http", {})
    method = http.get("method") or event.get("httpMethod") or "GET"
    path = http.get("path") or event.get("rawPath") or event.get("path") or "/"
    return method.upper(), path


def _parse_body(event: dict) -> dict:
    raw = event.get("body")
    if raw is None:
        return {}
    if event.get("isBase64Encoded"):
        import base64

        raw = base64.b64decode(raw).decode("utf-8")
    if isinstance(raw, dict):
        return raw
    return json.loads(raw) if raw else {}


def healthz() -> dict:
    bundle, _ = _load()
    meta = bundle.get("meta", {})
    payload = HealthResponse(
        status="ok",
        model_version=meta.get("version"),
        run_id=meta.get("run_id"),
        calibration=meta.get("calibration", "isotonic"),
        provenance=meta.get("provenance", "synthetic"),
        operating_threshold=bundle.get("threshold"),
        review_cost=bundle.get("review_cost"),
    )
    return _response(200, payload.model_dump())


def score_endpoint(event: dict) -> dict:
    try:
        body = _parse_body(event)
    except (json.JSONDecodeError, ValueError):
        return _response(400, {"error": "invalid JSON body"})
    try:
        request = ScoreRequest(**body)
    except ValidationError as exc:
        return _response(400, {"error": "validation failed", "detail": json.loads(exc.json())})

    bundle, explainer = _load()
    try:
        scored = score_mod.score(bundle, explainer, request.model_dump(exclude_none=True))
    except Exception as exc:  # noqa: BLE001 - any featurization/model error is bad input
        # extra="allow" lets arbitrary IEEE-CIS fields through; a hostile type (e.g.
        # a string in a numeric column) must be a 400, not an uncontrolled 5xx that
        # also trips the Lambda-error alarm.
        print(f"score error: {type(exc).__name__}: {exc}")  # -> CloudWatch logs
        return _response(400, {"error": "could not score transaction — check field types"})
    response = ScoreResponse(
        fraud_probability=scored["fraud_probability"],
        decision=scored["decision"],
        threshold=scored["threshold"],
        top_factors=scored["top_factors"],
        model_version=scored["model_version"],
        run_id=scored["run_id"],
        calibration=scored["calibration"],
        provenance=scored["provenance"],
    )
    return _response(200, response.model_dump())


def handler(event, context=None):  # noqa: ARG001 - Lambda signature
    method, path = _method_and_path(event)
    if method == "OPTIONS":
        return _response(200, {"ok": True})
    if path.endswith("/healthz"):
        return healthz()
    if path.endswith("/score"):
        if method != "POST":
            return _response(405, {"error": "use POST"})
        return score_endpoint(event)
    return _response(404, {"error": "not found", "path": path})
