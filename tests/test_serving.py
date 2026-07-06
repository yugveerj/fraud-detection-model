"""Serving tests: feature-row build, scoring output hygiene, handler routing.

Trains a smoke model and injects the bundle into the handler's cache, so no AWS,
MLflow, or on-disk artifact is required.
"""

from __future__ import annotations

import json
import math

import pytest

from pipeline import data, decisions, modeling, schema
from serving import handler
from serving import score as score_mod
from serving.models import ScoreRequest


@pytest.fixture(scope="module")
def bundle(featured_frame):
    sp = data.temporal_split(featured_frame, provenance="synthetic")
    y_tr = sp.train[schema.TARGET].to_numpy()
    y_val = sp.val[schema.TARGET].to_numpy()
    base = modeling.make_model("xgboost", y_train=y_tr, profile="smoke").fit(sp.train, y_tr)
    cal = modeling.calibrate(base, sp.val, y_val, method="isotonic")
    op = decisions.optimize_operating_point(
        y_val, cal.predict_proba(sp.val)[:, 1], sp.val[schema.AMT_COL].to_numpy(), 25.0
    )
    return {
        "calibrated": cal,
        "base": base,
        "threshold": op["threshold"],
        "review_cost": 25.0,
        "meta": {
            "version": "test",
            "run_id": "test-run",
            "calibration": "isotonic",
            "provenance": "synthetic",
        },
    }


@pytest.fixture(scope="module")
def explainer(bundle):
    return score_mod.build_explainer(bundle["base"])


@pytest.fixture()
def wired_handler(bundle, explainer, monkeypatch):
    monkeypatch.setitem(handler._STATE, "bundle", bundle)
    monkeypatch.setitem(handler._STATE, "explainer", explainer)
    yield handler


PAYLOAD = {"TransactionDT": 500000, "TransactionAmt": 200.0, "ProductCD": "W", "card6": "credit"}


@pytest.mark.slow
def test_build_feature_row_dtypes():
    feat = score_mod.build_feature_row(PAYLOAD)
    assert len(feat) == 1
    # Categorical columns are object dtype even when absent (avoids the OrdinalEncoder crash).
    assert feat["card4"].dtype == object
    assert feat[schema.TIME_COL].dtype.kind in "iu"


@pytest.mark.slow
def test_score_output_is_json_clean(bundle, explainer):
    result = score_mod.score(bundle, explainer, PAYLOAD)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["decision"] in {"review", "approve"}
    assert len(result["top_factors"]) == 5
    # No NaN anywhere -> valid JSON (regression for the float32-NaN leak).
    dumped = json.dumps(result, allow_nan=False)
    assert "NaN" not in dumped
    for f in result["top_factors"]:
        assert f["value"] is None or math.isfinite(f["value"])


@pytest.mark.slow
def test_healthz(wired_handler):
    r = wired_handler.handler({"requestContext": {"http": {"method": "GET", "path": "/healthz"}}})
    assert r["statusCode"] == 200
    body = json.loads(r["body"])
    assert body["status"] == "ok"
    assert body["model_version"] == "test"
    assert body["operating_threshold"] is not None


@pytest.mark.slow
def test_score_endpoint_ok(wired_handler):
    event = {
        "requestContext": {"http": {"method": "POST", "path": "/score"}},
        "body": json.dumps(PAYLOAD),
    }
    r = wired_handler.handler(event)
    assert r["statusCode"] == 200
    body = json.loads(r["body"])
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["disclaimer"].startswith("Demonstration")


@pytest.mark.slow
def test_score_endpoint_validation_error(wired_handler):
    event = {
        "requestContext": {"http": {"method": "POST", "path": "/score"}},
        "body": json.dumps({"TransactionDT": 1, "TransactionAmt": -5}),  # amount must be > 0
    }
    r = wired_handler.handler(event)
    assert r["statusCode"] == 400


def test_unknown_route_and_method(bundle, explainer, monkeypatch):
    monkeypatch.setitem(handler._STATE, "bundle", bundle)
    monkeypatch.setitem(handler._STATE, "explainer", explainer)
    assert (
        handler.handler({"requestContext": {"http": {"method": "GET", "path": "/nope"}}})[
            "statusCode"
        ]
        == 404
    )
    wrong = handler.handler({"requestContext": {"http": {"method": "GET", "path": "/score"}}})
    assert wrong["statusCode"] == 405


def test_score_request_rejects_bad_amount():
    with pytest.raises(ValueError):
        ScoreRequest(TransactionDT=1, TransactionAmt=0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "hostile",
    [
        {"TransactionDT": 1, "TransactionAmt": 50, "V1": "abc"},  # string in numeric col
        {"TransactionDT": 1, "TransactionAmt": 50, "C1": [1, 2, 3]},  # array in numeric col
        {"TransactionDT": 1, "TransactionAmt": 1e308},  # non-finite after transforms
    ],
)
def test_hostile_but_valid_json_returns_400_not_5xx(wired_handler, hostile):
    """extra='allow' fields with hostile types must be 400, never an unguarded 5xx."""
    event = {
        "requestContext": {"http": {"method": "POST", "path": "/score"}},
        "body": json.dumps(hostile),
    }
    r = wired_handler.handler(event)
    assert r["statusCode"] == 400
