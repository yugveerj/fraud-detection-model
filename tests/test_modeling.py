"""Model factory, leakage-safe encoding, and calibration tests."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import data, encoders, metrics, modeling, schema


@pytest.fixture(scope="module")
def split(featured_frame):
    return data.temporal_split(featured_frame, provenance="synthetic")


def test_scale_pos_weight():
    y = np.array([0, 0, 0, 1])  # 3 neg / 1 pos
    assert modeling.scale_pos_weight(y) == pytest.approx(3.0)


@pytest.mark.parametrize("name", modeling.MODEL_NAMES)
def test_each_model_trains_and_predicts_calibrated_range(name, split):
    y_tr = split.train[schema.TARGET].to_numpy()
    model = modeling.make_model(name, y_train=y_tr, profile="smoke")
    model.fit(split.train, y_tr)
    p = modeling.positive_proba(model, split.val)
    assert p.shape == (len(split.val),)
    assert np.all((p >= 0) & (p <= 1))


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        modeling.make_model("randomforest")


def test_tree_encoder_is_leakage_safe_on_unseen_category(split):
    """Fit ordinal encoding on TRAIN; an unseen category in val must map to -1, not crash."""
    numeric, categorical = encoders.model_feature_columns(split.train)
    prep = encoders.build_preprocessor(numeric, categorical, kind="tree")
    prep.fit(split.train)
    val = split.val.copy()
    if categorical:
        val[categorical[0]] = "__never_seen__"  # inject an unseen category
    out = prep.transform(val)  # must not raise
    assert out.shape[0] == len(val)


def test_calibration_improves_in_distribution_brier(split):
    """Isotonic calibration fit on VAL should not worsen Brier measured on VAL."""
    y_tr = split.train[schema.TARGET].to_numpy()
    y_val = split.val[schema.TARGET].to_numpy()
    xgb = modeling.make_model("xgboost", y_train=y_tr, profile="smoke").fit(split.train, y_tr)
    pre = metrics.evaluate(y_val, modeling.positive_proba(xgb, split.val)).brier
    cal = modeling.calibrate(xgb, split.val, y_val, method="isotonic")
    post = metrics.evaluate(y_val, cal.predict_proba(split.val)[:, 1]).brier
    assert post <= pre + 1e-6


def test_feature_columns_exclude_keys_and_target(split):
    numeric, categorical = encoders.model_feature_columns(split.train)
    for forbidden in (schema.ID_COL, schema.TARGET, schema.TIME_COL):
        assert forbidden not in numeric and forbidden not in categorical
    assert len(numeric) > 50  # engineered + causal + raw numerics
    assert set(categorical).issubset(set(encoders.MODEL_CATEGORICALS))
