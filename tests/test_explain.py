"""SHAP explainability tests (slow: trains a model + runs TreeExplainer)."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import data, explain, modeling, schema


@pytest.fixture(scope="module")
def trained(featured_frame):
    sp = data.temporal_split(featured_frame, provenance="synthetic")
    y = sp.train[schema.TARGET].to_numpy()
    model = modeling.make_model("xgboost", y_train=y, profile="smoke").fit(sp.train, y)
    return model, sp


@pytest.mark.slow
def test_compute_shap_shapes(trained):
    model, sp = trained
    sample = sp.holdout.iloc[:200]
    sv, Xt, names, base = explain.compute_shap(model, sample)
    assert sv.shape[0] == len(sample)
    assert sv.shape[1] == len(names) == Xt.shape[1]
    assert np.isfinite(base)


@pytest.mark.slow
def test_global_importance_is_sorted_desc(trained):
    model, sp = trained
    sv, Xt, names, _ = explain.compute_shap(model, sp.holdout.iloc[:200])
    gi = explain.global_importance(sv, names, top_k=10)
    assert len(gi) == 10
    vals = gi["mean_abs_shap"].to_numpy()
    assert np.all(vals[:-1] >= vals[1:])


@pytest.mark.slow
def test_case_study_selection_and_factors(trained):
    model, sp = trained
    y = sp.holdout[schema.TARGET].to_numpy()
    proba = modeling.positive_proba(model, sp.holdout)
    amt = sp.holdout[schema.AMT_COL].to_numpy()
    sv, Xt, names, _ = explain.compute_shap(model, sp.holdout)
    sel = explain.select_case_studies(proba, y, threshold=0.5)
    assert set(sel) == {"fraud", "legit", "borderline"}
    assert y[sel["fraud"]] == 1  # selected fraud is an actual fraud
    assert y[sel["legit"]] == 0
    c = explain.case_study("fraud", sel["fraud"], sv, Xt, names, proba, y, amt)
    shaps = [abs(f["shap"]) for f in c.top_factors]
    assert shaps == sorted(shaps, reverse=True)  # top factors sorted by |shap|
