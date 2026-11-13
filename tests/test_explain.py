import numpy as np
import pandas as pd
import pytest

from celtics.features.build import FEATURES
from celtics.models.explain import DRIVER_ORDER, explain
from celtics.models.margin import make_model


def synthetic_xy(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(0.0, index=range(n), columns=FEATURES)
    X["elo_diff"] = rng.normal(0, 150, n)
    X["net_diff"] = X["elo_diff"] / 28 + rng.normal(0, 2, n)
    X["home_rest"] = rng.integers(1, 5, n).astype(float)
    X["away_rest"] = rng.integers(1, 5, n).astype(float)
    X["home_b2b"] = (X["home_rest"] == 1).astype(float)
    X["away_b2b"] = (X["away_rest"] == 1).astype(float)
    y = pd.Series((X["elo_diff"] + 100) / 28 + 0.8 * (X["home_rest"] - X["away_rest"]) + rng.normal(0, 12, n))
    return X, y


def test_ridge_parts_sum_to_prediction():
    X, y = synthetic_xy()
    model = make_model("ridge").fit(X, y)
    parts = explain(model, X.iloc[:50])
    assert list(parts.columns) == DRIVER_ORDER
    assert list(parts.index) == list(X.index[:50])
    np.testing.assert_allclose(parts.sum(axis=1), model.predict_margin(X.iloc[:50]), atol=1e-9)


def test_home_court_is_equal_teams_on_normal_rest():
    X, y = synthetic_xy()
    model = make_model("elo").fit(X, y)
    parts = explain(model, X.iloc[:3])
    assert parts["Home court"].tolist() == pytest.approx([100 / 28] * 3)
    assert parts["Season strength"].tolist() == pytest.approx((X["elo_diff"].iloc[:3] / 28).tolist())
    assert parts["Recent form"].abs().max() == 0
    assert parts["Rest & schedule"].abs().max() == 0


def test_rest_part_is_zero_on_normal_rest():
    X, y = synthetic_xy()
    model = make_model("ridge").fit(X, y)
    row = X.iloc[[0]].copy()
    row[["home_rest", "away_rest"]] = 2.0
    row[["home_b2b", "away_b2b", "home_3in4", "away_3in4"]] = 0.0
    assert explain(model, row)["Rest & schedule"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_nonlinear_model_has_no_breakdown():
    X, y = synthetic_xy(500)
    model = make_model("gbm").fit(X, y)
    assert model.is_linear is False
    assert explain(model, X.iloc[:5]) is None


def test_homecourt_model_explains_as_all_home_court():
    X, y = synthetic_xy(500)
    model = make_model("homecourt").fit(X, y)
    parts = explain(model, X.iloc[:4])
    assert parts["Home court"].tolist() == pytest.approx([model.mean_] * 4)
    for col in DRIVER_ORDER:
        if col != "Home court":
            assert parts[col].abs().max() == 0
