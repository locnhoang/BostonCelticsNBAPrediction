import numpy as np
import pandas as pd
import pytest

from celtics.features.build import FEATURES
from celtics.models.margin import MODELS, load_model, make_model, save_model, win_prob


def synthetic_xy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(0.0, index=range(n), columns=FEATURES)
    X["elo_diff"] = rng.normal(0, 150, n)
    X["net_diff"] = X["elo_diff"] / 28 + rng.normal(0, 2, n)
    X["home_rest"] = rng.integers(1, 5, n).astype(float)
    X["away_rest"] = rng.integers(1, 5, n).astype(float)
    y = pd.Series((X["elo_diff"] + 100) / 28 + rng.normal(0, 12, n))
    return X, y


def test_win_prob_symmetry():
    assert win_prob(0.0, 12.0) == pytest.approx(0.5)
    assert win_prob(5.0, 12.0) + win_prob(-5.0, 12.0) == pytest.approx(1.0)


def test_homecourt_predicts_mean_margin():
    X, y = synthetic_xy()
    m = make_model("homecourt").fit(X, y)
    assert np.allclose(m.predict_margin(X), y.mean())
    assert m.sigma == pytest.approx(y.std(), rel=1e-6)


def test_elo_model_formula():
    X, y = synthetic_xy()
    m = make_model("elo").fit(X, y)
    assert np.allclose(m.predict_margin(X), (X["elo_diff"] + 100) / 28)
    assert m.sigma == pytest.approx(12, rel=0.05)


@pytest.mark.parametrize("name", ["ridge", "gbm"])
def test_learned_models_recover_signal(name):
    X, y = synthetic_xy()
    m = make_model(name).fit(X, y)
    truth = (X["elo_diff"] + 100) / 28
    assert np.corrcoef(m.predict_margin(X), truth)[0, 1] > 0.9
    assert 11 < m.sigma < 13
    p = m.win_prob(X)
    assert ((p > 0) & (p < 1)).all()


def test_registry_and_roundtrip(tmp_path):
    assert set(MODELS) == {"homecourt", "elo", "ridge", "gbm"}
    X, y = synthetic_xy(500)
    m = make_model("ridge").fit(X, y)
    path = tmp_path / "ridge.joblib"
    save_model(m, path)
    loaded = load_model(path)
    assert np.allclose(loaded.predict_margin(X), m.predict_margin(X))
    assert loaded.sigma == m.sigma
