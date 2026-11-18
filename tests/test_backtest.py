import celtics.evaluate.backtest as bt
from celtics.features.build import build_features
from celtics.models.margin import make_model
from tests.synth import make_league


def test_walk_forward_only_trains_on_prior_seasons(monkeypatch):
    feats = build_features(make_league([2020, 2021, 2022, 2023]))
    seen = []

    def spy(name):
        model = make_model(name)
        original = model.fit

        def fit(X, y):
            seen.append(feats.loc[X.index, "season"].max())
            return original(X, y)

        model.fit = fit
        return model

    monkeypatch.setattr(bt, "make_model", spy)
    preds = bt.walk_forward(feats, "ridge", [2022, 2023])
    assert seen == [2021, 2022]
    assert len(preds) == 2 * 82 * 15
    assert set(preds["season"]) == {2022, 2023}
    assert preds["prob"].between(0, 1).all()
    assert set(preds["outcome"].unique()) <= {0.0, 1.0}


def test_walk_forward_beats_coin_flip_on_synthetic_league():
    feats = build_features(make_league([2020, 2021, 2022, 2023]))
    preds = bt.walk_forward(feats, "ridge", [2023])
    from celtics.evaluate.metrics import log_loss

    assert log_loss(preds["prob"], preds["outcome"]) < 0.69


def test_walk_forward_ridge_avail_runs_on_synthetic_league():
    feats = build_features(make_league([2020, 2021, 2022, 2023]))
    preds = bt.walk_forward(feats, "ridge_avail", [2023])
    assert len(preds) > 0
    assert preds["prob"].between(0, 1).all()
    assert (preds["prob"] > 0).all() and (preds["prob"] < 1).all()
