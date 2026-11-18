import pandas as pd
import pytest

from celtics.config import FIRST_TRAIN_SEASON
from celtics.export.league import migrate_pregame_log, update_league_pregame_log
from celtics.features.build import FEATURES, build_features
from celtics.models.margin import make_model
from tests.synth import make_league


@pytest.fixture(scope="module")
def season_feats_model():
    feats = build_features(make_league([2022, 2023], current=2023))
    train = feats[feats["completed"] & (feats["season"] >= FIRST_TRAIN_SEASON)]
    model = make_model("ridge").fit(train[FEATURES], train["margin"])
    return feats, model


def test_migrate_converts_boston_perspective_to_home():
    games = pd.DataFrame({"game_id": ["g1", "g2"], "home_team": ["BOS", "DEN"], "away_team": ["NYK", "BOS"]})
    old = {
        "g1": {"win_prob": 0.7, "pred_margin": 5.0, "logged_at": "t"},
        "g2": {"win_prob": 0.6, "pred_margin": 3.0, "logged_at": "t"},
        "gx": {"win_prob": 0.5, "pred_margin": 0.0, "logged_at": "t"},  # unknown game: dropped
    }
    out = migrate_pregame_log(old, games)
    assert out["g1"] == {"home_win_prob": 0.7, "pred_margin": 5.0, "logged_at": "t"}
    assert out["g2"] == {"home_win_prob": 0.4, "pred_margin": -3.0, "logged_at": "t"}
    assert "gx" not in out
    assert migrate_pregame_log(out, games) == out  # idempotent on new format


def test_league_log_covers_every_upcoming_game_and_freezes_past(season_feats_model):
    feats, model = season_feats_model  # synthetic league from tests.synth, ridge model
    log = update_league_pregame_log({}, feats, model, 2023, "2023-11-19T14:00:00", "2023-11-19")
    upcoming = feats[(feats["season"] == 2023) & ~feats["completed"] & (feats["date"] >= "2023-11-19")]
    assert set(log) == set(upcoming["game_id"])
    frozen = {"old": {"home_win_prob": 0.3, "pred_margin": -4.0, "logged_at": "x"}}
    assert update_league_pregame_log(frozen, feats, model, 2023, "2023-11-19T14:00:00", "2023-11-19")["old"] == frozen["old"]
