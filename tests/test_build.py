import numpy as np
import pandas as pd

from celtics.features.build import (
    FEATURES,
    as_of,
    build_features,
    current_team_state,
    matchup_frame,
    matchup_matrix,
)
from celtics.teams import TEAM_INDEX
from tests.synth import make_league


def test_build_features_shape_and_columns():
    games = make_league([2022, 2023], current=2023)
    feats = build_features(games)
    assert len(feats) == len(games)
    assert set(FEATURES) <= set(feats.columns)
    assert feats["completed"].sum() == 15 * 82 + 15 * 15
    assert feats.loc[feats["completed"], "margin"].notna().all()
    assert feats.loc[~feats["completed"], "margin"].isna().all()
    assert not feats[FEATURES].isna().any().any()


def test_no_future_leakage():
    games = make_league([2022, 2023])
    base = build_features(games)
    changed = games.copy()
    last = changed.index[-1]
    changed.loc[last, ["home_pts", "away_pts"]] = [200.0, 50.0]
    after = build_features(changed)
    # every game's pre-game features are unaffected by the final game's result
    pd.testing.assert_frame_equal(base[FEATURES], after[FEATURES])


def test_as_of_blanks_future_games():
    games = make_league([2022])
    cutoff = pd.Timestamp("2022-11-01")
    view = as_of(games, cutoff)
    assert view.loc[view["date"] > cutoff, "home_pts"].isna().all()
    assert view.loc[view["date"] <= cutoff, "home_pts"].notna().all()
    assert games["home_pts"].notna().all()  # original untouched


def test_team_state_and_matchup_matrix():
    games = make_league([2022, 2023], current=2023)
    state = current_team_state(games)
    assert len(state.elo) == 30 and len(state.net) == 30
    frame = matchup_frame(state, [("BOS", "NYK")])
    assert list(frame.columns) == FEATURES
    assert frame.loc[0, "elo_diff"] == state.elo["BOS"] - state.elo["NYK"]

    def predict(X):
        return X["elo_diff"].to_numpy() / 28 + 3

    m = matchup_matrix(predict, state)
    b, n = TEAM_INDEX["BOS"], TEAM_INDEX["NYK"]
    assert m.shape == (30, 30) and np.all(np.diag(m) == 0)
    assert m[b, n] == (state.elo["BOS"] - state.elo["NYK"]) / 28 + 3


def test_result_change_only_affects_later_dates():
    games = make_league([2022, 2023])
    base = build_features(games).set_index("game_id")
    changed = games.copy()
    mid = len(changed) // 2
    changed.loc[mid, ["home_pts", "away_pts"]] = [200.0, 50.0]
    cutoff = changed.loc[mid, "date"]
    after = build_features(changed).set_index("game_id")
    on_or_before = base.index[base["date"] <= cutoff]
    pd.testing.assert_frame_equal(base.loc[on_or_before, FEATURES], after.loc[on_or_before, FEATURES])
    later = base.index[base["date"] > cutoff]
    assert not base.loc[later, FEATURES].equals(after.loc[later, FEATURES])
