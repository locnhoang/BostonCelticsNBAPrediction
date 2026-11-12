import numpy as np
import pytest

from celtics.config import FIRST_TRAIN_SEASON
from celtics.features.build import FEATURES, build_features, current_team_state
from celtics.models.margin import make_model
from celtics.simulate.inputs import prepare_sim_inputs
from celtics.teams import TEAM_INDEX
from tests.synth import make_league


@pytest.fixture(scope="module")
def setup():
    games = make_league([2022, 2023], current=2023, completed_rounds=15)
    feats = build_features(games)
    train = feats[feats["completed"] & (feats["season"] >= FIRST_TRAIN_SEASON)]
    model = make_model("ridge").fit(train[FEATURES], train["margin"])
    return games, feats, model


def test_counts_are_consistent(setup):
    games, feats, model = setup
    inp = prepare_sim_inputs(feats, model, current_team_state(games), 2023)
    assert inp.current_wins.sum() == inp.current_losses.sum() == 15 * 15
    assert inp.h2h_wins.sum() == 15 * 15
    assert len(inp.rem_home) == len(inp.rem_margin) == 67 * 15
    assert np.all(inp.total_games == 82)
    assert np.all(inp.extra_games == 0)
    assert (inp.conf_wins + inp.conf_losses).sum() <= 2 * 15 * 15
    assert inp.neutral_margin.sum() == pytest.approx(0, abs=1e-6)
    assert inp.sigma == model.sigma


def test_missing_scheduled_games_become_extra_games(setup):
    games, feats, model = setup
    rem = feats[(feats["season"] == 2023) & ~feats["completed"]]
    dropped = rem.iloc[0]
    trimmed = feats.drop(index=rem.index[0])
    inp = prepare_sim_inputs(trimmed, model, current_team_state(games), 2023)
    assert inp.extra_games.sum() == 2
    assert np.all(inp.total_games == 82)
    assert inp.extra_games[TEAM_INDEX[dropped["home_team"]]] == 1


def test_no_extra_games_when_target_is_none(setup):
    games, feats, model = setup
    trimmed = feats.drop(index=feats[~feats["completed"]].index[0])
    inp = prepare_sim_inputs(trimmed, model, current_team_state(games), 2023, target_games=None)
    assert inp.extra_games.sum() == 0


def test_h2h_orientation_winner_is_row(setup):
    _, _, model = setup
    games = make_league([2022, 2023], current=2023, completed_rounds=1)  # each pair meets at most once
    feats = build_features(games)
    done = feats[(feats["season"] == 2023) & feats["completed"]].iloc[0]
    home_won = done["home_pts"] > done["away_pts"]
    winner, loser = (
        (done["home_team"], done["away_team"]) if home_won else (done["away_team"], done["home_team"])
    )
    inp = prepare_sim_inputs(feats, model, current_team_state(games), 2023)
    assert inp.h2h_wins[TEAM_INDEX[winner], TEAM_INDEX[loser]] == 1
    assert inp.h2h_wins[TEAM_INDEX[loser], TEAM_INDEX[winner]] == 0
