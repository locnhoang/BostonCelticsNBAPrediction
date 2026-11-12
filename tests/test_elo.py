import pytest

from celtics.features.elo import compute_elo
from tests.factories import make_games


def test_first_game_update_matches_formula():
    games = make_games([("g1", 2020, "2020-12-22", "BOS", "NYK", 110, 100)])
    pre, ratings = compute_elo(games)
    assert pre.loc[0, "home_elo_pre"] == 1500 and pre.loc[0, "away_elo_pre"] == 1500
    expected_home = 1 / (1 + 10 ** (-100 / 400))
    multiplier = (13**0.8) / (7.5 + 0.006 * 100)
    delta = 20 * multiplier * (1 - expected_home)
    assert ratings["BOS"] == pytest.approx(1500 + delta)
    assert ratings["NYK"] == pytest.approx(1500 - delta)


def test_updates_are_zero_sum_and_used_as_next_pre_rating():
    games = make_games(
        [
            ("g1", 2020, "2020-12-22", "BOS", "NYK", 110, 100),
            ("g2", 2020, "2020-12-24", "NYK", "BOS", 90, 120),
        ]
    )
    pre, ratings = compute_elo(games)
    assert pre.loc[1, "away_elo_pre"] > 1500  # BOS carries its win into game 2
    assert sum(ratings.values()) == pytest.approx(3000)


def test_unplayed_game_has_pre_rating_but_no_update():
    games = make_games(
        [
            ("g1", 2020, "2020-12-22", "BOS", "NYK", 110, 100),
            ("g2", 2020, "2020-12-24", "BOS", "NYK", None, None),
        ]
    )
    pre, ratings = compute_elo(games)
    assert pre.loc[1, "home_elo_pre"] == pytest.approx(ratings["BOS"])


def test_season_carryover_regresses_25_percent():
    rows = [("g1", 2020, "2021-01-01", "BOS", "NYK", 150, 50)]
    rows.append(("g2", 2021, "2021-10-20", "BOS", "NYK", None, None))
    pre, ratings = compute_elo(make_games(rows))
    _, end_2020 = compute_elo(make_games(rows[:1]))
    assert pre.loc[1, "home_elo_pre"] == pytest.approx(0.75 * end_2020["BOS"] + 0.25 * 1500)
    assert ratings["BOS"] == pytest.approx(pre.loc[1, "home_elo_pre"])
