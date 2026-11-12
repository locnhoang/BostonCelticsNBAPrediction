import pytest

from celtics.features.ratings import compute_net_ratings, compute_rest
from tests.factories import make_games


def test_first_game_uses_prior_zero_then_updates():
    games = make_games(
        [
            ("g1", 2020, "2020-12-22", "BOS", "NYK", 110, 100),
            ("g2", 2020, "2020-12-24", "BOS", "MIA", None, None),
        ]
    )
    pre, current = compute_net_ratings(games)
    assert pre.loc[0, "home_net_pre"] == 0.0
    # x = +10 per 100; num = 0.97*0 + 10, den = 0.97*8 + 1
    expected = 10 / (0.97 * 8 + 1)
    assert pre.loc[1, "home_net_pre"] == pytest.approx(expected)
    assert current["BOS"] == pytest.approx(expected)
    assert current["NYK"] == pytest.approx(-expected)


def test_prior_comes_from_last_season_average():
    games = make_games(
        [
            ("g1", 2020, "2021-01-01", "BOS", "NYK", 110, 100),
            ("g2", 2021, "2021-10-20", "BOS", "NYK", None, None),
        ]
    )
    pre, _ = compute_net_ratings(games)
    assert pre.loc[1, "home_net_pre"] == pytest.approx(0.67 * 10)
    assert pre.loc[1, "away_net_pre"] == pytest.approx(0.67 * -10)


def test_rest_back_to_back_and_three_in_four():
    games = make_games(
        [
            ("g1", 2020, "2020-12-01", "BOS", "NYK", 100, 90),
            ("g2", 2020, "2020-12-02", "MIA", "BOS", 100, 90),
            ("g3", 2020, "2020-12-04", "BOS", "CHI", 100, 90),
            ("g4", 2020, "2020-12-05", "BOS", "ATL", None, None),
        ]
    )
    rest = compute_rest(games).set_index("game_id")
    assert rest.loc["g1", "home_rest"] == 4  # season opener
    assert rest.loc["g2", "away_rest"] == 1 and rest.loc["g2", "away_b2b"] == 1
    assert rest.loc["g3", "home_rest"] == 2 and rest.loc["g3", "home_b2b"] == 0
    assert rest.loc["g4", "home_b2b"] == 1
    assert rest.loc["g4", "home_3in4"] == 1  # Dec 2, 4, 5
    assert rest.loc["g3", "home_3in4"] == 1  # Dec 1, 2, 4
    assert rest.loc["g2", "home_3in4"] == 0
