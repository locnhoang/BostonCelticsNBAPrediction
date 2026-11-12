import pandas as pd

from celtics.evaluate.backtest import checkpoint_date, season_coverage
from tests.synth import make_league


def test_checkpoint_date_is_15th_game():
    games = make_league([2022])
    assert checkpoint_date(games[games["season"] == 2022]) == pd.Timestamp("2022-10-20") + pd.Timedelta(days=28)


def test_coverage_in_range_and_wider_with_more_uncertainty():
    games = make_league([2020, 2021, 2022])
    cov = season_coverage(games, "ridge", [2022], [0.0, 6.0], n_sims=300)
    assert set(cov) == {0.0, 6.0}
    assert all(0.0 <= v <= 1.0 for v in cov.values())
    assert cov[6.0] >= cov[0.0]
