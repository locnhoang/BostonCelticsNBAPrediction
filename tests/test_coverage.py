import pandas as pd

import celtics.evaluate.backtest as backtest
from celtics.evaluate.backtest import checkpoint_date, season_coverage
from celtics.teams import TEAM_INDEX
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


def _synthetic_players(games):
    rows = [
        {
            "game_id": g.game_id, "season": g.season, "date": g.date, "team": team,
            "player_id": TEAM_INDEX[team] * 100 + k, "player_name": f"P{team}{k}", "min": 30.0,
            "pts": 10.0 + k, "fgm": 4.0, "fga": 9.0, "ftm": 2.0, "fta": 2.0, "oreb": 1.0, "dreb": 3.0,
            "ast": 2.0, "stl": 1.0, "blk": 0.0, "tov": 1.0, "pf": 2.0,
        }
        for g in games[games["home_pts"].notna()].itertuples()
        for team in (g.home_team, g.away_team)
        for k in range(8)
    ]  # fmt: skip
    return pd.DataFrame(rows)


def test_season_coverage_ignores_player_rows_after_the_checkpoint(monkeypatch):
    games = make_league([2020, 2021, 2022])
    players = _synthetic_players(games)
    cutoff = checkpoint_date(games[games["season"] == 2022])
    seen = []
    real = backtest.build_features

    def spy(g, p=None):
        if p is not None:
            seen.append(p)
        return real(g, p)

    monkeypatch.setattr(backtest, "build_features", spy)
    tweaked = players.copy()
    late = (tweaked["season"] == 2022) & (tweaked["date"] > cutoff)
    assert late.any()
    tweaked.loc[late, "pts"] = 99.0
    a = season_coverage(games, "ridge_avail", [2022], [6.0], n_sims=50, players=players)
    b = season_coverage(games, "ridge_avail", [2022], [6.0], n_sims=50, players=tweaked)
    assert a == b
    assert len(seen) == 2 and seen[0].equals(seen[1])
    assert seen[0]["date"].max() <= cutoff
