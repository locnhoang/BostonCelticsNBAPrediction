import numpy as np
import pandas as pd
import pytest

from celtics.data.validate import DataValidationError, validate_games
from celtics.teams import TEAMS


def league_season(season=2026, games_per_team=82, played=10):
    """Round-robin-ish schedule where every team plays games_per_team games."""
    rows = []
    rng = np.random.default_rng(0)
    start = pd.Timestamp(f"{season}-10-20")
    for day in range(games_per_team):
        order = rng.permutation(TEAMS)
        for k in range(15):
            done = day < played
            rows.append(
                {
                    "game_id": f"002{season % 100:02d}{day:03d}{k:02d}",
                    "season": season,
                    "date": start + pd.Timedelta(days=2 * day),
                    "home_team": order[2 * k],
                    "away_team": order[2 * k + 1],
                    "home_pts": 110.0 if done else np.nan,
                    "away_pts": 100.0 if done else np.nan,
                    "poss": 100.0 if done else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_valid_season_passes():
    validate_games(league_season(), 2026)


def test_duplicate_ids_fail():
    g = league_season()
    g = pd.concat([g, g.iloc[[0]]])
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_games(g, 2026)


def test_unknown_team_fails():
    g = league_season()
    g.loc[0, "home_team"] = "SEA"
    with pytest.raises(DataValidationError, match="unknown teams"):
        validate_games(g, 2026)


def test_completed_game_missing_possessions_fails():
    g = league_season()
    g.loc[0, "poss"] = np.nan
    with pytest.raises(DataValidationError, match="missing"):
        validate_games(g, 2026)


def test_too_few_games_fails():
    with pytest.raises(DataValidationError, match="games per team"):
        validate_games(league_season(games_per_team=70), 2026)


def test_eighty_games_allowed_before_cup_games_are_scheduled():
    validate_games(league_season(games_per_team=80), 2026)
