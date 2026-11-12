from datetime import date, timedelta

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


def _blank(g, mask):
    g.loc[mask, ["home_pts", "away_pts", "poss"]] = np.nan


def _stale_league(n_stale):
    """10 rounds played (day 0..9, every other day from Oct 20); blank n_stale old games."""
    g = league_season(played=10)
    _blank(g, g.index.isin(g.index[:n_stale]))
    return g


TODAY = date(2026, 11, 8)  # day after the last played synthetic round (Nov 7)


def test_three_stale_games_are_tolerated_as_postponements():
    validate_games(_stale_league(3), 2026, today=TODAY)


def test_four_stale_games_fail():
    with pytest.raises(DataValidationError, match=r"4 past-dated games have no result \(allowing 3 postponements\): \['"):
        validate_games(_stale_league(4), 2026, today=TODAY)


def test_log_cut_off_at_the_end_fails():
    g = league_season(played=10)
    last_two = sorted(g.loc[g["home_pts"].notna(), "date"].unique())[-2:]
    _blank(g, g["date"].isin(last_two))
    today = pd.Timestamp(last_two[-1]).date() + timedelta(days=5)
    with pytest.raises(DataValidationError, match="past-dated games have no result"):
        validate_games(g, 2026, today=today)


def test_stale_check_skipped_without_today():
    validate_games(_stale_league(30), 2026)


def test_games_inside_grace_window_pass():
    g = league_season(played=10)
    last = g.loc[g["home_pts"].notna(), "date"].max()
    _blank(g, g["date"] == last)  # 15 unplayed games dated 2 days before today
    validate_games(g, 2026, today=(last + pd.Timedelta(days=2)).date())
    with pytest.raises(DataValidationError):
        validate_games(g, 2026, today=(last + pd.Timedelta(days=3)).date())
