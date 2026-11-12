"""Hand-built inputs for unit tests."""

import numpy as np
import pandas as pd

from celtics.data.parse import GAME_COLUMNS


def gamelog_rows(game_id, date, home, away, home_pts, away_pts, neutral=False):
    """Two LeagueGameLog rows (home + away) with stats that give 100 possessions each.

    neutral=True mimics a neutral-site game: both MATCHUPs use "@".
    """
    base = {"GAME_ID": game_id, "GAME_DATE": date, "FGA": 90, "OREB": 10, "TOV": 15, "FTA": 11.363636}
    home_matchup = f"{home} @ {away}" if neutral else f"{home} vs. {away}"
    return [
        {**base, "TEAM_ABBREVIATION": home, "MATCHUP": home_matchup, "PTS": home_pts},
        {**base, "TEAM_ABBREVIATION": away, "MATCHUP": f"{away} @ {home}", "PTS": away_pts},
    ]


def schedule_payload(season_label, games):
    """Minimal ScheduleLeagueV2 frame. games: (game_id, 'YYYY-MM-DD', home, away)."""
    return pd.DataFrame(
        {
            "seasonYear": [season_label] * len(games),
            "gameId": [g[0] for g in games],
            "gameDateEst": [f"{g[1]}T00:00:00Z" for g in games],
            "gameStatus": [1] * len(games),
            "homeTeam_teamTricode": [g[2] for g in games],
            "awayTeam_teamTricode": [g[3] for g in games],
        },
        columns=[
            "seasonYear",
            "gameId",
            "gameDateEst",
            "gameStatus",
            "homeTeam_teamTricode",
            "awayTeam_teamTricode",
        ],
    )


def make_games(rows):
    """rows: (game_id, season, 'YYYY-MM-DD', home, away, home_pts|None, away_pts|None)."""
    records = []
    for game_id, season, date, home, away, hp, ap in rows:
        played = hp is not None
        records.append(
            {
                "game_id": game_id,
                "season": season,
                "date": pd.Timestamp(date),
                "home_team": home,
                "away_team": away,
                "home_pts": float(hp) if played else np.nan,
                "away_pts": float(ap) if played else np.nan,
                "poss": 100.0 if played else np.nan,
            }
        )
    return pd.DataFrame(records, columns=GAME_COLUMNS)
