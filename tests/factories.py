"""Hand-built inputs for unit tests."""

import numpy as np
import pandas as pd

from celtics.data.parse import GAME_COLUMNS


def gamelog_rows(game_id, date, home, away, home_pts, away_pts):
    """Two LeagueGameLog rows (home + away) with stats that give 100 possessions each."""
    base = {"GAME_ID": game_id, "GAME_DATE": date, "FGA": 90, "OREB": 10, "TOV": 15, "FTA": 11.363636}
    return [
        {**base, "TEAM_ABBREVIATION": home, "MATCHUP": f"{home} vs. {away}", "PTS": home_pts},
        {**base, "TEAM_ABBREVIATION": away, "MATCHUP": f"{away} @ {home}", "PTS": away_pts},
    ]


def schedule_payload(season_label, games):
    """Minimal scheduleLeagueV2.json. games: (game_id, 'YYYY-MM-DD', home, away)."""
    days: dict[str, list] = {}
    for game_id, date, home, away in games:
        days.setdefault(date, []).append(
            {
                "gameId": game_id,
                "gameDateEst": f"{date}T00:00:00Z",
                "gameStatus": 1,
                "homeTeam": {"teamTricode": home, "score": 0},
                "awayTeam": {"teamTricode": away, "score": 0},
            }
        )
    return {
        "leagueSchedule": {
            "seasonYear": season_label,
            "gameDates": [{"gameDate": d, "games": g} for d, g in days.items()],
        }
    }


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
