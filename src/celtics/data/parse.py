"""Convert raw NBA payloads into the canonical games frame."""

import numpy as np
import pandas as pd

from celtics.config import season_label

GAME_COLUMNS = ["game_id", "season", "date", "home_team", "away_team", "home_pts", "away_pts", "poss"]


def empty_games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": pd.Series(dtype=str),
            "season": pd.Series(dtype=np.int64),
            "date": pd.Series(dtype="datetime64[ns]"),
            "home_team": pd.Series(dtype=str),
            "away_team": pd.Series(dtype=str),
            "home_pts": pd.Series(dtype=float),
            "away_pts": pd.Series(dtype=float),
            "poss": pd.Series(dtype=float),
        }
    )


def _sorted(games: pd.DataFrame) -> pd.DataFrame:
    return games.sort_values(["date", "game_id"]).reset_index(drop=True)[GAME_COLUMNS]


def gamelog_to_games(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    if raw.empty:
        return empty_games()
    df = raw.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    df = df[df["GAME_ID"].str.startswith("002")].copy()
    df["poss"] = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]
    is_home = df["MATCHUP"].str.contains(" vs. ", regex=False)
    home = df[is_home].set_index("GAME_ID")
    away = df[~is_home].set_index("GAME_ID")
    ids = home.index.intersection(away.index)
    home, away = home.loc[ids], away.loc[ids]
    games = pd.DataFrame(
        {
            "game_id": ids.astype(str),
            "season": season,
            "date": pd.to_datetime(home["GAME_DATE"]).dt.normalize().to_numpy(),
            "home_team": home["TEAM_ABBREVIATION"].to_numpy(),
            "away_team": away["TEAM_ABBREVIATION"].to_numpy(),
            "home_pts": home["PTS"].astype(float).to_numpy(),
            "away_pts": away["PTS"].astype(float).to_numpy(),
            "poss": ((home["poss"] + away["poss"]) / 2).astype(float).to_numpy(),
        }
    )
    return _sorted(games)


def schedule_to_games(raw: dict, season: int) -> pd.DataFrame:
    sched = raw["leagueSchedule"]
    if sched.get("seasonYear") != season_label(season):
        raise ValueError(
            f"schedule is for {sched.get('seasonYear')}, expected {season_label(season)}"
        )
    rows = []
    for day in sched["gameDates"]:
        for g in day["games"]:
            game_id = str(g["gameId"])
            home = g["homeTeam"].get("teamTricode") or ""
            away = g["awayTeam"].get("teamTricode") or ""
            if not game_id.startswith("002") or not home or not away:
                continue
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "date": pd.Timestamp(g["gameDateEst"][:10]),
                    "home_team": home,
                    "away_team": away,
                    "home_pts": np.nan,
                    "away_pts": np.nan,
                    "poss": np.nan,
                }
            )
    if not rows:
        return empty_games()
    return _sorted(pd.DataFrame(rows).drop_duplicates("game_id"))


def merge_schedule_results(schedule: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    unplayed = schedule[~schedule["game_id"].isin(results["game_id"])]
    frames = [f for f in (results, unplayed) if not f.empty]
    if not frames:
        return empty_games()
    return _sorted(pd.concat(frames, ignore_index=True))
