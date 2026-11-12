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


def schedule_to_games(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    if raw.empty:
        return empty_games()
    found = set(raw["seasonYear"].astype(str))
    if found != {season_label(season)}:
        raise ValueError(f"schedule is for {sorted(found)}, expected {season_label(season)}")
    df = pd.DataFrame(
        {
            "game_id": raw["gameId"].astype(str).to_numpy(),
            "home_team": raw["homeTeam_teamTricode"].fillna("").astype(str).to_numpy(),
            "away_team": raw["awayTeam_teamTricode"].fillna("").astype(str).to_numpy(),
            "date_str": raw["gameDateEst"].astype(str).str[:10].to_numpy(),
        }
    )
    keep = df["game_id"].str.startswith("002") & (df["home_team"] != "") & (df["away_team"] != "")
    df = df[keep]
    if df.empty:
        return empty_games()
    games = pd.DataFrame(
        {
            "game_id": df["game_id"].to_numpy(),
            "season": season,
            "date": pd.to_datetime(df["date_str"]).to_numpy(),
            "home_team": df["home_team"].to_numpy(),
            "away_team": df["away_team"].to_numpy(),
            "home_pts": np.nan,
            "away_pts": np.nan,
            "poss": np.nan,
        }
    )
    return _sorted(games.drop_duplicates("game_id"))


def merge_schedule_results(schedule: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    unplayed = schedule[~schedule["game_id"].isin(results["game_id"])]
    frames = [f for f in (results, unplayed) if not f.empty]
    if not frames:
        return empty_games()
    return _sorted(pd.concat(frames, ignore_index=True))
