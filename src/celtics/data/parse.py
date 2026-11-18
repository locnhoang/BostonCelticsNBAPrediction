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
    counts = df["GAME_ID"].value_counts()
    bad = sorted(counts[counts != 2].index)
    if bad:
        raise ValueError(f"games without exactly 2 log rows: {bad[:5]}")
    df["is_home"] = df["MATCHUP"].str.contains(" vs. ", regex=False)
    # Neutral-site games have "@" on both rows. Pick home as the team whose abbreviation
    # sorts second; merge_schedule_results corrects this from the schedule when available.
    df = df.sort_values(["GAME_ID", "is_home", "TEAM_ABBREVIATION"], ascending=[True, True, True])
    pairs = df.groupby("GAME_ID", sort=True)
    away = pairs.nth(0).set_index("GAME_ID")
    home = pairs.nth(1).set_index("GAME_ID")
    games = pd.DataFrame(
        {
            "game_id": home.index.astype(str),
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


def _align_results_to_schedule(schedule: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Take home/away from the schedule; swap points if the results had them reversed."""
    if results.empty:
        return results
    sched = schedule.set_index("game_id")[["home_team", "away_team"]]
    res = results.set_index("game_id", drop=False)
    common = res.index.intersection(sched.index)
    if common.empty:
        return results
    r, sc = res.loc[common], sched.loc[common]
    same = (r["home_team"] == sc["home_team"]) & (r["away_team"] == sc["away_team"])
    swapped = (r["home_team"] == sc["away_team"]) & (r["away_team"] == sc["home_team"])
    mismatch = common[~(same | swapped).to_numpy()]
    if len(mismatch):
        raise ValueError(f"result teams do not match schedule for games: {list(mismatch[:5])}")
    out = res.copy()
    flip = common[swapped.to_numpy()]
    out.loc[flip, ["home_pts", "away_pts"]] = res.loc[flip, ["away_pts", "home_pts"]].to_numpy()
    out.loc[common, "home_team"] = sc["home_team"]
    out.loc[common, "away_team"] = sc["away_team"]
    return out.reset_index(drop=True)


def merge_schedule_results(schedule: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    results = _align_results_to_schedule(schedule, results)
    unplayed = schedule[~schedule["game_id"].isin(results["game_id"])]
    frames = [f for f in (results, unplayed) if not f.empty]
    if not frames:
        return empty_games()
    return _sorted(pd.concat(frames, ignore_index=True))


PLAYER_COLUMNS = [
    "game_id", "season", "date", "team", "player_id", "player_name",
    "min", "pts", "fgm", "fga", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
]  # fmt: skip
_PLAYER_STATS = ["pts", "fgm", "fga", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf"]


def empty_players() -> pd.DataFrame:
    dtypes = {"game_id": str, "season": np.int64, "date": "datetime64[ns]", "team": str,
              "player_id": np.int64, "player_name": str, "min": float}  # fmt: skip
    dtypes.update({c: float for c in _PLAYER_STATS})
    return pd.DataFrame({c: pd.Series(dtype=dtypes[c]) for c in PLAYER_COLUMNS})


def playerlog_to_frame(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    if raw.empty:
        return empty_players()
    df = raw.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    df = df[df["GAME_ID"].str.startswith("002")]
    if df.empty:
        return empty_players()
    out = pd.DataFrame(
        {
            "game_id": df["GAME_ID"].to_numpy(),
            "season": np.int64(season),
            "date": pd.to_datetime(df["GAME_DATE"]).dt.normalize().to_numpy(),
            "team": df["TEAM_ABBREVIATION"].astype(str).to_numpy(),
            "player_id": df["PLAYER_ID"].astype("int64").to_numpy(),
            "player_name": df["PLAYER_NAME"].astype(str).to_numpy(),
            "min": df["MIN"].astype(float).fillna(0.0).to_numpy(),
        }
    )
    for col in _PLAYER_STATS:
        out[col] = df[col.upper()].astype(float).fillna(0.0).to_numpy()
    return out.sort_values(["date", "game_id", "team", "player_id"]).reset_index(drop=True)[PLAYER_COLUMNS]
