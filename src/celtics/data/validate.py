"""Integrity checks run before any modeling. A failure aborts the run."""

import pandas as pd

from celtics.teams import TEAMS


class DataValidationError(ValueError):
    """The fetched data is inconsistent; nothing should be published."""


def validate_games(
    games: pd.DataFrame,
    season: int,
    min_games: int = 80,
    max_games: int = 82,
    max_stale_days: int = 1,
) -> None:
    problems: list[str] = []

    dup = games["game_id"].duplicated()
    if dup.any():
        problems.append(f"duplicate game_ids: {games.loc[dup, 'game_id'].tolist()[:5]}")

    seen = set(games["home_team"]) | set(games["away_team"])
    unknown = seen - set(TEAMS)
    if unknown:
        problems.append(f"unknown teams: {sorted(unknown)}")

    played = games["home_pts"].notna() | games["away_pts"].notna()
    incomplete = games[played & games[["home_pts", "away_pts", "poss"]].isna().any(axis=1)]
    if len(incomplete):
        problems.append(f"completed games missing scores/possessions: {incomplete['game_id'].tolist()[:5]}")

    cur = games[games["season"] == season]
    cur_teams = set(cur["home_team"]) | set(cur["away_team"])
    if cur_teams != set(TEAMS):
        problems.append(f"season {season} has {len(cur_teams)} teams, expected 30")
    counts = pd.concat([cur["home_team"], cur["away_team"]]).value_counts()
    bad = counts[(counts < min_games) | (counts > max_games)]
    if len(bad):
        problems.append(f"games per team out of [{min_games}, {max_games}]: {bad.to_dict()}")

    # A postponed game is normally re-dated in the schedule, so an unplayed game dated well
    # before the latest result means its result is missing.
    cur_played = cur["home_pts"].notna() | cur["away_pts"].notna()
    if cur_played.any():
        cutoff = cur.loc[cur_played, "date"].max() - pd.Timedelta(days=max_stale_days)
        stale = cur[~cur_played & (cur["date"] < cutoff)]
        if len(stale):
            ids = stale["game_id"].tolist()
            problems.append(f"{len(ids)} past-dated games have no result: {ids[:5]}")

    if problems:
        raise DataValidationError("; ".join(problems))
