"""Integrity checks run before any modeling. A failure aborts the run."""

from datetime import date, timedelta

import pandas as pd

from celtics.teams import TEAMS


class DataValidationError(ValueError):
    """The fetched data is inconsistent; nothing should be published."""


def validate_games(
    games: pd.DataFrame,
    season: int,
    min_games: int = 80,
    max_games: int = 82,
    today: date | None = None,
    stale_grace_days: int = 2,
    max_stale_games: int = 3,
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
        problems.append(
            f"completed games missing scores/possessions: {incomplete['game_id'].tolist()[:5]}"
        )

    cur = games[games["season"] == season]
    cur_teams = set(cur["home_team"]) | set(cur["away_team"])
    if cur_teams != set(TEAMS):
        problems.append(f"season {season} has {len(cur_teams)} teams, expected 30")
    counts = pd.concat([cur["home_team"], cur["away_team"]]).value_counts()
    bad = counts[(counts < min_games) | (counts > max_games)]
    if len(bad):
        problems.append(f"games per team out of [{min_games}, {max_games}]: {bad.to_dict()}")

    # Relative to today, so a log cut off at the end is caught. A few unplayed past-dated games
    # are real postponements (they stay in the remaining schedule and get simulated).
    if today is not None:
        cur_played = cur["home_pts"].notna() | cur["away_pts"].notna()
        cutoff = pd.Timestamp(today - timedelta(days=stale_grace_days))
        stale = cur[~cur_played & (cur["date"] < cutoff)]
        if len(stale) > max_stale_games:
            ids = stale["game_id"].tolist()
            problems.append(
                f"{len(ids)} past-dated games have no result (allowing {max_stale_games} postponements): {ids[:5]}"
            )

    if problems:
        raise DataValidationError("; ".join(problems))


def validate_players(players: pd.DataFrame, games: pd.DataFrame) -> None:
    problems: list[str] = []
    dup = players.duplicated(["game_id", "player_id"])
    if dup.any():
        problems.append(
            f"duplicate (game_id, player_id) rows: {players.loc[dup, ['game_id', 'player_id']].values.tolist()[:5]}"
        )
    pairs = pd.concat(
        [
            games[["game_id", "home_team"]].set_axis(["game_id", "team"], axis=1),
            games[["game_id", "away_team"]].set_axis(["game_id", "team"], axis=1),
        ]
    )
    known = set(zip(pairs["game_id"], pairs["team"], strict=True))
    seen = pd.Series(
        list(zip(players["game_id"], players["team"], strict=True)), index=players.index
    )
    unknown = seen[~seen.isin(known)]
    if len(unknown):
        problems.append(
            f"player rows with (game_id, team) not in games: {unknown.drop_duplicates().tolist()[:5]}"
        )
    have = set(zip(players["game_id"], players["team"], strict=True))
    done = games[games["home_pts"].notna()]
    expected = set(zip(done["game_id"], done["home_team"], strict=True)) | set(
        zip(done["game_id"], done["away_team"], strict=True)
    )
    absent = sorted(expected - have)
    if absent:
        problems.append(f"no player rows for completed (game_id, team): {absent[:5]}")
    if (players["min"] < 0).any():
        problems.append("negative minutes in player log")
    if problems:
        raise DataValidationError("; ".join(problems))
