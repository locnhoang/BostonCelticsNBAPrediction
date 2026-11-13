"""Local parquet cache: one file per season."""

import os
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd

from celtics.config import FIRST_SEASON, current_season
from celtics.data.fetch import FetchBlockedError, fetch_season_gamelog, fetch_season_playerlog
from celtics.data.fetch import fetch_schedule as _fetch_schedule
from celtics.data.parse import (
    GAME_COLUMNS,
    PLAYER_COLUMNS,
    empty_games,
    empty_players,
    gamelog_to_games,
    merge_schedule_results,
    playerlog_to_frame,
    schedule_to_games,
)
from celtics.data.validate import DataValidationError

MIN_PAST_TEAMS = 30
MIN_PAST_GAMES = 900  # covers the shortened 2019-20 (1059) and 2020-21 (1080) seasons


def _parse(fn, *args):
    """Run a parser; inconsistent fetched data is a validation failure, not a crash."""
    try:
        return fn(*args)
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def season_path(raw_dir: Path, season: int) -> Path:
    return Path(raw_dir) / f"games_{season}.parquet"


def load_all(raw_dir: Path) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("games_*.parquet"))
    if not files:
        return empty_games()
    games = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    games["date"] = pd.to_datetime(games["date"])
    games["season"] = games["season"].astype("int64")
    return games.sort_values(["date", "game_id"]).reset_index(drop=True)[GAME_COLUMNS]


def refresh(
    raw_dir: Path,
    today: date,
    fetch_gamelog: Callable[[int], pd.DataFrame] = fetch_season_gamelog,
    fetch_schedule: Callable[[int], pd.DataFrame] = _fetch_schedule,
    pause: float = 1.5,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    season = current_season(today)
    for past in range(FIRST_SEASON, season):
        path = season_path(raw_dir, past)
        if not path.exists():
            games = _parse(gamelog_to_games, fetch_gamelog(past), past)
            teams = len(set(games["home_team"]) | set(games["away_team"]))
            if teams < MIN_PAST_TEAMS or len(games) < MIN_PAST_GAMES:
                raise FetchBlockedError(f"season {past} looks incomplete: {len(games)} games")
            _write_parquet_atomic(games, path)
            time.sleep(pause)
    results = _parse(gamelog_to_games, fetch_gamelog(season), season)
    try:
        schedule = schedule_to_games(fetch_schedule(season), season)
    except ValueError as exc:
        raise FetchBlockedError(f"schedule for season {season} not published yet: {exc}") from exc
    merged = _parse(merge_schedule_results, schedule, results)
    _write_parquet_atomic(merged, season_path(raw_dir, season))
    return load_all(raw_dir)


def player_season_path(raw_dir: Path, season: int) -> Path:
    return Path(raw_dir) / f"players_{season}.parquet"


def load_all_players(raw_dir: Path) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("players_*.parquet"))
    if not files:
        return empty_players()
    players = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    players["date"] = pd.to_datetime(players["date"])
    players["season"] = players["season"].astype("int64")
    return players.sort_values(["date", "game_id", "team", "player_id"]).reset_index(drop=True)[PLAYER_COLUMNS]


def refresh_players(
    raw_dir: Path,
    today: date,
    fetch: Callable[[int], pd.DataFrame] = fetch_season_playerlog,
    pause: float = 1.5,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    season = current_season(today)
    for past in range(FIRST_SEASON, season):
        path = player_season_path(raw_dir, past)
        if not path.exists():
            players = _parse(playerlog_to_frame, fetch(past), past)
            n_games, n_teams = players["game_id"].nunique(), players["team"].nunique()
            if n_teams < MIN_PAST_TEAMS or n_games < MIN_PAST_GAMES:
                raise FetchBlockedError(f"player log for season {past} looks incomplete: {n_games} games")
            _write_parquet_atomic(players, path)
            time.sleep(pause)
    current = _parse(playerlog_to_frame, fetch(season), season)
    _write_parquet_atomic(current, player_season_path(raw_dir, season))
    return load_all_players(raw_dir)
