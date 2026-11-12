"""Network access to NBA data. Everything here is retried and fails loudly."""

import time
from collections.abc import Callable

import pandas as pd
import requests

from celtics.config import season_label

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"


class FetchBlockedError(RuntimeError):
    """NBA data sources refused, rate-limited, or timed out after all retries."""


def with_retries[T](
    fn: Callable[[], T],
    attempts: int = 4,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # network stacks raise many exception types
            last = exc
            if i < attempts - 1:
                sleep(base_delay * 2**i)
    raise FetchBlockedError(f"fetch blocked or failing after {attempts} attempts: {last}") from last


def fetch_season_gamelog(season: int) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    def call() -> pd.DataFrame:
        endpoint = leaguegamelog.LeagueGameLog(
            season=season_label(season),
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="T",
            timeout=30,
        )
        return endpoint.get_data_frames()[0]

    return with_retries(call)


def fetch_schedule() -> dict:
    def call() -> dict:
        resp = requests.get(SCHEDULE_URL, timeout=30)
        resp.raise_for_status()
        return resp.json()

    return with_retries(call)
