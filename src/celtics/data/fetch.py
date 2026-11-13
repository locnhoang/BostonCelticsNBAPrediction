"""Network access to NBA data. Everything here is retried and fails loudly."""

import time
from collections.abc import Callable

import pandas as pd

from celtics.config import season_label


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


def fetch_season_playerlog(season: int) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    def call() -> pd.DataFrame:
        endpoint = leaguegamelog.LeagueGameLog(
            season=season_label(season),
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="P",
            timeout=60,
        )
        return endpoint.get_data_frames()[0]

    return with_retries(call)


def fetch_schedule(season: int) -> pd.DataFrame:
    from nba_api.stats.endpoints import scheduleleaguev2

    def call() -> pd.DataFrame:
        endpoint = scheduleleaguev2.ScheduleLeagueV2(season=season_label(season), timeout=30)
        return endpoint.get_data_frames()[0]

    return with_retries(call)


def fetch_team_roster(team: str, season: int) -> pd.DataFrame:
    """Players on a team's roster (columns player_id, player_name)."""
    from nba_api.stats.endpoints import commonteamroster
    from nba_api.stats.static import teams

    team_ids = {t["abbreviation"]: t["id"] for t in teams.get_teams()}

    def call() -> pd.DataFrame:
        endpoint = commonteamroster.CommonTeamRoster(
            team_id=team_ids[team], season=season_label(season), timeout=30
        )
        return endpoint.get_data_frames()[0]

    raw = with_retries(call)
    return pd.DataFrame(
        {"player_id": raw["PLAYER_ID"].astype("int64"), "player_name": raw["PLAYER"].astype(str)}
    )


ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


def fetch_espn_injuries() -> dict:
    import requests

    def call() -> dict:
        resp = requests.get(ESPN_INJURIES_URL, timeout=30)
        resp.raise_for_status()
        return resp.json()

    return with_retries(call)
