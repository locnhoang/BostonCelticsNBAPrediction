from datetime import date

import pandas as pd
import pytest

from celtics.data.cache import load_all, refresh, season_path
from celtics.data.fetch import FetchBlockedError, with_retries
from tests.factories import gamelog_rows, schedule_payload


def test_with_retries_succeeds_after_failures():
    calls = {"n": 0}
    sleeps = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("nope")
        return "ok"

    assert with_retries(flaky, attempts=4, base_delay=1.0, sleep=sleeps.append) == "ok"
    assert sleeps == [1.0, 2.0]


def test_with_retries_raises_fetch_blocked():
    def always_fails():
        raise TimeoutError("blocked")

    with pytest.raises(FetchBlockedError, match="blocked"):
        with_retries(always_fails, attempts=2, base_delay=0.0, sleep=lambda s: None)


def _fake_gamelog(season):
    gid = f"002{season % 100:02d}00001"
    return pd.DataFrame(gamelog_rows(gid, f"{season}-11-01", "BOS", "NYK", 110, 100))


def test_refresh_fetches_past_seasons_once_and_current_always(tmp_path, monkeypatch):
    monkeypatch.setattr("celtics.data.cache.FIRST_SEASON", 2024)
    monkeypatch.setattr("celtics.data.cache.MIN_PAST_GAMES", 1)
    monkeypatch.setattr("celtics.data.cache.MIN_PAST_TEAMS", 2)
    fetched = []

    def fake_gamelog(season):
        fetched.append(season)
        return _fake_gamelog(season)

    def fake_schedule(season):
        return schedule_payload(
            "2026-27",
            [("0022600001", "2026-11-01", "BOS", "NYK"), ("0022600002", "2026-11-20", "MIA", "BOS")],
        )

    kwargs = dict(fetch_gamelog=fake_gamelog, fetch_schedule=fake_schedule, pause=0)
    games = refresh(tmp_path, date(2026, 11, 12), **kwargs)
    assert fetched == [2024, 2025, 2026]
    assert season_path(tmp_path, 2025).exists()
    assert games["season"].tolist() == [2024, 2025, 2026, 2026]
    assert games["home_pts"].isna().sum() == 1  # the unplayed Nov 20 game

    fetched.clear()
    refresh(tmp_path, date(2026, 11, 13), **kwargs)
    assert fetched == [2026]


def test_load_all_empty_dir(tmp_path):
    assert load_all(tmp_path).empty


def test_refresh_refuses_incomplete_past_season(tmp_path, monkeypatch):
    monkeypatch.setattr("celtics.data.cache.FIRST_SEASON", 2025)
    with pytest.raises(FetchBlockedError, match="season 2025 looks incomplete: 1 games"):
        refresh(
            tmp_path,
            date(2026, 11, 12),
            fetch_gamelog=_fake_gamelog,
            fetch_schedule=lambda s: schedule_payload("2026-27", []),
            pause=0,
        )
    assert not season_path(tmp_path, 2025).exists()
    assert list(tmp_path.glob("*.tmp")) == []
