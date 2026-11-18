from datetime import date

import pandas as pd
import pytest

from celtics.data.cache import player_season_path, refresh_players
from celtics.data.fetch import FetchBlockedError, nba_team_ids
from celtics.data.parse import PLAYER_COLUMNS, playerlog_to_frame
from celtics.data.validate import DataValidationError, validate_players
from celtics.teams import TEAMS
from tests.factories import make_games, player_rows


def test_playerlog_to_frame_maps_columns_and_filters():
    raw = pd.DataFrame(
        [
            player_rows("0022600001", "2026-10-22", "BOS", 11, "A Player", 30.5, pts=20, fgm=8),
            player_rows("0062600001", "2026-12-15", "BOS", 11, "A Player", 35.0, pts=30),
        ]
    )
    out = playerlog_to_frame(raw, 2026)
    assert list(out.columns) == PLAYER_COLUMNS
    assert out["game_id"].nunique() == 1
    row = out.iloc[0]
    assert (row["team"], row["player_id"], row["min"], row["season"]) == ("BOS", 11, 30.5, 2026)
    assert (row["pts"], row["fgm"]) == (20.0, 8.0)
    assert row["date"] == pd.Timestamp("2026-10-22")


def _fake_players(season, n_games=900):
    rows = []
    for g in range(n_games):
        home, away = TEAMS[g % 30], TEAMS[(g + 1 + g // 30) % 30]
        if home == away:
            away = TEAMS[(g + 2) % 30]
        for t in (home, away):
            rows.append(
                player_rows(
                    f"002{season % 100:02d}{g:05d}", f"{season}-11-01", t, 100 + g, "P", 20.0
                )
            )
    return pd.DataFrame(rows)


def test_refresh_players_fetches_past_once_and_current_always(tmp_path, monkeypatch):
    monkeypatch.setattr("celtics.data.cache.FIRST_SEASON", 2024)
    fetched = []

    def fake(season):
        fetched.append(season)
        return _fake_players(season)

    out = refresh_players(tmp_path, date(2026, 11, 13), fetch=fake, pause=0)
    assert fetched == [2024, 2025, 2026]
    assert player_season_path(tmp_path, 2025).exists()
    assert set(out["season"]) == {2024, 2025, 2026}
    fetched.clear()
    refresh_players(tmp_path, date(2026, 11, 14), fetch=fake, pause=0)
    assert fetched == [2026]
    assert list(tmp_path.glob("*.tmp")) == []


def test_refresh_players_rejects_incomplete_past_season(tmp_path, monkeypatch):
    monkeypatch.setattr("celtics.data.cache.FIRST_SEASON", 2025)
    with pytest.raises(FetchBlockedError, match="incomplete"):
        refresh_players(tmp_path, date(2026, 11, 13), fetch=lambda s: _fake_players(s, 10), pause=0)
    assert not player_season_path(tmp_path, 2025).exists()


def test_validate_players():
    games = make_games([("0022600001", 2026, "2026-10-22", "BOS", "NYK", 100, 90)])

    def frame(rows):
        return playerlog_to_frame(pd.DataFrame(rows), 2026)

    good = [
        player_rows("0022600001", "2026-10-22", "BOS", 1, "A", 30.0),
        player_rows("0022600001", "2026-10-22", "NYK", 2, "B", 25.0),
    ]
    validate_players(frame(good), games)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_players(frame(good + [good[0]]), games)
    with pytest.raises(DataValidationError, match="not in games"):
        validate_players(
            frame([player_rows("0022600001", "2026-10-22", "MIA", 3, "C", 10.0)]), games
        )
    with pytest.raises(DataValidationError, match="not in games"):
        validate_players(
            frame([player_rows("0022600009", "2026-10-22", "BOS", 3, "C", 10.0)]), games
        )
    with pytest.raises(DataValidationError, match="minutes"):
        validate_players(
            frame([player_rows("0022600001", "2026-10-22", "BOS", 1, "A", -1.0)]), games
        )
    # a completed game with no player rows for one side looks like the rotation sat out
    with pytest.raises(DataValidationError, match="no player rows"):
        validate_players(frame([player_rows("0022600001", "2026-10-22", "BOS", 1, "A", 30.0)]), games)
    incomplete = make_games([("0022600001", 2026, "2026-10-22", "BOS", "NYK", None, None)])
    validate_players(frame([player_rows("0022600001", "2026-10-22", "BOS", 1, "A", 30.0)]), incomplete)


def test_every_team_has_an_nba_team_id():
    ids = nba_team_ids()
    assert set(TEAMS) <= set(ids)


def _raising(exc):
    def fake(season):
        raise exc

    return fake


@pytest.mark.parametrize("exc", [FetchBlockedError("403"), ConnectionError("timed out")])
def test_refresh_players_keeps_cached_current_season_on_fetch_failure(tmp_path, monkeypatch, exc, caplog):
    monkeypatch.setattr("celtics.data.cache.FIRST_SEASON", 2026)
    today = date(2026, 11, 13)
    refresh_players(tmp_path, today, fetch=_fake_players, pause=0)
    path = player_season_path(tmp_path, 2026)
    before = path.read_bytes()
    out = refresh_players(tmp_path, today, fetch=_raising(exc), pause=0)
    assert path.read_bytes() == before
    assert set(out["season"]) == {2026}
    assert "cached" in caplog.text


def test_refresh_players_without_cache_reraises_fetch_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("celtics.data.cache.FIRST_SEASON", 2026)
    with pytest.raises(FetchBlockedError):
        refresh_players(tmp_path, date(2026, 11, 13), fetch=_raising(FetchBlockedError("403")), pause=0)
