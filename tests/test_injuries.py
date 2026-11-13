import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from celtics.config import DTD_MIN_CASES, DTD_PRIOR
from celtics.data.parse import empty_players
from celtics.injuries import (
    ESPN_TEAM,
    learn_p_dtd,
    match_players,
    normalize_name,
    parse_injuries,
    update_dtd_log,
)
from celtics.teams import TEAMS
from tests.factories import make_games, player_rows

FIXTURE = Path(__file__).parent / "fixtures" / "espn_injuries.json"


def _raw() -> dict:
    return json.loads(FIXTURE.read_text())


def _players(rows: list[tuple[str, int, str]]) -> pd.DataFrame:
    base = empty_players()
    df = pd.DataFrame(
        {
            "team": [r[0] for r in rows],
            "player_id": [r[1] for r in rows],
            "player_name": [r[2] for r in rows],
            "season": 2026,
        }
    )
    return pd.concat([base, df], ignore_index=True)


def test_espn_team_map_covers_all_teams():
    assert set(ESPN_TEAM.values()) == set(TEAMS)
    assert ESPN_TEAM["LA Clippers"] == "LAC"
    assert ESPN_TEAM["Los Angeles Lakers"] == "LAL"


def test_normalize_name():
    assert normalize_name("Luka Dončić") == "luka doncic"
    assert normalize_name("Dereck Lively II") == "dereck lively"
    assert normalize_name("Gary Payton Jr.") == "gary payton"


def test_parse_injuries_maps_status_and_dates():
    inj = parse_injuries(_raw())
    assert list(inj.columns) == ["team", "name", "status", "comment", "return_date", "reported_at"]
    by = inj.set_index("name")
    assert by.loc["Marcus Smart", "status"] == "Out"
    assert by.loc["Jaylen Brown", "status"] == "Day-To-Day"
    assert by.loc["Jayson Tatum", "return_date"] == date(2026, 12, 1)
    assert by.loc["Jaylen Brown", "return_date"] is None
    assert by.loc["Kawhi Leonard", "team"] == "LAC"


def test_match_players_in_team_and_unmatched():
    inj = parse_injuries(_raw())
    players = _players(
        [("BOS", 1, "Jaylen Brown"), ("BOS", 2, "Jayson Tatum"), ("DAL", 3, "Dereck Lively")]
    )
    matched, unmatched = match_players(inj, players, 2026)
    ids = dict(zip(matched["name"], matched["player_id"], strict=True))
    assert ids == {"Jaylen Brown": 1, "Jayson Tatum": 2, "Dereck Lively II": 3}
    assert "Unknown Player (BOS)" in unmatched
    assert "Marcus Smart (HOU)" in unmatched
    assert "Kawhi Leonard (LAC)" in unmatched


def test_match_players_league_wide_fallback_for_traded_player():
    inj = parse_injuries(_raw())
    players = _players([("TOR", 9, "Kawhi Leonard")])
    matched, unmatched = match_players(inj, players, 2026)
    assert matched.set_index("name").loc["Kawhi Leonard", "player_id"] == 9
    assert "Kawhi Leonard (LAC)" not in unmatched


def test_match_players_first_initial_and_ambiguous_league_wide():
    inj = pd.DataFrame(
        {
            "team": ["BOS", "BOS"],
            "name": ["Nic Claxton", "John Smith"],
            "status": ["Out", "Out"],
            "comment": ["", ""],
            "return_date": [None, None],
            "reported_at": [pd.NaT, pd.NaT],
        }
    )
    players = _players(
        [("BOS", 1, "Nicolas Claxton"), ("NYK", 2, "John Smith"), ("MIA", 3, "John Smith")]
    )
    matched, unmatched = match_players(inj, players, 2026)
    assert dict(zip(matched["name"], matched["player_id"], strict=True)) == {"Nic Claxton": 1}
    assert unmatched == ["John Smith (BOS)"]


def _dtd_inputs():
    from celtics.data.parse import playerlog_to_frame

    games = make_games([("0021400001", 2014, "2014-11-02", "BOS", "NYK", 100, 90)])
    raw = pd.DataFrame([player_rows("0021400001", "2014-11-02", "BOS", 1, "Tatum", 30)])
    inj = pd.DataFrame(
        [
            {"team": "BOS", "name": "Brown", "status": "Day-To-Day", "player_id": 2},
            {"team": "BOS", "name": "Tatum", "status": "Out", "player_id": 1},
        ]
    )
    return games, playerlog_to_frame(raw, 2014), inj


def test_update_dtd_log_appends_and_resolves():
    games, players, inj = _dtd_inputs()
    empty_games = games.iloc[0:0]
    log = update_dtd_log([], inj, empty_games, players.iloc[0:0], date(2014, 11, 1))
    assert log == [
        {"date": "2014-11-01", "team": "BOS", "player_id": 2, "name": "Brown", "played": None}
    ]
    log = update_dtd_log(log, inj.iloc[0:0], games, players, date(2014, 11, 3))
    assert log[0]["played"] is False  # Brown has no row in BOS's next game
    again = update_dtd_log(log, inj, games, players, date(2014, 11, 1))
    assert len(again) == 1 and again[0]["played"] is False


def test_update_dtd_log_player_who_appears_played():
    games, players, inj = _dtd_inputs()
    inj = inj.assign(player_id=[1, 1], status="Day-To-Day")
    log = update_dtd_log([], inj, games.iloc[0:0], players.iloc[0:0], date(2014, 11, 1))
    assert len(log) == 1
    assert (
        update_dtd_log(log, inj.iloc[0:0], games, players, date(2014, 11, 3))[0]["played"] is True
    )


def test_learn_p_dtd_threshold():
    def cases(n, missed):
        return [
            {"date": "d", "team": "BOS", "player_id": i, "name": "x", "played": i >= missed}
            for i in range(n)
        ]

    assert learn_p_dtd(cases(DTD_MIN_CASES - 1, 30)) == DTD_PRIOR
    assert learn_p_dtd(cases(DTD_MIN_CASES, 30)) == pytest.approx(0.30)
    unresolved = cases(DTD_MIN_CASES - 1, 30) + [
        {"date": "d", "team": "BOS", "player_id": 999, "name": "x", "played": None}
    ]
    assert learn_p_dtd(unresolved) == DTD_PRIOR
