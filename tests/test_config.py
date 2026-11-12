from datetime import date

from celtics.config import Paths, current_season, season_label
from celtics.teams import (
    CONFERENCE_MEMBERS,
    CONFERENCE_OF,
    DIVISION_MEMBERS,
    IS_EAST,
    TEAM_INDEX,
    TEAMS,
)


def test_season_label():
    assert season_label(2026) == "2026-27"
    assert season_label(2099) == "2099-00"


def test_current_season_rolls_over_in_august():
    assert current_season(date(2026, 11, 12)) == 2026
    assert current_season(date(2027, 3, 1)) == 2026
    assert current_season(date(2027, 8, 1)) == 2027


def test_thirty_teams_split_evenly():
    assert len(TEAMS) == 30 and TEAMS == sorted(TEAMS)
    assert TEAM_INDEX["BOS"] == TEAMS.index("BOS")
    assert sum(1 for t in TEAMS if CONFERENCE_OF[t] == "East") == 15
    assert IS_EAST.sum() == 15 and IS_EAST[TEAM_INDEX["BOS"]]
    assert [len(m) for m in CONFERENCE_MEMBERS] == [15, 15]
    assert sorted(len(m) for m in DIVISION_MEMBERS) == [5] * 6


def test_paths(tmp_path):
    p = Paths(tmp_path)
    assert p.raw_dir == tmp_path / "data" / "raw"
    assert p.model_dir == tmp_path / "data" / "models"
    assert p.site_data == tmp_path / "site" / "data"
