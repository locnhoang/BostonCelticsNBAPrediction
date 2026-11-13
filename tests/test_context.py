import pytest

from celtics.export.context import team_context
from celtics.teams import CONFERENCE_OF, TEAMS
from tests.factories import make_games

GAMES = make_games(
    [
        ("g1", 2026, "2026-10-22", "BOS", "NYK", 110, 100),
        ("g2", 2026, "2026-10-24", "NYK", "BOS", 105, 100),
        ("g3", 2026, "2026-10-26", "BOS", "MIA", 120, 100),
        ("g4", 2026, "2026-10-28", "BOS", "PHI", 99, 98),
        ("g5", 2026, "2026-10-30", "BOS", "ATL", None, None),  # unplayed: ignored
        ("old", 2025, "2026-03-01", "BOS", "NYK", 50, 150),  # other season: ignored
    ]
)


def test_record_form_and_ratings():
    ctx = team_context(GAMES, 2026)
    assert set(ctx) == set(TEAMS)
    bos = ctx["BOS"]
    assert (bos["wins"], bos["losses"]) == (3, 1)
    assert bos["last10"] == "3-1" and bos["streak"] == "W2"
    assert bos["off_rating"] == pytest.approx(107.2, abs=0.05)  # 429 / 400 poss
    assert bos["def_rating"] == pytest.approx(100.8, abs=0.05)  # 403 / 400 poss
    assert bos["net_rating"] == pytest.approx(6.5, abs=0.05)
    assert ctx["NYK"]["streak"] == "W1"
    assert ctx["ATL"] == {
        "wins": 0, "losses": 0, "conf_rank": 5, "games_back": 1.0, "last10": "0-0", "streak": "",
        "off_rating": 0.0, "def_rating": 0.0, "net_rating": 0.0,
        "off_rank": 5, "def_rank": 5, "net_rank": 5,
    }


def test_conference_rank_and_games_back():
    ctx = team_context(GAMES, 2026)
    assert ctx["BOS"]["conf_rank"] == 1 and ctx["BOS"]["games_back"] == 0.0
    assert ctx["NYK"]["conf_rank"] == 2 and ctx["NYK"]["games_back"] == 1.0
    assert ctx["MIA"]["conf_rank"] == 3 and ctx["MIA"]["games_back"] == 1.5  # 0-1 tie broken by abbreviation
    assert ctx["PHI"]["conf_rank"] == 4
    assert ctx["DAL"]["conf_rank"] == 1 and ctx["DEN"]["conf_rank"] == 2  # West: nobody played; abbreviation order


def test_league_ranks_direction():
    ctx = team_context(GAMES, 2026)
    # offense: BOS 107.25, NYK 102.5, MIA 100, PHI 98 -> higher is better
    assert [ctx[t]["off_rank"] for t in ("BOS", "NYK", "MIA", "PHI")] == [1, 2, 3, 4]
    # defense: PHI 99, BOS 100.75, NYK 105, MIA 120 -> lower is better
    assert [ctx[t]["def_rank"] for t in ("PHI", "BOS", "NYK", "MIA")] == [1, 2, 3, 4]
    # net: BOS +6.5, PHI -1, NYK -2.5, MIA -20
    assert [ctx[t]["net_rank"] for t in ("BOS", "PHI", "NYK", "MIA")] == [1, 2, 3, 4]


def test_last10_uses_latest_ten_games():
    rows = [
        (f"g{i}", 2026, f"2026-11-{i + 1:02d}", "BOS", "NYK", 100 if i < 2 else 110, 105)
        for i in range(12)
    ]  # two losses, then ten wins
    ctx = team_context(make_games(rows), 2026)
    assert ctx["BOS"]["last10"] == "10-0" and ctx["BOS"]["streak"] == "W10"


def test_games_back_can_be_negative_and_validates():
    from celtics.export.schema import TeamContext

    games = make_games(
        [
            ("a", 2026, "2026-10-22", "BOS", "DAL", 110, 100),  # BOS 1-0 (1.000) leads
            ("b", 2026, "2026-10-22", "NYK", "DAL", 110, 100),
            ("c", 2026, "2026-10-24", "NYK", "DEN", 110, 100),
            ("d", 2026, "2026-10-26", "NYK", "DAL", 110, 100),
            ("e", 2026, "2026-10-28", "NYK", "DEN", 100, 110),  # NYK 3-1 (.750)
        ]
    )
    ctx = team_context(games, 2026)
    assert ctx["BOS"]["conf_rank"] == 1 and ctx["NYK"]["conf_rank"] == 2
    assert ctx["NYK"]["games_back"] == -0.5
    TeamContext.model_validate(ctx["NYK"])


def test_opening_day_all_zero_and_alphabetical():
    games = make_games([("u1", 2026, "2026-10-22", "BOS", "NYK", None, None)])
    ctx = team_context(games, 2026)
    assert all(c["wins"] == 0 and c["losses"] == 0 and c["games_back"] == 0.0 for c in ctx.values())
    east = sorted((t for t in TEAMS if CONFERENCE_OF[t] == "East"), key=lambda t: ctx[t]["conf_rank"])
    assert east == sorted(east) and ctx["ATL"]["conf_rank"] == 1
