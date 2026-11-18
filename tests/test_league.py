from datetime import timedelta

import pandas as pd
import pytest

from celtics.config import FIRST_TRAIN_SEASON
from celtics.export.build import injury_teams
from celtics.export.context import team_context
from celtics.export.league import (
    migrate_pregame_log,
    pick,
    pick_record,
    slate_days,
    update_league_pregame_log,
)
from celtics.export.schema import Slate
from celtics.features.build import FEATURES, build_features
from celtics.models.margin import make_model
from tests.synth import make_league

GAMES = make_league([2022, 2023], current=2023)
CONTEXT_30 = team_context(GAMES, 2023)


@pytest.fixture(scope="module")
def season_feats_model():
    feats = build_features(GAMES)
    train = feats[feats["completed"] & (feats["season"] >= FIRST_TRAIN_SEASON)]
    model = make_model("ridge").fit(train[FEATURES], train["margin"])
    return feats, model


def test_migrate_converts_boston_perspective_to_home():
    games = pd.DataFrame({"game_id": ["g1", "g2"], "home_team": ["BOS", "DEN"], "away_team": ["NYK", "BOS"]})
    old = {
        "g1": {"win_prob": 0.7, "pred_margin": 5.0, "logged_at": "t"},
        "g2": {"win_prob": 0.6, "pred_margin": 3.0, "logged_at": "t"},
        "gx": {"win_prob": 0.5, "pred_margin": 0.0, "logged_at": "t"},  # unknown game: dropped
    }
    out = migrate_pregame_log(old, games)
    assert out["g1"] == {"home_win_prob": 0.7, "pred_margin": 5.0, "logged_at": "t"}
    assert out["g2"] == {"home_win_prob": 0.4, "pred_margin": -3.0, "logged_at": "t"}
    assert "gx" not in out
    assert migrate_pregame_log(out, games) == out  # idempotent on new format


def test_league_log_covers_every_upcoming_game_and_freezes_past(season_feats_model):
    feats, model = season_feats_model  # synthetic league from tests.synth, ridge model
    log = update_league_pregame_log({}, feats, model, 2023, "2023-11-19T14:00:00", "2023-11-19")
    upcoming = feats[(feats["season"] == 2023) & ~feats["completed"] & (feats["date"] >= "2023-11-19")]
    assert set(log) == set(upcoming["game_id"])
    frozen = {"old": {"home_win_prob": 0.3, "pred_margin": -4.0, "logged_at": "x"}}
    assert update_league_pregame_log(frozen, feats, model, 2023, "2023-11-19T14:00:00", "2023-11-19")["old"] == frozen["old"]


def test_pick_ties_go_home():
    assert pick("BOS", "DEN", 0.5) == "BOS"
    assert pick("BOS", "DEN", 0.4999) == "DEN"


def _g(gid, day, home, away, p, hp, ap):
    return {"game_id": gid, "date": day, "home": home, "away": away, "completed": True,
            "pregame_home_win_prob": p, "home_pts": hp, "away_pts": ap}


def test_pick_record_since_and_per_team():
    games = [
        _g("a", "2026-11-17", "BOS", "DEN", 0.6, 100, 90),   # before since: ignored
        _g("b", "2026-11-18", "BOS", "DEN", 0.6, 100, 90),   # correct
        _g("c", "2026-11-18", "NYK", "MIA", 0.7, 90, 100),   # wrong
        _g("d", "2026-11-19", "LAL", "BOS", 0.5, 80, 90),    # pick LAL (home), BOS won: wrong
        {**_g("e", "2026-11-19", "UTA", "SAC", None, 1, 0)},  # no logged pick: ignored
    ]
    assert pick_record(games, "2026-11-18") == {"since": "2026-11-18", "correct": 1, "wrong": 2}
    assert pick_record(games, "2026-11-18", team="BOS") == {"since": "2026-11-18", "correct": 1, "wrong": 1}


def _today(feats):
    cur = feats[feats["season"] == 2023]
    dates = sorted({d.date() for d in cur["date"]})
    return next(d for d in dates if d + timedelta(days=1) not in dates)  # tomorrow has no games


def test_slate_has_three_days_and_empty_day(season_feats_model):
    feats, model = season_feats_model
    today = _today(feats)
    days = slate_days(feats, model, {}, CONTEXT_30, today, {}, None)
    assert [d["date"] for d in days] == [(today + timedelta(days=k)).isoformat() for k in (-1, 0, 1)]
    assert days[2]["games"] == []
    Slate.model_validate({"generated_at": "x", "model_version": "m", "days": days,
                          "record": {"since": "2026-11-18", "yesterday": {"correct": 0, "wrong": 0},
                                     "total": {"correct": 0, "wrong": 0}}})


def test_slate_game_fields(season_feats_model):
    feats, model = season_feats_model
    cur = feats[feats["season"] == 2023]
    first_open = cur[~cur["completed"]]["date"].min().date()
    up = slate_days(feats, model, {}, CONTEXT_30, first_open, {}, None)[1]["games"][0]
    assert not up["completed"]
    assert up["home_win_prob"] is not None
    assert [d["label"] for d in up["drivers"]] == [
        "Season strength", "Recent form", "Home court", "Rest & schedule"
    ]
    assert up["pick"] == pick(up["home"], up["away"], up["home_win_prob"])
    assert up["injuries"] == {"home": [], "away": []}
    assert up["home_win_prob_healthy"] == up["home_win_prob"]
    home_ctx = CONTEXT_30[up["home"]]
    assert up["home_record"] == f"{home_ctx['wins']}-{home_ctx['losses']}"

    played = cur[cur["completed"]]["date"].max().date()  # slate for the day after: it is "yesterday"
    today = played + timedelta(days=1)
    done = slate_days(feats, model, {}, CONTEXT_30, today, {}, None)[0]["games"][0]
    assert done["pregame_home_win_prob"] is None and done["correct"] is None
    log = {done["game_id"]: {"home_win_prob": 0.7, "pred_margin": 4.0, "logged_at": "t"}}
    done = slate_days(feats, model, log, CONTEXT_30, today, {}, None)[0]["games"][0]
    assert done["completed"]
    assert done["home_win_prob"] is None and done["drivers"] is None
    assert done["pregame_home_win_prob"] == 0.7
    assert done["correct"] == (done["home_pts"] > done["away_pts"])


def test_slate_injuries_callable_maps_by_team_and_not_next(season_feats_model):
    feats, model = season_feats_model
    cur = feats[feats["season"] == 2023]
    first_open = cur[~cur["completed"]]["date"].min().date()
    seen = []

    def injuries(home, away, game_date, not_next):
        seen.append((home, away, game_date, not_next))
        return {home: [{"name": "H"}], away: [{"name": "A"}]}

    game = slate_days(feats, model, {}, CONTEXT_30, first_open, {}, injuries)[1]["games"][0]
    assert game["injuries"] == {"home": [{"name": "H"}], "away": [{"name": "A"}]}
    # the season's first open day: every team's next game is on it, so nobody is "not next"
    assert seen[0][3] == frozenset()


def test_injury_teams_only_injured_and_not_next():
    parsed = pd.DataFrame({
        "team": ["AAA", "AAA"], "name": ["Out Guy", "Dtd Guy"], "status": ["Out", "Day-To-Day"],
        "comment": ["", ""], "return_date": [pd.NaT, pd.NaT],
    })
    matched = pd.DataFrame({"team": ["AAA", "AAA"], "name": ["Out Guy", "Dtd Guy"], "player_id": [1, 2]})
    current = pd.DataFrame({
        "team": ["AAA"] * 3, "player_id": [1, 2, 3], "player_name": ["Out Guy", "Dtd Guy", "Fit Guy"],
        "expected_minutes": [10.0, 30.0, 20.0], "game_value": [1.0, 2.0, 3.0], "rotation": [True] * 3,
    })
    day = pd.Timestamp("2024-11-20").date()

    def rows(**kw):
        return injury_teams(["AAA"], parsed, matched, current, {"AAA": None}, day, day, 0.4, **kw)["AAA"]

    got = rows()
    assert [r["status"] for r in got] == ["Out", "Day-To-Day"]
    assert got[1]["p_out"] == 0.4
    assert rows(not_next=frozenset({"AAA"}))[1]["p_out"] == 0.0
