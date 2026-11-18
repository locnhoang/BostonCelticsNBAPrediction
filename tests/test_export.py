from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from celtics.config import FIRST_TRAIN_SEASON
from celtics.export.build import (
    game_rows,
    history_entry,
    read_json,
    seed_probabilities,
    team_summary,
    upsert_history,
    write_json_atomic,
)
from celtics.export.league import update_league_pregame_log
from celtics.export.schema import HistoryEntry, PregameLog, TeamSummary
from celtics.features.build import FEATURES, build_features
from celtics.models.margin import make_model
from celtics.simulate.season import SimResult
from celtics.teams import TEAMS
from tests.synth import make_league


def fake_result(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    seeds = np.tile(np.arange(1, 16), (n, 2))
    wins = rng.integers(30, 60, size=(n, 30))
    made = seeds <= 8
    return SimResult(teams=list(TEAMS), wins=wins, seed=seeds, made_playoffs=made)


def test_team_summary_is_consistent():
    s = team_summary(fake_result(), "BOS", 9, 4)
    TeamSummary.model_validate(s)
    assert s["p_top6"] + s["p_playin"] + s["p_miss"] == pytest.approx(1, abs=1e-3)
    assert sum(s["win_distribution"].values()) == pytest.approx(1, abs=1e-3)
    assert s["proj_wins_p10"] <= s["proj_wins_median"] <= s["proj_wins_p90"]


def test_seed_probabilities_sum_to_one():
    sp = seed_probabilities(fake_result(), "BOS")
    assert list(sp) == [str(k) for k in range(1, 16)]
    assert sum(sp.values()) == pytest.approx(1, abs=1e-3)


def test_schema_rejects_bad_probability():
    with pytest.raises(ValidationError):
        HistoryEntry.model_validate(
            {"date": "2026-11-12", "record": "9-4", "projected_wins_median": 50, "p10": 45,
             "p90": 55, "playoff_prob": 1.5, "top6_prob": 0.5, "seed1_prob": 0.1}
        )


@pytest.fixture(scope="module")
def season_rows():
    feats = build_features(make_league([2022, 2023], current=2023))
    train = feats[feats["completed"] & (feats["season"] >= FIRST_TRAIN_SEASON)]
    model = make_model("ridge").fit(train[FEATURES], train["margin"])
    return feats, model


def test_game_rows_perspective_and_pregame_log(season_rows):
    feats, model = season_rows
    rows = game_rows(feats, model, 2023, "BOS", pregame_log={})
    assert len(rows) == 82
    upcoming = [r for r in rows if not r["completed"]]
    done = [r for r in rows if r["completed"]]
    assert len(done) == 15 and all(r["result"][0] in "WL" for r in done)
    assert all(r["pregame_win_prob"] is None for r in done)  # nothing logged yet
    away = next(r for r in upcoming if not r["home"])
    assert (away["win_prob"] > 0.5) == (away["pred_margin"] > 0)

    log = update_league_pregame_log({}, feats, model, 2023, "2023-11-19T14:00:00", "2023-11-19")
    assert {r["game_id"] for r in upcoming} <= set(log)  # the league log covers every team's games
    first = upcoming[0]["game_id"]
    log[first]["home_win_prob"] = 0.42
    # once the game is completed the logged value is kept and shown
    feats2 = feats.copy()
    i = feats2.index[feats2["game_id"] == first][0]
    feats2.loc[i, ["home_pts", "away_pts", "completed", "margin"]] = [100.0, 90.0, True, 10.0]
    rows2 = game_rows(feats2, model, 2023, "BOS", pregame_log=log)
    shown = next(r for r in rows2 if r["game_id"] == first)
    assert shown["completed"]
    assert shown["pregame_win_prob"] == (0.42 if shown["home"] else 0.58)
    log2 = update_league_pregame_log(log, feats2, model, 2023, "2023-11-21T14:00:00", "2023-11-21")
    assert log2[first]["home_win_prob"] == 0.42  # completed: not refreshed


def test_history_upsert_replaces_same_day():
    s = team_summary(fake_result(), "BOS", 9, 4)
    s["p_playoffs"] = 0.8
    e1 = history_entry(date(2026, 11, 12), s)
    HistoryEntry.model_validate(e1)
    hist = upsert_history([], e1)
    hist = upsert_history(hist, {**e1, "playoff_prob": 0.9})
    assert len(hist) == 1 and hist[0]["playoff_prob"] == 0.9
    hist = upsert_history(hist, {**e1, "date": "2026-11-11"})
    assert [h["date"] for h in hist] == ["2026-11-11", "2026-11-12"]


def test_atomic_write_roundtrip(tmp_path):
    path = tmp_path / "nested" / "x.json"
    write_json_atomic(path, {"a": 1})
    assert read_json(path, None) == {"a": 1}
    assert read_json(tmp_path / "missing.json", []) == []


def _toy_feats(rows):
    return pd.DataFrame(
        [{"game_id": g, "season": 2026, "date": pd.Timestamp(d), "completed": False} for g, d in rows]
    )


class _FixedModel:
    def win_prob(self, df):
        return np.full(len(df), 0.7)

    def predict_margin(self, df):
        return np.full(len(df), 4.0)


def test_pregame_log_tracks_latest_prediction_until_game_day_passes():
    old = {
        "future": {"home_win_prob": 0.55, "pred_margin": 1.0, "logged_at": "2026-11-01T14:00:00"},
        "past": {"home_win_prob": 0.55, "pred_margin": 1.0, "logged_at": "2026-11-10T14:00:00"},
    }
    feats = _toy_feats([("future", "2026-11-15"), ("past", "2026-11-11"), ("today", "2026-11-12")])
    out = update_league_pregame_log(old, feats, _FixedModel(), 2026, "2026-11-12T14:00:00", "2026-11-12")
    assert out["future"]["home_win_prob"] == 0.7 and out["future"]["logged_at"] == "2026-11-12T14:00:00"
    assert out["past"]["home_win_prob"] == 0.55  # frozen
    assert out["today"]["home_win_prob"] == 0.7  # game today: still refreshable
    # a past-dated game with no entry yet is not back-filled
    assert "past" not in update_league_pregame_log({}, feats, _FixedModel(), 2026, "x", "2026-11-12")


def test_pregame_log_schema_rejects_bad_entry():
    PregameLog.validate_python({"g1": {"home_win_prob": 0.5, "pred_margin": None, "logged_at": "x"}})
    with pytest.raises(ValidationError):
        PregameLog.validate_python({"g1": {"home_win_prob": 1.5, "pred_margin": 1.0, "logged_at": "x"}})


def test_update_league_pregame_log_freeze_uses_given_today_not_utc_clock():
    feats = _toy_feats([("g", "2026-11-11")])
    # UTC clock already says Nov 12 but the Eastern game day is still Nov 11: not frozen
    out = update_league_pregame_log({}, feats, _FixedModel(), 2026, "2026-11-12T02:00:00", "2026-11-11")
    assert out["g"]["home_win_prob"] == 0.7
    assert update_league_pregame_log({}, feats, _FixedModel(), 2026, "2026-11-12T02:00:00", "2026-11-12") == {}


def test_game_rows_drivers_sum_and_flip(season_rows):
    feats, model = season_rows
    rows = game_rows(feats, model, 2023, "BOS", pregame_log={})
    upcoming = [r for r in rows if not r["completed"]]
    for r in upcoming:
        assert [d["label"] for d in r["drivers"]] == [
            "Season strength", "Recent form", "Home court", "Rest & schedule"
        ]
        assert sum(d["value"] for d in r["drivers"]) == pytest.approx(r["pred_margin"], abs=0.25 + 1e-9)
    home = next(r for r in upcoming if r["home"])
    away = next(r for r in upcoming if not r["home"])
    court = {r["game_id"]: next(d["value"] for d in r["drivers"] if d["label"] == "Home court") for r in (home, away)}
    assert court[home["game_id"]] > 0 > court[away["game_id"]]
    assert all(r["drivers"] is None for r in rows if r["completed"])


def test_game_rows_no_drivers_for_nonlinear_model(season_rows):
    feats, _ = season_rows
    train = feats[feats["completed"] & (feats["season"] >= FIRST_TRAIN_SEASON)]
    gbm = make_model("gbm").fit(train[FEATURES], train["margin"])
    rows = game_rows(feats, gbm, 2023, "BOS", pregame_log={})
    assert all(r["drivers"] is None for r in rows)


def test_schema_rejects_bad_team_context():
    from celtics.export.schema import TeamContext

    good = {
        "wins": 3, "losses": 1, "conf_rank": 1, "games_back": 0.0, "last10": "3-1", "streak": "W2",
        "off_rating": 107.2, "def_rating": 100.8, "net_rating": 6.5,
        "off_rank": 1, "def_rank": 2, "net_rank": 1,
    }
    TeamContext.model_validate(good)
    for bad in ({"conf_rank": 16}, {"net_rank": 0}, {"streak": "X3"}, {"last10": "ten"}):
        with pytest.raises(ValidationError):
            TeamContext.model_validate({**good, **bad})
