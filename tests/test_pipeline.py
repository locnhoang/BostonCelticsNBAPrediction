import json
from datetime import date, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

import celtics.pipeline as pipeline_module
from celtics.cli import main
from celtics.config import DEFAULT_TEAM_SD, Paths
from celtics.data.cache import player_season_path, season_path
from celtics.data.fetch import FetchBlockedError
from celtics.data.validate import DataValidationError
from celtics.export.schema import BacktestReport, DtdLog, HistoryEntry, Predictions
from celtics.injuries import ESPN_TEAM
from celtics.pipeline import read_team_sd, run_backtest, run_train, run_update
from celtics.teams import TEAM_INDEX
from tests.synth import make_league

TODAY = date(2024, 11, 20)


FULL_NAME = {abbr: name for name, abbr in reversed(list(ESPN_TEAM.items()))}


def _pname(team, k):
    return f"Player {team} {k}"


def _player_logs(games):
    rows = []
    for g in games[games["home_pts"].notna()].itertuples():
        for team in (g.home_team, g.away_team):
            for k in range(8):
                if k == 7 and int(g.game_id[-5:]) % 4 == 0:
                    continue  # the best player sits out some games, so historical avail varies
                rows.append(
                    {
                        "game_id": g.game_id, "season": g.season, "date": g.date, "team": team,
                        "player_id": TEAM_INDEX[team] * 100 + k, "player_name": _pname(team, k),
                        "min": 30.0, "pts": 10.0 + 2 * k, "fgm": 4.0, "fga": 9.0, "ftm": 2.0, "fta": 2.0,
                        "oreb": 1.0, "dreb": 3.0, "ast": 2.0, "stl": 1.0, "blk": 0.0, "tov": 1.0, "pf": 2.0,
                    }
                )  # fmt: skip
    return pd.DataFrame(rows)


def _next_opponent(repo):
    cur = pd.read_parquet(season_path(repo.raw_dir, 2024))
    mine = cur[((cur["home_team"] == "BOS") | (cur["away_team"] == "BOS")) & cur["home_pts"].isna()]
    g = mine.sort_values(["date", "game_id"]).iloc[0]
    return g["away_team"] if g["home_team"] == "BOS" else g["home_team"]


def _feed(team, k=0, status="Out"):
    entry = {
        "status": status,
        "date": "2024-11-19T10:00Z",
        "shortComment": "synthetic",
        "athlete": {"displayName": _pname(team, k)},
        "details": {},
    }
    return {
        "timestamp": "2024-11-20T12:00:00Z",
        "injuries": [{"displayName": FULL_NAME[team], "injuries": [entry]}],
    }


def _patch_feeds(repo, monkeypatch):
    monkeypatch.setattr("celtics.pipeline.fetch_espn_injuries", lambda: _feed(_next_opponent(repo)))
    monkeypatch.setattr(
        "celtics.pipeline.fetch_team_roster",
        lambda team, season: pd.DataFrame(
            {"player_id": [TEAM_INDEX[team] * 100 + k for k in range(8)],
             "player_name": [_pname(team, k) for k in range(8)]}
        ),
    )  # fmt: skip


def _write_league(tmp_path, seasons, with_players):
    games = make_league(seasons, current=2024, completed_rounds=15)
    raw = Paths(tmp_path).raw_dir
    raw.mkdir(parents=True)
    for season, g in games.groupby("season"):
        g.to_parquet(season_path(raw, season), index=False)
    if with_players:
        for season, g in _player_logs(games).groupby("season"):
            g.to_parquet(player_season_path(raw, season), index=False)
    return Paths(tmp_path)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Four synthetic seasons with no player logs (availability 0); ESPN and rosters patched."""
    repo = _write_league(tmp_path, [2021, 2022, 2023, 2024], with_players=False)
    _patch_feeds(repo, monkeypatch)
    return repo


@pytest.fixture
def repo_players(tmp_path, monkeypatch):
    """Two synthetic seasons with player logs (the availability code is slow on more)."""
    repo = _write_league(tmp_path, [2023, 2024], with_players=True)
    _patch_feeds(repo, monkeypatch)
    return repo


@pytest.fixture
def opp(repo):
    return _next_opponent(repo)


@pytest.fixture
def opp_p(repo_players):
    return _next_opponent(repo_players)


@pytest.fixture(autouse=True)
def _pin_ridge(monkeypatch):
    # these tests exercise the baseline path; tests that need ridge_avail call _force_avail_model
    monkeypatch.setattr("celtics.pipeline.MODEL_NAME", "ridge")


def test_run_update_writes_valid_outputs(repo):
    run_update(repo, TODAY, offline=True, n_sims=200, now=datetime(2024, 11, 19, 14, 0))  # day of the next synthetic round
    pred = json.loads((repo.site_data / "predictions.json").read_text())
    Predictions.model_validate(pred)
    assert len(pred["team_context"]) == 30
    upcoming = [g for g in pred["games"] if not g["completed"]]
    assert upcoming and all(len(g["drivers"]) == 4 for g in upcoming)
    assert pred["season"] == "2024-25"
    assert len(pred["games"]) == 82
    assert pred["team_summary"]["wins"] + pred["team_summary"]["losses"] == 15
    history = json.loads((repo.site_data / "history.json").read_text())
    assert len(history) == 1
    HistoryEntry.model_validate(history[0])
    log = json.loads((repo.site_data / "pregame_log.json").read_text())
    # the Nov 19 round is dated before today (Nov 20) and unplayed: frozen, never back-filled
    assert len(log) == 66
    assert (repo.model_dir / "ridge.joblib").exists()


def test_run_update_twice_same_day_keeps_one_history_entry(repo):
    run_update(repo, TODAY, offline=True, n_sims=100)
    run_update(repo, TODAY, offline=True, n_sims=100)
    assert len(json.loads((repo.site_data / "history.json").read_text())) == 1


def test_run_backtest_writes_report(repo):
    report = run_backtest(repo, TODAY, offline=True, test_seasons=[2022, 2023], n_sims=100)
    BacktestReport.model_validate(report)
    assert set(report["models"]) == {"homecourt", "elo", "ridge", "ridge_avail", "gbm"}
    assert report["chosen_team_sd"] in [0.0, 1.0, 2.0, 2.5, 3.0, 4.0]
    assert (repo.site_data / "backtest.json").exists()
    assert report["availability_note"].startswith("Backtest uses actual absences")


def test_cli_update_offline(repo, monkeypatch):
    monkeypatch.chdir(repo.root)
    assert main(["--date", "2024-11-20", "update", "--offline", "--sims", "100"]) == 0


def test_cli_reports_blocked_fetch(repo, monkeypatch, capsys):
    monkeypatch.chdir(repo.root)

    def blocked(*args, **kwargs):
        raise FetchBlockedError("403")

    monkeypatch.setattr("celtics.pipeline.refresh", blocked)
    assert main(["--date", "2024-11-20", "update"]) == 2
    assert "fetch blocked" in capsys.readouterr().err
    for name in ("predictions.json", "history.json", "pregame_log.json"):
        assert not (repo.site_data / name).exists()


def _corrupt(repo):
    path = season_path(repo.raw_dir, 2023)
    df = pd.read_parquet(path)
    pd.concat([df, df.iloc[[0]]], ignore_index=True).to_parquet(path, index=False)


def _meta(repo):
    return json.loads((repo.model_dir / "ridge.json").read_text())["trained_on"]


def test_refit_cadence(repo, monkeypatch):
    # The synthetic league does not advance with the later dates below, so skip the stale check.
    monkeypatch.setattr("celtics.pipeline.validate_games", lambda *a, **k: None)
    run_update(repo, TODAY, offline=True, n_sims=50)
    assert _meta(repo) == "2024-11-20"
    pred = run_update(repo, date(2024, 11, 23), offline=True, n_sims=50)
    assert _meta(repo) == "2024-11-20"
    assert pred["model_version"].endswith("2024-11-20")
    pred = run_update(repo, date(2024, 11, 27), offline=True, n_sims=50)
    assert _meta(repo) == "2024-11-27"
    assert pred["model_version"].endswith("2024-11-27")
    run_update(repo, date(2024, 11, 28), offline=True, n_sims=50, force_train=True)
    assert _meta(repo) == "2024-11-28"


def test_cli_validation_failure_exits_3_and_writes_nothing(repo, monkeypatch, capsys):
    _corrupt(repo)
    monkeypatch.chdir(repo.root)
    assert main(["--date", "2024-11-20", "update", "--offline"]) == 3
    assert "data validation failed" in capsys.readouterr().err
    assert not (repo.site_data / "predictions.json").exists()
    assert not (repo.site_data / "history.json").exists()
    assert not (repo.site_data / "pregame_log.json").exists()


def test_run_backtest_rejects_invalid_data(repo):
    _corrupt(repo)
    with pytest.raises(DataValidationError):
        run_backtest(repo, TODAY, offline=True, test_seasons=[2022, 2023], n_sims=50)
    assert not (repo.site_data / "backtest.json").exists()


def test_run_train_rejects_invalid_data(repo):
    _corrupt(repo)
    with pytest.raises(DataValidationError):
        run_train(repo, TODAY)
    assert not (repo.model_dir / "ridge.joblib").exists()


def test_read_team_sd(repo):
    assert read_team_sd(repo) == DEFAULT_TEAM_SD
    repo.site_data.mkdir(parents=True)
    (repo.site_data / "backtest.json").write_text(json.dumps({"chosen_team_sd": 3.0}))
    assert read_team_sd(repo) == 3.0


def test_run_update_rejects_completed_games_going_backwards(repo):
    run_update(repo, TODAY, offline=True, n_sims=50)
    path = season_path(repo.raw_dir, 2024)
    df = pd.read_parquet(path)
    mine = df[(df["home_team"] == "BOS") | (df["away_team"] == "BOS")]
    last_id = mine.loc[mine["home_pts"].notna(), "game_id"].iloc[-1]
    df.loc[df["game_id"] == last_id, ["home_pts", "away_pts", "poss"]] = float("nan")  # a result vanishes
    df.to_parquet(path, index=False)
    before = (repo.site_data / "predictions.json").read_text()
    with pytest.raises(DataValidationError, match="completed games went backwards"):
        run_update(repo, TODAY, offline=True, n_sims=50)
    assert (repo.site_data / "predictions.json").read_text() == before


def test_run_update_rejects_corrupt_pregame_log(repo):
    repo.site_data.mkdir(parents=True)
    bad = {"g1": {"win_prob": 3.0, "pred_margin": 1.0, "logged_at": "2024-11-19T14:00:00"}}
    (repo.site_data / "pregame_log.json").write_text(json.dumps(bad))
    with pytest.raises(ValidationError):
        run_update(repo, TODAY, offline=True, n_sims=50)


def test_schema_rejects_incomplete_team_context(repo):
    run_update(repo, TODAY, offline=True, n_sims=200, now=datetime(2024, 11, 19, 14, 0))
    pred = json.loads((repo.site_data / "predictions.json").read_text())
    del pred["team_context"]["NYK"]
    with pytest.raises(ValidationError):
        Predictions.model_validate(pred)


def _next_row(pred):
    return next(g for g in pred["games"] if not g["completed"])


def test_run_update_publishes_injury_report(repo_players, opp_p):
    pred = run_update(repo_players, TODAY, offline=True, n_sims=100)
    report = pred["injury_report"]
    assert report["available"] is True
    assert report["as_of"] == "2024-11-20T12:00:00Z"
    assert set(report["teams"]) == {"BOS", opp_p}
    out = next(p for p in report["teams"][opp_p] if p["status"] == "Out")
    assert out["p_out"] == 1 and out["name"] == _pname(opp_p, 0)
    assert out["player_id"] == TEAM_INDEX[opp_p] * 100
    assert len(report["teams"]["BOS"]) == 8
    assert all(p["status"] == "Available" and p["p_out"] == 0 for p in report["teams"]["BOS"])
    assert report["unmatched"] == []
    assert (repo_players.root / "data" / "injuries" / "2024-11-20.json").exists()
    on_disk = json.loads((repo_players.site_data / "predictions.json").read_text())
    Predictions.model_validate(on_disk)
    assert _next_row(on_disk)["win_prob_healthy"] is not None


def _force_avail_model(monkeypatch):
    monkeypatch.setattr("celtics.pipeline.MODEL_NAME", "ridge_avail")


def _boom():
    raise RuntimeError("espn down")


@pytest.mark.parametrize("bad", [_boom, lambda: [], lambda: {}, lambda: {"injuries": None}])
def test_run_update_without_usable_feed_applies_no_adjustment(repo_players, monkeypatch, bad):
    _force_avail_model(monkeypatch)
    seen = {}
    real = pipeline_module.prepare_sim_inputs

    def spy(feats, *args, **kwargs):
        seen["avail"] = feats.loc[~feats["completed"], ["home_avail", "away_avail"]]
        return real(feats, *args, **kwargs)

    monkeypatch.setattr("celtics.pipeline.fetch_espn_injuries", bad)
    monkeypatch.setattr("celtics.pipeline.prepare_sim_inputs", spy)
    pred = run_update(repo_players, TODAY, offline=True, n_sims=50)
    Predictions.model_validate(pred)
    assert pred["injury_report"]["available"] is False
    assert (seen["avail"] == 0).all().all() and len(seen["avail"]) > 0
    row = _next_row(pred)
    assert row["win_prob"] == row["win_prob_healthy"]
    assert [d["label"] for d in row["drivers"]][-1] == "Injuries & availability"


def test_run_update_with_feed_applies_live_avail(repo_players, opp_p, monkeypatch):
    _force_avail_model(monkeypatch)
    monkeypatch.setattr("celtics.pipeline.fetch_espn_injuries", lambda: _feed(opp_p, 7))  # best player
    pred = run_update(repo_players, TODAY, offline=True, n_sims=50)
    row = _next_row(pred)
    assert row["win_prob"] != row["win_prob_healthy"]


def test_run_update_rejects_invalid_dtd_log(repo_players):
    repo_players.site_data.mkdir(parents=True)
    (repo_players.site_data / "dtd_log.json").write_text(json.dumps([{"date": "2024-11-19"}]))
    with pytest.raises(DataValidationError, match="dtd_log"):
        run_update(repo_players, TODAY, offline=True, n_sims=50)


def test_run_train_uses_player_logs(repo_players, monkeypatch):
    _force_avail_model(monkeypatch)
    seen = {}
    real = pipeline_module.load_or_train

    def spy(feats, *args, **kwargs):
        seen["avail"] = feats.loc[feats["completed"], "home_avail"]
        return real(feats, *args, **kwargs)

    monkeypatch.setattr("celtics.pipeline.load_or_train", spy)
    run_train(repo_players, TODAY)
    assert (seen["avail"] != 0).any()


def test_run_update_reports_unmatched_and_unknown_team(repo, opp, monkeypatch):
    feed = _feed(opp)
    feed["injuries"][0]["injuries"].append(
        {"status": "Out", "athlete": {"displayName": "Nobody Special"}, "details": {}}
    )
    feed["injuries"].append({"displayName": "Seattle SuperSonics", "injuries": []})
    monkeypatch.setattr("celtics.pipeline.fetch_espn_injuries", lambda: feed)
    pred = run_update(repo, TODAY, offline=True, n_sims=100)
    report = pred["injury_report"]
    assert f"Nobody Special ({opp})" in report["unmatched"]
    assert "Seattle SuperSonics (unknown team)" in report["unmatched"]
    ghost = next(p for p in report["teams"][opp] if p["name"] == "Nobody Special")
    assert ghost["player_id"] is None and ghost["game_value"] == 0 and ghost["expected_minutes"] == 0


def test_run_update_without_player_logs_still_publishes_report(repo):
    pred = run_update(repo, TODAY, offline=True, n_sims=100)
    Predictions.model_validate(pred)
    row = _next_row(pred)
    assert row["win_prob"] == row["win_prob_healthy"]
    assert pred["injury_report"]["available"] is True
    assert not (repo.site_data / "dtd_log.json").exists()


def test_invalid_player_log_exits_3(repo_players, monkeypatch, capsys):
    path = player_season_path(repo_players.raw_dir, 2023)
    df = pd.read_parquet(path)
    pd.concat([df, df.iloc[[0]]], ignore_index=True).to_parquet(path, index=False)
    monkeypatch.chdir(repo_players.root)
    assert main(["--date", "2024-11-20", "update", "--offline"]) == 3
    assert "data validation failed" in capsys.readouterr().err
    assert not (repo_players.site_data / "predictions.json").exists()


def test_dtd_log_written_and_validated(repo_players, opp_p, monkeypatch):
    monkeypatch.setattr("celtics.pipeline.fetch_espn_injuries", lambda: _feed(opp_p, 1, "Day-To-Day"))
    pred = run_update(repo_players, TODAY, offline=True, n_sims=100)
    log = DtdLog.validate_python(json.loads((repo_players.site_data / "dtd_log.json").read_text()))
    assert [c.name for c in log] == [_pname(opp_p, 1)] and log[0].played is None
    assert pred["injury_report"]["p_dtd"] == 0.5
    dtd = next(p for p in pred["injury_report"]["teams"][opp_p] if p["status"] == "Day-To-Day")
    assert dtd["p_out"] == 0.5



def test_healthy_equals_prediction_when_no_injuries(repo_players, monkeypatch):
    _force_avail_model(monkeypatch)
    monkeypatch.setattr("celtics.pipeline.fetch_espn_injuries", lambda: {"injuries": []})
    seen = {}
    real = pipeline_module.live_avail_and_baselines

    def spy(*args, **kwargs):
        live, baselines = real(*args, **kwargs)
        seen["baselines"], seen["live"] = baselines, live
        return live, baselines

    monkeypatch.setattr("celtics.pipeline.live_avail_and_baselines", spy)
    pred = run_update(repo_players, TODAY, offline=True, n_sims=50)
    row = _next_row(pred)
    assert pred["injury_report"]["available"] is True
    assert max(seen["baselines"].values()) > 0  # healthy is not avail 0
    first = seen["live"][seen["live"]["game_id"] == row["game_id"]].iloc[0]
    assert (first["home_avail"], first["away_avail"]) == (
        seen["baselines"]["BOS"] if row["home"] else seen["baselines"][row["opponent"]],
        seen["baselines"][row["opponent"]] if row["home"] else seen["baselines"]["BOS"],
    )
    assert round(row["win_prob"], 4) == round(row["win_prob_healthy"], 4)


def test_raw_feed_write_failure_keeps_the_parsed_feed(repo_players, opp_p, monkeypatch):
    def broken(path, obj):
        if "injuries" in str(path):
            raise OSError("disk full")
        return real(path, obj)

    real = pipeline_module.write_json_atomic
    monkeypatch.setattr("celtics.pipeline.write_json_atomic", broken)
    pred = run_update(repo_players, TODAY, offline=True, n_sims=50)
    assert pred["injury_report"]["available"] is True
    assert pred["injury_report"]["teams"][opp_p]
