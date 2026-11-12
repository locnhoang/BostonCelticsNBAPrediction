import json
from datetime import date, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from celtics.cli import main
from celtics.config import DEFAULT_TEAM_SD, Paths
from celtics.data.cache import season_path
from celtics.data.fetch import FetchBlockedError
from celtics.data.validate import DataValidationError
from celtics.export.schema import BacktestReport, HistoryEntry, Predictions
from celtics.pipeline import read_team_sd, run_backtest, run_train, run_update
from tests.synth import make_league

TODAY = date(2024, 11, 20)


@pytest.fixture
def repo(tmp_path):
    games = make_league([2021, 2022, 2023, 2024], current=2024, completed_rounds=15)
    raw = Paths(tmp_path).raw_dir
    raw.mkdir(parents=True)
    for season, g in games.groupby("season"):
        g.to_parquet(season_path(raw, season), index=False)
    return Paths(tmp_path)


def test_run_update_writes_valid_outputs(repo):
    run_update(repo, TODAY, offline=True, n_sims=200, now=datetime(2024, 11, 19, 14, 0))  # day of the next synthetic round
    pred = json.loads((repo.site_data / "predictions.json").read_text())
    Predictions.model_validate(pred)
    assert pred["season"] == "2024-25"
    assert len(pred["games"]) == 82
    assert pred["team_summary"]["wins"] + pred["team_summary"]["losses"] == 15
    history = json.loads((repo.site_data / "history.json").read_text())
    assert len(history) == 1
    HistoryEntry.model_validate(history[0])
    log = json.loads((repo.site_data / "pregame_log.json").read_text())
    assert len(log) == 67
    assert (repo.model_dir / "ridge.joblib").exists()


def test_run_update_twice_same_day_keeps_one_history_entry(repo):
    run_update(repo, TODAY, offline=True, n_sims=100)
    run_update(repo, TODAY, offline=True, n_sims=100)
    assert len(json.loads((repo.site_data / "history.json").read_text())) == 1


def test_run_backtest_writes_report(repo):
    report = run_backtest(repo, TODAY, offline=True, test_seasons=[2022, 2023], n_sims=100)
    BacktestReport.model_validate(report)
    assert set(report["models"]) == {"homecourt", "elo", "ridge", "gbm"}
    assert report["chosen_team_sd"] in [0.0, 1.0, 2.0, 2.5, 3.0, 4.0]
    assert (repo.site_data / "backtest.json").exists()


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


def test_refit_cadence(repo):
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
    last_day = df.loc[df["home_pts"].notna(), "date"].max()
    df.loc[df["date"] == last_day, ["home_pts", "away_pts", "poss"]] = float("nan")  # results vanish
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
