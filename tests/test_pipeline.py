import json
from datetime import date, datetime

import pytest

from celtics.cli import main
from celtics.config import Paths
from celtics.data.cache import season_path
from celtics.data.fetch import FetchBlockedError
from celtics.export.schema import BacktestReport, HistoryEntry, Predictions
from celtics.pipeline import run_backtest, run_update
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
    run_update(repo, TODAY, offline=True, n_sims=200, now=datetime(2024, 11, 20, 14, 0))
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
