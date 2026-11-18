import json
from datetime import date, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

import celtics.pipeline as pipeline_module
from celtics.cli import main
from celtics.config import DEFAULT_TEAM_SD, TRACK_PICKS_FROM, Paths
from celtics.data.cache import player_season_path, season_path
from celtics.data.fetch import FetchBlockedError
from celtics.data.validate import DataValidationError
from celtics.export.schema import BacktestReport, DtdLog, HistoryEntry, League, Slate, TeamFile
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
    """Four synthetic seasons with no player logs (availability 0); ESPN patched."""
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


@pytest.fixture
def _pin_ridge(monkeypatch):
    """For tests asserting ridge-specific artifacts (model file names, exactly 4 drivers)."""
    monkeypatch.setattr("celtics.pipeline.MODEL_NAME", "ridge")


def _read(path):
    return json.loads(path.read_text())


def _bos(repo):
    return _read(repo.site_data / "teams" / "BOS.json")


def _snapshot(repo):
    return {p.relative_to(repo.site_data): p.read_bytes() for p in repo.site_data.rglob("*") if p.is_file()}


@pytest.mark.usefixtures("_pin_ridge")
def test_run_update_writes_valid_outputs(repo):
    returned = run_update(repo, TODAY, offline=True, n_sims=200, now=datetime(2024, 11, 19, 14, 0))  # day of the next synthetic round
    data = repo.site_data
    assert sorted(p.stem for p in (data / "teams").glob("*.json")) == sorted(TEAM_INDEX)
    assert sorted(p.stem for p in (data / "history").glob("*.json")) == sorted(TEAM_INDEX)
    slate = Slate.model_validate(_read(data / "slate.json"))
    assert [d.date for d in slate.days] == ["2024-11-19", "2024-11-20", "2024-11-21"]
    assert len(slate.days[0].games) == 15 and slate.days[1].games == [] and len(slate.days[2].games) == 15
    league = League.model_validate(_read(data / "league.json"))
    assert league.completed_games == 15 * 15 and league.season == "2024-25"
    for name in ("slate.json", "league.json", "teams/BOS.json"):
        assert "\n" not in (data / name).read_text()  # compact
    bos = _bos(repo)
    TeamFile.model_validate(bos)
    assert returned == bos  # the CLI summary line reads the Boston file
    assert bos["season"] == "2024-25"
    assert bos["metrics"]["record"]["since"] == TRACK_PICKS_FROM
    assert len(bos["games"]) == 82
    first = next(g for g in bos["games"] if not g["completed"])
    assert len(first["drivers"]) == 4
    assert bos["team_summary"]["wins"] + bos["team_summary"]["losses"] == 15
    for team in TEAM_INDEX:
        TeamFile.model_validate(_read(data / "teams" / f"{team}.json"))
        history = _read(data / "history" / f"{team}.json")
        assert len(history) == 1
        HistoryEntry.model_validate(history[0])
    assert not (data / "predictions.json").exists()
    assert not (data / "history.json").exists()
    log = json.loads((repo.site_data / "pregame_log.json").read_text())
    # the Nov 19 round is dated before today (Nov 20) and unplayed: frozen, never back-filled (league-wide log: 30*66/2 games)
    assert len(log) == 990
    assert (repo.model_dir / "ridge.joblib").exists()


def test_run_update_twice_same_day_keeps_one_history_entry(repo):
    run_update(repo, TODAY, offline=True, n_sims=100)
    run_update(repo, TODAY, offline=True, n_sims=100)
    assert len(_read(repo.site_data / "history" / "BOS.json")) == 1


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
    assert not repo.site_data.exists() or _snapshot(repo) == {}


def _corrupt(repo):
    path = season_path(repo.raw_dir, 2023)
    df = pd.read_parquet(path)
    pd.concat([df, df.iloc[[0]]], ignore_index=True).to_parquet(path, index=False)


def _meta(repo):
    return json.loads((repo.model_dir / "ridge.json").read_text())["trained_on"]


@pytest.mark.usefixtures("_pin_ridge")
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
    assert not repo.site_data.exists() or _snapshot(repo) == {}


def test_run_backtest_rejects_invalid_data(repo):
    _corrupt(repo)
    with pytest.raises(DataValidationError):
        run_backtest(repo, TODAY, offline=True, test_seasons=[2022, 2023], n_sims=50)
    assert not (repo.site_data / "backtest.json").exists()


@pytest.mark.usefixtures("_pin_ridge")
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


def _erase_result(repo, involving_bos):
    """A completed result vanishes from the cache (BOS's last one, or a non-BOS game's)."""
    path = season_path(repo.raw_dir, 2024)
    df = pd.read_parquet(path)
    has_bos = (df["home_team"] == "BOS") | (df["away_team"] == "BOS")
    pool = df[(has_bos if involving_bos else ~has_bos) & df["home_pts"].notna()]
    df.loc[df["game_id"] == pool["game_id"].iloc[-1], ["home_pts", "away_pts", "poss"]] = float("nan")
    df.to_parquet(path, index=False)


def test_backwards_check_league_wide(repo):
    run_update(repo, TODAY, offline=True, n_sims=50)
    _erase_result(repo, involving_bos=False)  # Boston's record is unchanged; the league's is not
    before = _snapshot(repo)
    with pytest.raises(DataValidationError, match="completed games went backwards"):
        run_update(repo, TODAY, offline=True, n_sims=50)
    assert _snapshot(repo) == before


def test_backwards_check_skips_another_season(repo):
    run_update(repo, TODAY, offline=True, n_sims=50)
    path = repo.site_data / "league.json"
    league = _read(path)
    path.write_text(json.dumps({**league, "season": "2023-24", "completed_games": 10_000}))
    run_update(repo, TODAY, offline=True, n_sims=50)
    assert _read(path)["completed_games"] == 15 * 15


def test_backwards_check_uses_old_predictions_before_first_league_file(repo):
    repo.site_data.mkdir(parents=True)
    old = {"season": "2024-25", "team_summary": {"wins": 10, "losses": 6}}  # BOS has 15 done
    (repo.site_data / "predictions.json").write_text(json.dumps(old))
    before = _snapshot(repo)
    with pytest.raises(DataValidationError, match="completed games went backwards"):
        run_update(repo, TODAY, offline=True, n_sims=50)
    assert _snapshot(repo) == before


def test_first_launch_migrates_old_files(repo):
    cur = pd.read_parquet(season_path(repo.raw_dir, 2024))
    away = cur[(cur["away_team"] == "BOS") & cur["home_pts"].notna()]
    gid = away["game_id"].iloc[0]
    repo.site_data.mkdir(parents=True)
    old_log = {gid: {"win_prob": 0.6, "pred_margin": 2.0, "logged_at": "2024-10-20T14:00:00+00:00"}}
    (repo.site_data / "pregame_log.json").write_text(json.dumps(old_log))
    seeded = {
        "date": "2024-11-19", "record": "9-5", "projected_wins_median": 45.0, "p10": 38.0,
        "p90": 52.0, "playoff_prob": 0.7, "top6_prob": 0.5, "seed1_prob": 0.1,
    }  # fmt: skip
    (repo.site_data / "history.json").write_text(json.dumps([seeded]))
    old = {"season": "2024-25", "team_summary": {"wins": 10, "losses": 5}}
    (repo.site_data / "predictions.json").write_text(json.dumps(old))
    run_update(repo, TODAY, offline=True, n_sims=50)
    log = _read(repo.site_data / "pregame_log.json")
    assert log[gid]["home_win_prob"] == 0.4 and log[gid]["pred_margin"] == -2.0
    history = _read(repo.site_data / "history" / "BOS.json")
    assert history[0] == seeded and len(history) == 2
    assert not (repo.site_data / "predictions.json").exists()
    assert not (repo.site_data / "history.json").exists()


def test_invalid_output_writes_nothing(repo, monkeypatch):
    run_update(repo, TODAY, offline=True, n_sims=50)
    before = _snapshot(repo)
    real = pipeline_module.team_file

    def broken(team, *args, **kwargs):
        out = real(team, *args, **kwargs)
        if team == "NYK":
            out["team_summary"]["p_playoffs"] = 1.5
        return out

    monkeypatch.setattr("celtics.pipeline.team_file", broken)
    with pytest.raises(DataValidationError):
        run_update(repo, date(2024, 11, 21), offline=True, n_sims=50)
    assert _snapshot(repo) == before


def test_run_update_rejects_corrupt_pregame_log(repo):
    repo.site_data.mkdir(parents=True)
    bad = {"g1": {"home_win_prob": 3.0, "pred_margin": 1.0, "logged_at": "2024-11-19T14:00:00"}}
    (repo.site_data / "pregame_log.json").write_text(json.dumps(bad))
    with pytest.raises(DataValidationError, match="pregame_log"):
        run_update(repo, TODAY, offline=True, n_sims=50)


def test_schema_rejects_incomplete_team_context(repo):
    run_update(repo, TODAY, offline=True, n_sims=200, now=datetime(2024, 11, 19, 14, 0))
    league = _read(repo.site_data / "league.json")
    del league["team_context"]["NYK"]
    with pytest.raises(ValidationError):
        League.model_validate(league)


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
    assert report["teams"]["BOS"] == []  # injured players only; BOS has none listed
    assert report["unmatched"] == []
    assert (repo_players.root / "data" / "injuries" / "2024-11-20.json").exists()
    on_disk = _bos(repo_players)
    TeamFile.model_validate(on_disk)
    assert on_disk == pred
    assert _next_row(on_disk)["win_prob_healthy"] is not None
    slate = _read(repo_players.site_data / "slate.json")
    game = next(
        g for d in slate["days"] for g in d["games"] if not g["completed"] and opp_p in (g["home"], g["away"])
    )
    side = "home" if game["home"] == opp_p else "away"
    assert [p["name"] for p in game["injuries"][side]] == [_pname(opp_p, 0)]


def _force_avail_model(monkeypatch):
    monkeypatch.setattr("celtics.pipeline.MODEL_NAME", "ridge_avail")


def _boom():
    raise RuntimeError("espn down")


@pytest.mark.parametrize("bad", [_boom, lambda: [], lambda: {}, lambda: {"injuries": None}])
def test_feed_unavailable_publishes_without_adjustment(repo_players, monkeypatch, bad):
    _force_avail_model(monkeypatch)
    seen = {}
    real = pipeline_module.prepare_sim_inputs

    def spy(feats, *args, **kwargs):
        seen["avail"] = feats.loc[~feats["completed"], ["home_avail", "away_avail"]]
        return real(feats, *args, **kwargs)

    monkeypatch.setattr("celtics.pipeline.fetch_espn_injuries", bad)
    monkeypatch.setattr("celtics.pipeline.prepare_sim_inputs", spy)
    pred = run_update(repo_players, TODAY, offline=True, n_sims=50)
    TeamFile.model_validate(pred)
    assert pred["injury_report"]["available"] is False
    assert _bos(repo_players)["injury_report"]["available"] is False
    assert (seen["avail"] == 0).all().all() and len(seen["avail"]) > 0
    row = _next_row(pred)
    assert row["win_prob"] == row["win_prob_healthy"]
    assert [d["label"] for d in row["drivers"]][-1] == "Injuries & availability"
    upcoming = [g for d in _read(repo_players.site_data / "slate.json")["days"] for g in d["games"] if not g["completed"]]
    assert upcoming
    for g in upcoming:
        assert g["injuries"] == {"home": [], "away": []}
        assert g["home_win_prob_healthy"] == g["home_win_prob"]


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
    assert set(report["teams"]) == {"BOS", opp}
    assert f"Nobody Special ({opp})" in report["unmatched"]
    assert all(u.endswith((f"({opp})", "(BOS)")) for u in report["unmatched"])  # only this game's teams
    ghost = next(p for p in report["teams"][opp] if p["name"] == "Nobody Special")
    assert ghost["player_id"] is None and ghost["game_value"] == 0 and ghost["expected_minutes"] == 0


def test_run_update_without_player_logs_still_publishes_report(repo):
    pred = run_update(repo, TODAY, offline=True, n_sims=100)
    TeamFile.model_validate(pred)
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
    assert not repo_players.site_data.exists() or _snapshot(repo_players) == {}


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
    def broken(path, obj, **kwargs):
        if "injuries" in str(path):
            raise OSError("disk full")
        return real(path, obj, **kwargs)

    real = pipeline_module.write_json_atomic
    monkeypatch.setattr("celtics.pipeline.write_json_atomic", broken)
    pred = run_update(repo_players, TODAY, offline=True, n_sims=50)
    assert pred["injury_report"]["available"] is True
    assert pred["injury_report"]["teams"][opp_p]
