"""End-to-end orchestration: data -> features -> model -> simulation -> JSON."""

from datetime import UTC, date, datetime

from celtics.config import (
    BACKTEST_COVERAGE_SIMS,
    BACKTEST_SEASONS,
    DEFAULT_TEAM_SD,
    FIRST_TRAIN_SEASON,
    GAMES_PER_TEAM,
    MODEL_NAME,
    N_SIMS,
    RETRAIN_DAYS,
    TEAM,
    TEAM_SD_GRID,
    Paths,
    current_season,
    season_label,
)
from celtics.data.cache import load_all, refresh
from celtics.data.validate import validate_games
from celtics.evaluate.backtest import season_coverage, walk_forward
from celtics.evaluate.metrics import calibration_table, summarize
from celtics.export.build import (
    conference_table,
    game_rows,
    history_entry,
    live_metrics,
    read_json,
    seed_probabilities,
    team_summary,
    update_pregame_log,
    upsert_history,
    write_json_atomic,
)
from celtics.export.schema import BacktestReport, HistoryEntry, Predictions
from celtics.features.build import FEATURES, build_features, current_team_state
from celtics.models.margin import MODELS, MarginModel, load_model, make_model, save_model
from celtics.simulate.inputs import prepare_sim_inputs
from celtics.simulate.season import simulate_season
from celtics.teams import CONFERENCE_OF, TEAM_INDEX


def _now(now: datetime | None) -> str:
    return (now or datetime.now(UTC)).isoformat(timespec="seconds")


def _load_games(paths: Paths, today: date, offline: bool):
    return load_all(paths.raw_dir) if offline else refresh(paths.raw_dir, today)


def load_or_train(features, paths: Paths, today: date, force: bool = False) -> MarginModel:
    model_path = paths.model_dir / f"{MODEL_NAME}.joblib"
    meta_path = paths.model_dir / f"{MODEL_NAME}.json"
    meta = read_json(meta_path, None)
    if not force and model_path.exists() and meta:
        trained_on = date.fromisoformat(meta["trained_on"])
        if (today - trained_on).days < RETRAIN_DAYS:
            model = load_model(model_path)
            model.trained_on = meta["trained_on"]
            return model
    train = features[features["completed"] & (features["season"] >= FIRST_TRAIN_SEASON)]
    model = make_model(MODEL_NAME).fit(train[FEATURES], train["margin"])
    save_model(model, model_path)
    write_json_atomic(meta_path, {"trained_on": today.isoformat()})
    model.trained_on = today.isoformat()
    return model


def read_team_sd(paths: Paths) -> float:
    report = read_json(paths.site_data / "backtest.json", None)
    return float(report["chosen_team_sd"]) if report else DEFAULT_TEAM_SD


def run_update(
    paths: Paths,
    today: date,
    offline: bool = False,
    force_train: bool = False,
    n_sims: int = N_SIMS,
    now: datetime | None = None,
) -> dict:
    season = current_season(today)
    games = _load_games(paths, today, offline)
    validate_games(games, season)
    feats = build_features(games)
    model = load_or_train(feats, paths, today, force=force_train)
    inputs = prepare_sim_inputs(feats, model, current_team_state(games), season, GAMES_PER_TEAM)
    result = simulate_season(
        inputs, n_sims=n_sims, team_sd=read_team_sd(paths), seed=int(today.strftime("%Y%m%d"))
    )

    generated_at = _now(now)
    log = read_json(paths.site_data / "pregame_log.json", {})
    rows = game_rows(feats, model, season, TEAM, log)
    log = update_pregame_log(log, rows, generated_at)
    i = TEAM_INDEX[TEAM]
    summary = team_summary(result, TEAM, inputs.current_wins[i], inputs.current_losses[i])
    predictions = {
        "generated_at": generated_at,
        "model_version": f"{MODEL_NAME}-{model.trained_on}",
        "season": season_label(season),
        "team_summary": summary,
        "games": rows,
        "seed_probabilities": seed_probabilities(result, TEAM),
        "conference_table": conference_table(result, inputs, CONFERENCE_OF[TEAM]),
        "metrics": {"live": live_metrics(rows)},
    }
    Predictions.model_validate(predictions)
    entry = history_entry(today, summary)
    HistoryEntry.model_validate(entry)
    history = upsert_history(read_json(paths.site_data / "history.json", []), entry)

    write_json_atomic(paths.site_data / "predictions.json", predictions)
    write_json_atomic(paths.site_data / "history.json", history)
    write_json_atomic(paths.site_data / "pregame_log.json", log)
    return predictions


def run_backtest(
    paths: Paths,
    today: date,
    offline: bool = False,
    test_seasons: list[int] = BACKTEST_SEASONS,
    n_sims: int = BACKTEST_COVERAGE_SIMS,
    now: datetime | None = None,
) -> dict:
    games = _load_games(paths, today, offline)
    feats = build_features(games)
    scores, main_preds = {}, None
    for name in MODELS:
        preds = walk_forward(feats, name, test_seasons)
        scores[name] = summarize(preds["prob"], preds["outcome"])
        if name == MODEL_NAME:
            main_preds = preds
    coverage = season_coverage(games, MODEL_NAME, test_seasons, TEAM_SD_GRID, n_sims=n_sims)
    chosen = min(coverage, key=lambda sd: (abs(coverage[sd] - 0.8), sd))
    report = {
        "generated_at": _now(now),
        "test_seasons": list(test_seasons),
        "main_model": MODEL_NAME,
        "models": scores,
        "calibration": calibration_table(main_preds["prob"], main_preds["outcome"]),
        "coverage": {f"{sd:.1f}": round(v, 4) for sd, v in coverage.items()},
        "chosen_team_sd": chosen,
    }
    BacktestReport.model_validate(report)
    write_json_atomic(paths.site_data / "backtest.json", report)
    return report
