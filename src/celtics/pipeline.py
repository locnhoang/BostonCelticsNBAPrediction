"""End-to-end orchestration: data -> features -> model -> simulation -> JSON."""

import logging
from datetime import UTC, date, datetime

import pandas as pd
from pydantic import ValidationError

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
from celtics.data.cache import load_all, load_all_players, refresh, refresh_players
from celtics.data.fetch import fetch_espn_injuries, fetch_team_roster
from celtics.data.parse import empty_players
from celtics.data.validate import DataValidationError, validate_games, validate_players
from celtics.evaluate.backtest import season_coverage, walk_forward
from celtics.evaluate.metrics import calibration_table, summarize
from celtics.export.build import (
    conference_table,
    game_rows,
    healthy_win_prob,
    history_entry,
    injury_teams,
    live_metrics,
    read_json,
    seed_probabilities,
    team_summary,
    update_pregame_log,
    upsert_history,
    write_json_atomic,
)
from celtics.export.context import team_context
from celtics.export.schema import BacktestReport, DtdLog, HistoryEntry, Predictions, PregameLog
from celtics.features.availability import CURRENT_COLUMNS, current_player_values, live_avail
from celtics.features.build import AVAIL_FEATURES, build_features, current_team_state
from celtics.injuries import (
    INJURY_COLUMNS,
    learn_p_dtd,
    match_players,
    parse_injuries,
    unknown_teams,
    update_dtd_log,
)
from celtics.models.margin import MODELS, MarginModel, load_model, make_model, save_model
from celtics.simulate.inputs import prepare_sim_inputs
from celtics.simulate.season import simulate_season
from celtics.teams import CONFERENCE_OF, TEAM_INDEX

log = logging.getLogger(__name__)

AVAILABILITY_NOTE = (
    "Backtest uses actual absences from box scores; live predictions use the injury report, "
    "so live accuracy will be somewhat lower."
)


def _now(now: datetime | None) -> str:
    return (now or datetime.now(UTC)).isoformat(timespec="seconds")


def _load_games(paths: Paths, today: date, offline: bool):
    return load_all(paths.raw_dir) if offline else refresh(paths.raw_dir, today)


def _load_players(paths: Paths, today: date, offline: bool, games):
    """Player logs, validated; None (avail = 0) when none are cached offline."""
    players = load_all_players(paths.raw_dir) if offline else refresh_players(paths.raw_dir, today)
    if players.empty:
        log.warning("no player logs available; availability features are 0")
        return None
    validate_players(players, games)
    return players


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
    model = make_model(MODEL_NAME).fit(train, train["margin"])
    save_model(model, model_path)
    write_json_atomic(meta_path, {"trained_on": today.isoformat()})
    model.trained_on = today.isoformat()
    return model


def run_train(paths: Paths, today: date) -> MarginModel:
    games = load_all(paths.raw_dir)
    validate_games(games, current_season(today), today=today)
    players = _load_players(paths, today, offline=True, games=games)
    feats = build_features(games, players)
    return load_or_train(feats, paths, today, force=True)


def read_team_sd(paths: Paths) -> float:
    report = read_json(paths.site_data / "backtest.json", None)
    return float(report["chosen_team_sd"]) if report else DEFAULT_TEAM_SD


def _check_not_backwards(games, season: int, existing: dict | None) -> None:
    """Refuse to publish if the data has fewer completed games than the last published run."""
    if not existing or existing.get("season") != season_label(season):
        return
    cur = games[(games["season"] == season) & ((games["home_team"] == TEAM) | (games["away_team"] == TEAM))]
    done = int(cur["home_pts"].notna().sum())
    summary = existing["team_summary"]
    published = summary["wins"] + summary["losses"]
    if done < published:
        raise DataValidationError(
            f"completed games went backwards: {TEAM} has {done} now, {published} in last published run"
        )


def _empty_matched() -> pd.DataFrame:
    return pd.DataFrame(columns=INJURY_COLUMNS).assign(player_id=pd.array([], dtype="Int64"))


def _load_feed(paths: Paths, today: date, players, season: int):
    """Fetch, parse and match the ESPN feed -> (parsed, matched, unmatched, as_of), or None.

    Any failure (fetch, malformed feed, matching) is logged and means "no feed": no adjustment.
    """
    try:
        raw = fetch_espn_injuries()
        if not isinstance(raw, dict) or not isinstance(raw.get("injuries"), list):
            raise ValueError("unexpected injury feed shape")
        parsed = parse_injuries(raw)
        matched, unmatched = match_players(
            parsed, players if players is not None else empty_players(), season
        )
        unmatched = [*unmatched, *(f"{n} (unknown team)" for n in unknown_teams(raw))]
        as_of = raw.get("timestamp")
        as_of = as_of if isinstance(as_of, str) else None
        write_json_atomic(paths.injuries_dir / f"{today.isoformat()}.json", raw)
    except Exception as exc:  # an injury-feed problem never blocks a run
        log.warning("ESPN injury feed unusable, publishing no adjustment: %s", exc)
        return None
    return parsed, matched, unmatched, as_of


def _fetch_rosters(teams: list[str], season: int) -> dict[str, pd.DataFrame | None]:
    rosters: dict[str, pd.DataFrame | None] = {}
    for team in teams:
        try:
            rosters[team] = fetch_team_roster(team, season)
        except Exception as exc:
            log.warning("roster for %s unavailable, using recent players: %s", team, exc)
            rosters[team] = None
    return rosters


def _apply_live_avail(feats: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Overwrite the availability features of the games `live` covers."""
    if live.empty:
        return feats
    feats = feats.copy()
    lookup = live.set_index("game_id")
    mask = feats["game_id"].isin(lookup.index)
    for col in AVAIL_FEATURES:
        feats.loc[mask, col] = feats.loc[mask, "game_id"].map(lookup[col]).to_numpy()
    return feats


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
    validate_games(games, season, today=today)
    _check_not_backwards(games, season, read_json(paths.site_data / "predictions.json", None))
    players = _load_players(paths, today, offline, games)

    feed = _load_feed(paths, today, players, season)
    parsed, matched, unmatched, as_of = feed or (
        pd.DataFrame(columns=INJURY_COLUMNS), _empty_matched(), [], None
    )

    dtd_log = read_json(paths.site_data / "dtd_log.json", [])
    try:
        DtdLog.validate_python(dtd_log)
    except ValidationError as exc:
        raise DataValidationError(f"dtd_log.json is invalid: {exc}") from exc
    if players is not None:  # without box scores a case could not be resolved honestly
        dtd_log = update_dtd_log(dtd_log, matched, games, players, today)
    DtdLog.validate_python(dtd_log)
    p_dtd = learn_p_dtd(dtd_log)

    feats = build_features(games, players)
    current = pd.DataFrame(columns=CURRENT_COLUMNS)
    if players is not None:
        if feed is not None:  # no feed, no adjustment: incomplete games keep avail 0
            live = live_avail(games, players, matched, season, today, p_dtd)
            feats = _apply_live_avail(feats, live)
        current = current_player_values(games, players, season)
    model = load_or_train(feats, paths, today, force=force_train)
    inputs = prepare_sim_inputs(feats, model, current_team_state(games), season, GAMES_PER_TEAM)
    result = simulate_season(
        inputs, n_sims=n_sims, team_sd=read_team_sd(paths), seed=int(today.strftime("%Y%m%d"))
    )

    generated_at = _now(now)
    pregame_log = read_json(paths.site_data / "pregame_log.json", {})
    rows = game_rows(feats, model, season, TEAM, pregame_log)
    pregame_log = update_pregame_log(pregame_log, rows, generated_at, today=today.isoformat())
    upcoming = next((r for r in rows if not r["completed"]), None)
    teams: dict[str, list[dict]] = {}
    if upcoming is not None:
        upcoming["win_prob_healthy"] = healthy_win_prob(feats, model, upcoming["game_id"], TEAM)
        pair = [TEAM, upcoming["opponent"]]
        game_date = date.fromisoformat(upcoming["date"])
        rosters = _fetch_rosters(pair, season)
        try:
            teams = injury_teams(pair, parsed, matched, current, rosters, today, game_date, p_dtd)
        except Exception as exc:
            log.warning("injury display failed, falling back to recent players only: %s", exc)
            teams = injury_teams(
                pair, parsed.iloc[0:0], _empty_matched(), current, dict.fromkeys(pair), today, game_date, p_dtd
            )
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
        "team_context": team_context(games, season),
        "injury_report": {
            "available": feed is not None,
            "as_of": as_of,
            "p_dtd": p_dtd,
            "unmatched": unmatched,
            "teams": teams,
        },
    }
    Predictions.model_validate(predictions)
    PregameLog.validate_python(pregame_log)
    entry = history_entry(today, summary)
    HistoryEntry.model_validate(entry)
    history = upsert_history(read_json(paths.site_data / "history.json", []), entry)

    write_json_atomic(paths.site_data / "predictions.json", predictions)
    write_json_atomic(paths.site_data / "history.json", history)
    write_json_atomic(paths.site_data / "pregame_log.json", pregame_log)
    if players is not None:
        write_json_atomic(paths.site_data / "dtd_log.json", dtd_log)
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
    validate_games(games, current_season(today), today=today)
    players = _load_players(paths, today, offline, games)
    feats = build_features(games, players)
    scores, main_preds = {}, None
    for name in MODELS:
        preds = walk_forward(feats, name, test_seasons)
        scores[name] = summarize(preds["prob"], preds["outcome"])
        if name == MODEL_NAME:
            main_preds = preds
    coverage = season_coverage(games, MODEL_NAME, test_seasons, TEAM_SD_GRID, n_sims=n_sims, players=players)
    chosen = min(coverage, key=lambda sd: (abs(coverage[sd] - 0.8), sd))
    report = {
        "generated_at": _now(now),
        "test_seasons": list(test_seasons),
        "main_model": MODEL_NAME,
        "models": scores,
        "calibration": calibration_table(main_preds["prob"], main_preds["outcome"]),
        "coverage": {f"{sd:.1f}": round(v, 4) for sd, v in coverage.items()},
        "chosen_team_sd": chosen,
        "availability_note": AVAILABILITY_NOTE,
    }
    BacktestReport.model_validate(report)
    write_json_atomic(paths.site_data / "backtest.json", report)
    return report
