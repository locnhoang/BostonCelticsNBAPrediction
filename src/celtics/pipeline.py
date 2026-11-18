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
    LEAGUE_TRACK_FROM,
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
from celtics.data.fetch import fetch_espn_injuries
from celtics.data.parse import empty_players
from celtics.data.validate import DataValidationError, validate_games, validate_players
from celtics.evaluate.backtest import season_coverage, walk_forward
from celtics.evaluate.metrics import calibration_table, summarize
from celtics.export.build import (
    history_entry,
    injury_teams,
    read_json,
    upsert_history,
    write_json_atomic,
)
from celtics.export.context import team_context
from celtics.export.league import (
    league_file,
    migrate_pregame_log,
    next_game_ids,
    pick_record,
    slate_days,
    team_file,
    update_league_pregame_log,
)
from celtics.export.schema import (
    BacktestReport,
    DtdLog,
    HistoryEntry,
    League,
    PregameLog,
    Slate,
    TeamFile,
)
from celtics.features.availability import (
    CURRENT_COLUMNS,
    current_player_values,
    live_avail_and_baselines,
)
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
from celtics.teams import TEAM_INDEX

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


def _completed_games(games, season: int) -> int:
    return int(games.loc[games["season"] == season, "home_pts"].notna().sum())


def _check_not_backwards(games, season: int, league: dict | None, legacy: dict | None) -> None:
    """Refuse to publish if the data has fewer completed games than the last published run.

    Compares the league-wide count with `league.json`; before the first league file exists, falls
    back to Boston's record in the retired `predictions.json`.
    """
    if league is not None:
        if league.get("season") != season_label(season):
            return
        done, published = _completed_games(games, season), league["completed_games"]
        if done < published:
            raise DataValidationError(
                f"completed games went backwards: the league has {done} now, {published} in last published run"
            )
        return
    if not legacy or legacy.get("season") != season_label(season):
        return
    cur = games[(games["season"] == season) & ((games["home_team"] == TEAM) | (games["away_team"] == TEAM))]
    done = int(cur["home_pts"].notna().sum())
    summary = legacy["team_summary"]
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
    except Exception as exc:  # an injury-feed problem never blocks a run
        log.warning("ESPN injury feed unusable, publishing no adjustment: %s", exc)
        return None
    try:
        write_json_atomic(paths.injuries_dir / f"{today.isoformat()}.json", raw)
    except Exception as exc:  # the raw copy is an archive; losing it keeps the parsed feed
        log.warning("could not save the raw ESPN feed: %s", exc)
    return parsed, matched, unmatched, as_of


def _injury_lookup(parsed, matched, current, today: date, p_dtd: float):
    """`(home, away, game_date, not_next) -> {team: [injured players]}`, cached per game.

    A display failure falls back to empty lists for that game; it never blocks a run.
    """
    cache: dict[tuple, dict[str, list[dict]]] = {}

    def lookup(home: str, away: str, game_date: date, not_next: frozenset[str]) -> dict[str, list[dict]]:
        key = (home, away, game_date, not_next)
        if key not in cache:
            pair = [home, away]
            try:
                cache[key] = injury_teams(
                    pair, parsed, matched, current, dict.fromkeys(pair), today, game_date, p_dtd, not_next
                )
            except Exception as exc:
                log.warning("injury display failed for %s @ %s, showing none: %s", away, home, exc)
                cache[key] = {home: [], away: []}
        return cache[key]

    return lookup


def _team_injury_report(team: str, game, next_pairs, lookup, available, as_of, p_dtd, unmatched) -> dict:
    """The injury report for `team`'s next game (`game`, a feature row, or None if none is left)."""
    teams: dict[str, list[dict]] = {}
    shown = [team]
    if game is not None:
        home, away = game.home_team, game.away_team
        opp = away if home == team else home
        not_next = frozenset(t for t in (home, away) if (game.game_id, t) not in next_pairs)
        by_team = lookup(home, away, game.date.date(), not_next)
        teams = {team: by_team.get(team, []), opp: by_team.get(opp, [])}
        shown.append(opp)
    return {
        "available": available,
        "as_of": as_of,
        "p_dtd": p_dtd,
        "unmatched": [u for u in unmatched if u.endswith(tuple(f"({t})" for t in shown))],
        "teams": teams,
    }


def _slate_record(slate_games: list[dict], games, season: int, pregame_log: dict) -> dict:
    """League pick record: yesterday's slate games, and every completed game since tracking began."""
    done = games[(games["season"] == season) & games["home_pts"].notna()]
    finished = [
        {
            "completed": True,
            "date": d,
            "home": home,
            "away": away,
            "home_pts": hp,
            "away_pts": ap,
            "pregame_home_win_prob": pregame_log.get(gid, {}).get("home_win_prob"),
        }
        for gid, d, home, away, hp, ap in zip(
            done["game_id"], done["date"].dt.strftime("%Y-%m-%d"), done["home_team"],
            done["away_team"], done["home_pts"], done["away_pts"], strict=True,
        )
    ]  # fmt: skip

    def counts(rec: dict) -> dict:
        return {"correct": rec["correct"], "wrong": rec["wrong"]}

    return {
        "since": LEAGUE_TRACK_FROM,
        "yesterday": counts(pick_record(slate_games, LEAGUE_TRACK_FROM)),
        "total": counts(pick_record(finished, LEAGUE_TRACK_FROM)),
    }


def _validate(name: str, check) -> None:
    try:
        check()
    except ValidationError as exc:
        raise DataValidationError(f"{name} failed validation: {exc}") from exc


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
    data = paths.site_data
    games = _load_games(paths, today, offline)
    validate_games(games, season, today=today)
    _check_not_backwards(
        games, season, read_json(data / "league.json", None), read_json(data / "predictions.json", None)
    )
    players = _load_players(paths, today, offline, games)

    feed = _load_feed(paths, today, players, season)
    parsed, matched, unmatched, as_of = feed or (
        pd.DataFrame(columns=INJURY_COLUMNS), _empty_matched(), [], None
    )

    dtd_log = read_json(data / "dtd_log.json", [])
    _validate("dtd_log.json", lambda: DtdLog.validate_python(dtd_log))
    current = pd.DataFrame(columns=CURRENT_COLUMNS)
    if players is not None:  # without box scores a case could not be resolved honestly
        current = current_player_values(games, players, season)
        rotation_ids = {int(p) for p in current.loc[current["rotation"], "player_id"]}
        dtd_log = update_dtd_log(dtd_log, matched, games, players, today, rotation_ids)
    p_dtd = learn_p_dtd(dtd_log)

    feats = build_features(games, players)
    baselines: dict[str, float] = {}
    if players is not None:
        if feed is not None:  # no feed, no adjustment: incomplete games keep avail 0
            live, baselines = live_avail_and_baselines(games, players, matched, season, today, p_dtd)
            feats = _apply_live_avail(feats, live)
    model = load_or_train(feats, paths, today, force=force_train)
    inputs = prepare_sim_inputs(feats, model, current_team_state(games), season, GAMES_PER_TEAM)
    result = simulate_season(
        inputs, n_sims=n_sims, team_sd=read_team_sd(paths), seed=int(today.strftime("%Y%m%d"))
    )

    # Build every output first; nothing is written until all of them validate.
    generated_at = _now(now)
    model_version = f"{MODEL_NAME}-{model.trained_on}"
    pregame_log = migrate_pregame_log(read_json(data / "pregame_log.json", {}), games, TEAM)
    pregame_log = update_league_pregame_log(
        pregame_log, feats, model, season, generated_at, today.isoformat()
    )
    _validate("pregame_log.json", lambda: PregameLog.validate_python(pregame_log))
    context = team_context(games, season)
    lookup = _injury_lookup(parsed, matched, current, today, p_dtd)

    days = slate_days(feats, model, pregame_log, context, today, baselines, lookup if feed is not None else None)
    slate = {
        "generated_at": generated_at,
        "model_version": model_version,
        "days": days,
        "record": _slate_record(days[0]["games"], games, season, pregame_log),
    }
    league = league_file(
        context, generated_at, model_version, _completed_games(games, season), season_label(season)
    )

    next_pairs = next_game_ids(feats, season)
    next_game = {team: gid for gid, team in next_pairs}
    by_id = feats.set_index("game_id", drop=False)
    team_files: dict[str, dict] = {}
    histories: dict[str, list[dict]] = {}
    for team in TEAM_INDEX:
        gid = next_game.get(team)
        report = _team_injury_report(
            team, by_id.loc[gid] if gid else None, next_pairs, lookup, feed is not None, as_of, p_dtd, unmatched
        )
        team_files[team] = team_file(
            team, feats, model, result, inputs, season, pregame_log, report, generated_at,
            model_version, baselines,
        )  # fmt: skip
        history = read_json(data / "history" / f"{team}.json", None)
        if history is None:  # first launch: Boston's history moves from the retired history.json
            history = read_json(data / "history.json", []) if team == TEAM else []
        histories[team] = upsert_history(history, history_entry(today, team_files[team]["team_summary"]))

    _validate("slate.json", lambda: Slate.model_validate(slate))
    _validate("league.json", lambda: League.model_validate(league))
    for team in TEAM_INDEX:
        _validate(f"teams/{team}.json", lambda t=team: TeamFile.model_validate(team_files[t]))
        _validate(f"history/{team}.json", lambda t=team: [HistoryEntry.model_validate(h) for h in histories[t]])
    _validate("dtd_log.json", lambda: DtdLog.validate_python(dtd_log))

    for team in TEAM_INDEX:
        write_json_atomic(data / "teams" / f"{team}.json", team_files[team], compact=True)
        write_json_atomic(data / "history" / f"{team}.json", histories[team])
    write_json_atomic(data / "slate.json", slate, compact=True)
    write_json_atomic(data / "pregame_log.json", pregame_log)
    if players is not None:
        write_json_atomic(data / "dtd_log.json", dtd_log)
    write_json_atomic(data / "league.json", league, compact=True)
    for retired in ("predictions.json", "history.json"):  # replaced by teams/ and history/
        (data / retired).unlink(missing_ok=True)
    return team_files[TEAM]


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
