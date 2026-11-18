"""League-wide pregame log: every game's home-side win probability, frozen once its day passes."""

import logging
from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd

from celtics.config import LEAGUE_TRACK_FROM, TRACK_PICKS_FROM, season_label
from celtics.export.build import (
    conference_table,
    game_rows,
    healthy_win_prob,
    r4,
    seed_probabilities,
    team_summary,
)
from celtics.models.explain import explain
from celtics.models.margin import MarginModel
from celtics.simulate.inputs import SimInputs
from celtics.simulate.season import SimResult
from celtics.teams import CONFERENCE_OF, TEAM_INDEX

log = logging.getLogger(__name__)


def migrate_pregame_log(log_: dict, games: pd.DataFrame, team: str = "BOS") -> dict:
    """Convert old entries (`win_prob` for `team`) to the home perspective; new ones pass through."""
    home_of = dict(zip(games["game_id"], games["home_team"], strict=True))
    out = {}
    for gid, e in log_.items():
        if "win_prob" not in e:
            out[gid] = e
        elif gid not in home_of:
            log.warning("dropping old pregame entry for unknown game %s", gid)
        else:
            home = home_of[gid] == team
            margin = e["pred_margin"]
            out[gid] = {
                "home_win_prob": e["win_prob"] if home else round(1 - e["win_prob"], 4),
                "pred_margin": margin if home or margin is None else -margin,
                "logged_at": e["logged_at"],
            }
    return out


def update_league_pregame_log(
    log_: dict, feats: pd.DataFrame, model: MarginModel, season: int, now_iso: str, today: str
) -> dict:
    """Refresh every upcoming game dated `today` or later; past-dated entries stay frozen."""
    updated = dict(log_)
    upcoming = feats[
        (feats["season"] == season)
        & ~feats["completed"]
        & (feats["date"].dt.strftime("%Y-%m-%d") >= today)
    ]
    if upcoming.empty:
        return updated
    probs = model.win_prob(upcoming)
    margins = model.predict_margin(upcoming)
    for gid, p, m in zip(upcoming["game_id"], probs, margins, strict=True):
        updated[gid] = {
            "home_win_prob": round(float(p), 4),
            "pred_margin": round(float(m), 1),
            "logged_at": now_iso,
        }
    return updated


def pick(home: str, away: str, p_home: float) -> str:
    """The model's pick: the home team at 0.5 or better, else the away team."""
    return home if p_home >= 0.5 else away


def pick_record(games: list[dict], since: str, team: str | None = None) -> dict:
    """Correct and wrong picks over completed slate games with a logged pick, dated `since` or later."""
    correct = wrong = 0
    for g in games:
        if not g["completed"] or g["pregame_home_win_prob"] is None or g["date"] < since:
            continue
        if team is not None and team not in (g["home"], g["away"]):
            continue
        home_won = g["home_pts"] > g["away_pts"]
        if (g["pregame_home_win_prob"] >= 0.5) == home_won:
            correct += 1
        else:
            wrong += 1
    return {"since": since, "correct": correct, "wrong": wrong}


def _next_game_ids(feats: pd.DataFrame, season: int) -> set[tuple[str, str]]:
    """(game_id, team) pairs where the game is that team's next incomplete game."""
    left = feats[(feats["season"] == season) & ~feats["completed"]].sort_values(["date", "game_id"])
    seen: set[str] = set()
    ids: set[tuple[str, str]] = set()
    for gid, home, away in zip(left["game_id"], left["home_team"], left["away_team"], strict=True):
        for t in (home, away):
            if t not in seen:
                seen.add(t)
                ids.add((gid, t))
    return ids


def slate_days(
    feats: pd.DataFrame,
    model: MarginModel,
    pregame_log: dict,
    team_context: dict[str, dict],
    today: date,
    baselines: dict[str, float],
    injuries: Callable[[str, str, date, frozenset[str]], dict[str, list[dict]]] | None,
) -> list[dict]:
    """Yesterday, today and tomorrow, each with every game on that date."""
    season = int(feats["season"].max())
    next_pairs = _next_game_ids(feats, season)
    dates = feats["date"].dt.date
    days = []
    for k in (-1, 0, 1):
        day = today + timedelta(days=k)
        rows = feats[dates == day].sort_values("game_id")
        games = []
        if not rows.empty:
            probs = model.win_prob(rows)
            margins = model.predict_margin(rows)
            parts = explain(model, rows)
        for i, row in enumerate(rows.itertuples()):
            home, away = row.home_team, row.away_team
            game = {
                "game_id": row.game_id,
                "date": day.isoformat(),
                "home": home,
                "away": away,
                "home_record": f"{team_context[home]['wins']}-{team_context[home]['losses']}",
                "away_record": f"{team_context[away]['wins']}-{team_context[away]['losses']}",
                "completed": bool(row.completed),
                "home_win_prob": None,
                "home_win_prob_healthy": None,
                "pred_margin": None,
                "pick": None,
                "home_pts": None,
                "away_pts": None,
                "pregame_home_win_prob": None,
                "correct": None,
                "drivers": None,
                "injuries": None,
            }
            logged = pregame_log.get(row.game_id)
            logged_p = logged["home_win_prob"] if logged else None
            if row.completed:
                game["home_pts"], game["away_pts"] = int(row.home_pts), int(row.away_pts)
                if logged_p is not None:
                    game["pregame_home_win_prob"] = logged_p
                    game["pick"] = pick(home, away, logged_p)
                    game["correct"] = (game["pick"] == home) == (row.home_pts > row.away_pts)
            else:
                p = r4(probs[i])
                game["home_win_prob"] = p
                game["home_win_prob_healthy"] = healthy_win_prob(feats, model, row.game_id, home, baselines)
                game["pred_margin"] = round(float(margins[i]), 1)
                game["pick"] = pick(home, away, p)
                if parts is not None:
                    game["drivers"] = [
                        {"label": label, "value": round(float(parts.iloc[i][label]), 1)}
                        for label in parts.columns
                    ]
                game["injuries"] = {"home": [], "away": []}
                if injuries is not None:
                    not_next = frozenset(t for t in (home, away) if (row.game_id, t) not in next_pairs)
                    by_team = injuries(home, away, day, not_next)
                    game["injuries"] = {"home": by_team.get(home, []), "away": by_team.get(away, [])}
            games.append(game)
        days.append({"date": day.isoformat(), "games": games})
    return days


def record_since(team: str) -> str:
    """The date a team's pick record starts counting."""
    return TRACK_PICKS_FROM if team == "BOS" else LEAGUE_TRACK_FROM


def record_from_rows(rows: list[dict], since: str) -> dict:
    """Pick record from a team's game rows: their `correct` flags on games dated `since` or later."""
    flags = [r["correct"] for r in rows if r["correct"] is not None and r["date"] >= since]
    correct = sum(flags)
    return {"since": since, "correct": correct, "wrong": len(flags) - correct}


def team_file(
    team: str,
    feats: pd.DataFrame,
    model: MarginModel,
    result: SimResult,
    inputs: SimInputs,
    season: int,
    pregame_log: dict,
    injury_report: dict,
    generated_at: str,
    model_version: str,
    baselines: dict[str, float] | None = None,
) -> dict:
    """One team's page data: summary, schedule with picks, seeds, conference and injuries."""
    rows = game_rows(feats, model, season, team, pregame_log)
    upcoming = next((r for r in rows if not r["completed"]), None)
    for r in rows:
        if r is not upcoming:
            r["drivers"] = None
    if upcoming is not None:
        upcoming["win_prob_healthy"] = healthy_win_prob(
            feats, model, upcoming["game_id"], team, baselines or {}
        )
    i = TEAM_INDEX[team]
    return {
        "generated_at": generated_at,
        "model_version": model_version,
        "season": season_label(season),
        "team_summary": team_summary(result, team, inputs.current_wins[i], inputs.current_losses[i]),
        "games": rows,
        "seed_probabilities": seed_probabilities(result, team),
        "conference_table": conference_table(result, inputs, CONFERENCE_OF[team]),
        "metrics": {"record": record_from_rows(rows, record_since(team))},
        "injury_report": injury_report,
    }


def league_file(
    team_context: dict[str, dict],
    generated_at: str,
    model_version: str,
    completed_games: int,
    season: str,
) -> dict:
    return {
        "generated_at": generated_at,
        "model_version": model_version,
        "season": season,
        "completed_games": completed_games,
        "team_context": team_context,
    }
