"""Turn simulation output into website JSON."""

import json
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from celtics.evaluate.metrics import summarize
from celtics.injuries import normalize_name, p_out
from celtics.models.explain import explain
from celtics.models.margin import MarginModel
from celtics.simulate.inputs import SimInputs
from celtics.simulate.season import SimResult
from celtics.teams import CONFERENCE_OF, TEAM_INDEX


def r4(x) -> float:
    return round(float(x), 4)


def team_summary(res: SimResult, team: str, wins: int, losses: int) -> dict:
    i = TEAM_INDEX[team]
    w = res.wins[:, i]
    seeds = res.seed[:, i]
    p10, p25, p50, p75, p90 = np.percentile(w, [10, 25, 50, 75, 90])
    counts = np.bincount(w) / len(w)
    return {
        "team": team,
        "wins": int(wins),
        "losses": int(losses),
        "proj_wins_median": float(p50),
        "proj_wins_p10": float(p10),
        "proj_wins_p25": float(p25),
        "proj_wins_p75": float(p75),
        "proj_wins_p90": float(p90),
        "p_50_wins": r4((w >= 50).mean()),
        "p_playoffs": r4(res.made_playoffs[:, i].mean()),
        "p_top6": r4((seeds <= 6).mean()),
        "p_playin": r4(((seeds >= 7) & (seeds <= 10)).mean()),
        "p_miss": r4((seeds >= 11).mean()),
        "p_seed1": r4((seeds == 1).mean()),
        "win_distribution": {str(k): r4(v) for k, v in enumerate(counts) if v > 0},
    }


def seed_probabilities(res: SimResult, team: str) -> dict[str, float]:
    seeds = res.seed[:, TEAM_INDEX[team]]
    return {str(k): r4((seeds == k).mean()) for k in range(1, 16)}


def conference_table(res: SimResult, inp: SimInputs, conference: str) -> list[dict]:
    rows = []
    for team, i in TEAM_INDEX.items():
        if CONFERENCE_OF[team] != conference:
            continue
        rows.append(
            {
                "team": team,
                "wins": int(inp.current_wins[i]),
                "losses": int(inp.current_losses[i]),
                "proj_wins": round(float(res.wins[:, i].mean()), 1),
                "p_playoffs": r4(res.made_playoffs[:, i].mean()),
                "p_top6": r4((res.seed[:, i] <= 6).mean()),
                "p_seed1": r4((res.seed[:, i] == 1).mean()),
            }
        )
    return sorted(rows, key=lambda r: -r["proj_wins"])


def game_rows(
    features: pd.DataFrame, model: MarginModel, season: int, team: str, pregame_log: dict
) -> list[dict]:
    cur = features[
        (features["season"] == season)
        & ((features["home_team"] == team) | (features["away_team"] == team))
    ].sort_values(["date", "game_id"])
    if cur.empty:
        return []
    margins = model.predict_margin(cur)
    probs = model.win_prob(cur)
    parts = explain(model, cur)
    rows = []
    for idx, row, m, p in zip(cur.index, cur.itertuples(index=False), margins, probs, strict=True):
        home = row.home_team == team
        out = {
            "game_id": row.game_id,
            "date": row.date.date().isoformat(),
            "opponent": row.away_team if home else row.home_team,
            "home": bool(home),
            "rest_days": float(row.home_rest if home else row.away_rest),
            "opp_rest_days": float(row.away_rest if home else row.home_rest),
            "b2b": bool(row.home_b2b if home else row.away_b2b),
            "completed": bool(row.completed),
            "win_prob": None,
            "pred_margin": None,
            "result": None,
            "pregame_win_prob": None,
            "drivers": None,
            "win_prob_healthy": None,
        }
        if row.completed:
            mine, theirs = (row.home_pts, row.away_pts) if home else (row.away_pts, row.home_pts)
            out["result"] = f"{'W' if mine > theirs else 'L'} {int(mine)}-{int(theirs)}"
            logged = pregame_log.get(row.game_id)
            out["pregame_win_prob"] = logged["win_prob"] if logged else None
        else:
            out["win_prob"] = r4(p if home else 1 - p)
            out["pred_margin"] = round(float(m if home else -m), 1)
            if parts is not None:
                sign = 1.0 if home else -1.0
                out["drivers"] = [
                    {"label": label, "value": round(sign * float(parts.at[idx, label]), 1)}
                    for label in parts.columns
                ]
        rows.append(out)
    return rows


def update_pregame_log(log: dict, rows: list[dict], now_iso: str, today: str | None = None) -> dict:
    """`today` is the Eastern game date (ISO); defaults to the UTC date of `now_iso`."""
    updated = dict(log)
    today = today or now_iso[:10]
    for r in rows:
        # Keep the latest pre-game prediction for each game. Once a game's date has passed
        # its entry is frozen, even if a short game log makes it look unplayed again.
        if r["completed"] or r["date"] < today:
            continue
        updated[r["game_id"]] = {
            "win_prob": r["win_prob"],
            "pred_margin": r["pred_margin"],
            "logged_at": now_iso,
        }
    return updated


def live_metrics(rows: list[dict]) -> dict | None:
    scored = [r for r in rows if r["completed"] and r["pregame_win_prob"] is not None]
    if not scored:
        return None
    p = [r["pregame_win_prob"] for r in scored]
    y = [1.0 if r["result"].startswith("W") else 0.0 for r in scored]
    s = summarize(p, y)
    return {"n": s["n"], "log_loss": r4(s["log_loss"]), "brier": r4(s["brier"]), "accuracy": r4(s["accuracy"])}


def history_entry(day: date, summary: dict) -> dict:
    return {
        "date": day.isoformat(),
        "record": f"{summary['wins']}-{summary['losses']}",
        "projected_wins_median": summary["proj_wins_median"],
        "p10": summary["proj_wins_p10"],
        "p90": summary["proj_wins_p90"],
        "playoff_prob": summary["p_playoffs"],
        "top6_prob": summary["p_top6"],
        "seed1_prob": summary["p_seed1"],
    }


def upsert_history(history: list[dict], entry: dict) -> list[dict]:
    kept = [h for h in history if h["date"] != entry["date"]]
    return sorted([*kept, entry], key=lambda h: h["date"])


def read_json(path: Path, default):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def healthy_win_prob(
    features: pd.DataFrame,
    model: MarginModel,
    game_id: str,
    team: str,
    baselines: dict[str, float],
) -> float:
    """`team`'s win probability with nobody missing on either side.

    avail = baseline - missing, so missing = 0 means avail equals each team's baseline (the
    same baselines the live prediction uses; 0 for a team without one).
    """
    row = features[features["game_id"] == game_id].copy()
    home, away = row["home_team"].iloc[0], row["away_team"].iloc[0]
    row["home_avail"] = baselines.get(home, 0.0)
    row["away_avail"] = baselines.get(away, 0.0)
    p = float(model.win_prob(row)[0])
    return r4(p if row["home_team"].iloc[0] == team else 1 - p)


def injury_teams(
    teams: list[str],
    parsed: pd.DataFrame,
    matched: pd.DataFrame,
    current: pd.DataFrame,
    rosters: dict[str, pd.DataFrame | None],
    today: date,
    game_date: date,
    p_dtd: float,
) -> dict[str, list[dict]]:
    """Per-team player list for the display: roster (or recent players) plus every injured player.

    Players the matcher could not place keep player_id None with zero value and minutes.
    """
    ids = {(r.team, r.name): int(r.player_id) for r in matched.itertuples()}
    stats = {int(r.player_id): r for r in current.itertuples()}
    out: dict[str, list[dict]] = {}
    for team in teams:
        injured = {
            r.name: r for r in parsed[parsed["team"] == team].drop_duplicates("name").itertuples()
        }
        pool: dict[int, str] = {}
        roster = rosters.get(team)
        if roster is not None:
            pool = dict(zip(roster["player_id"].astype(int), roster["player_name"], strict=True))
        else:
            mine = current[current["team"] == team]
            pool = dict(zip(mine["player_id"].astype(int), mine["player_name"], strict=True))
        by_norm = {normalize_name(n): pid for pid, n in pool.items()}
        injured_ids = {
            ids.get((team, n), by_norm.get(normalize_name(n))) for n in injured
        } - {None}
        players = [(pid, name, None) for pid, name in pool.items() if pid not in injured_ids]
        for name, inj in injured.items():
            players.append((ids.get((team, name), by_norm.get(normalize_name(name))), name, inj))
        rows = []
        for pid, name, inj in players:
            s = stats.get(pid) if pid is not None else None
            rd = inj.return_date if inj is not None else None
            status = inj.status if inj is not None else "Available"
            rows.append(
                {
                    "player_id": pid,
                    "name": name,
                    "status": status,
                    "comment": inj.comment if inj is not None else "",
                    "return_date": rd.isoformat() if pd.notna(rd) else None,
                    "p_out": r4(p_out(status, rd, game_date, today, p_dtd)),
                    "expected_minutes": round(float(s.expected_minutes), 1) if s else 0.0,
                    "game_value": round(float(s.game_value), 2) if s else 0.0,
                    "rotation": bool(s.rotation) if s else False,
                }
            )
        rows.sort(key=lambda r: (not r["rotation"], -r["expected_minutes"], r["name"]))
        out[team] = rows
    return out
