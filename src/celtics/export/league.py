"""League-wide pregame log: every game's home-side win probability, frozen once its day passes."""

import logging

import pandas as pd

from celtics.models.margin import MarginModel

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
