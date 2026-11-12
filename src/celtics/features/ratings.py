"""Exponentially decayed net rating with a preseason prior, plus rest features."""

from collections import defaultdict

import numpy as np
import pandas as pd

from celtics.config import NET_DECAY, NET_PRIOR_SHRINK, NET_PRIOR_WEIGHT, REST_CAP


def _estimate(state: dict[str, float]) -> float:
    return state["num"] / state["den"] if state["den"] > 0 else 0.0


def compute_net_ratings(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    ordered = games.sort_values(["date", "game_id"])
    season = None
    state: dict[str, dict[str, float]] = {}
    totals: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    prev_avg: dict[str, float] = {}
    ids, home_pre, away_pre = [], [], []
    for row in ordered.itertuples(index=False):
        if row.season != season:
            prev_avg = {t: s / n for t, (s, n) in totals.items() if n > 0}
            totals = defaultdict(lambda: [0.0, 0.0])
            state = {}
            season = row.season
        for team in (row.home_team, row.away_team):
            if team not in state:
                prior = NET_PRIOR_SHRINK * prev_avg.get(team, 0.0)
                state[team] = {"num": NET_PRIOR_WEIGHT * prior, "den": NET_PRIOR_WEIGHT}
        ids.append(row.game_id)
        home_pre.append(_estimate(state[row.home_team]))
        away_pre.append(_estimate(state[row.away_team]))
        if np.isnan(row.home_pts):
            continue
        x = 100.0 * (row.home_pts - row.away_pts) / row.poss
        for team, value in ((row.home_team, x), (row.away_team, -x)):
            s = state[team]
            s["num"] = NET_DECAY * s["num"] + value
            s["den"] = NET_DECAY * s["den"] + 1.0
            totals[team][0] += value
            totals[team][1] += 1.0
    pre = pd.DataFrame({"game_id": ids, "home_net_pre": home_pre, "away_net_pre": away_pre})
    return pre, {t: _estimate(s) for t, s in state.items()}


def compute_rest(games: pd.DataFrame) -> pd.DataFrame:
    base = games[["game_id", "season", "date"]]
    long = pd.concat(
        [
            base.assign(team=games["home_team"], side="home"),
            base.assign(team=games["away_team"], side="away"),
        ],
        ignore_index=True,
    ).sort_values(["team", "date", "game_id"])
    grouped = long.groupby(["season", "team"])["date"]
    prev1, prev2 = grouped.shift(1), grouped.shift(2)
    long["rest"] = (long["date"] - prev1).dt.days.clip(upper=REST_CAP).fillna(REST_CAP).astype(float)
    long["b2b"] = (long["rest"] == 1).astype(float)
    long["three_in_four"] = ((long["date"] - prev2).dt.days <= 3).astype(float)
    wide = long.pivot(index="game_id", columns="side", values=["rest", "b2b", "three_in_four"])
    return pd.DataFrame(
        {
            "game_id": wide.index,
            "home_rest": wide[("rest", "home")].to_numpy(),
            "away_rest": wide[("rest", "away")].to_numpy(),
            "home_b2b": wide[("b2b", "home")].to_numpy(),
            "away_b2b": wide[("b2b", "away")].to_numpy(),
            "home_3in4": wide[("three_in_four", "home")].to_numpy(),
            "away_3in4": wide[("three_in_four", "away")].to_numpy(),
        }
    ).reset_index(drop=True)
