"""Margin-of-victory Elo (FiveThirtyEight-style)."""

from collections import defaultdict

import numpy as np
import pandas as pd

from celtics.config import ELO_CARRYOVER, ELO_HOME, ELO_K, ELO_MEAN


def expected_home(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(home_elo + ELO_HOME - away_elo) / 400))


def mov_multiplier(mov: float, winner_elo_edge: float) -> float:
    return ((abs(mov) + 3) ** 0.8) / (7.5 + 0.006 * winner_elo_edge)


def compute_elo(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    ratings: defaultdict[str, float] = defaultdict(lambda: ELO_MEAN)
    ordered = games.sort_values(["date", "game_id"])
    season = None
    out_ids, out_home, out_away = [], [], []
    for row in ordered.itertuples(index=False):
        if row.season != season:
            if season is not None:
                for team in list(ratings):
                    ratings[team] = ELO_CARRYOVER * ratings[team] + (1 - ELO_CARRYOVER) * ELO_MEAN
            season = row.season
        h, a = ratings[row.home_team], ratings[row.away_team]
        out_ids.append(row.game_id)
        out_home.append(h)
        out_away.append(a)
        if np.isnan(row.home_pts):
            continue
        mov = row.home_pts - row.away_pts
        exp = expected_home(h, a)
        home_won = mov > 0
        edge = (h + ELO_HOME - a) if home_won else (a - h - ELO_HOME)
        delta = ELO_K * mov_multiplier(mov, edge) * ((1.0 if home_won else 0.0) - exp)
        ratings[row.home_team] = h + delta
        ratings[row.away_team] = a - delta
    pre = pd.DataFrame({"game_id": out_ids, "home_elo_pre": out_home, "away_elo_pre": out_away})
    return pre, dict(ratings)
