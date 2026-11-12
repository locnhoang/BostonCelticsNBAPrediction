"""Synthetic full-league data for integration-style tests (no network)."""

import numpy as np
import pandas as pd

from celtics.data.parse import GAME_COLUMNS
from celtics.teams import TEAMS


def make_league(seasons, current=None, completed_rounds=15, seed=0):
    rng = np.random.default_rng(seed)
    strength = {t: rng.normal(0, 5) for t in TEAMS}
    rows = []
    for season in seasons:
        start = pd.Timestamp(f"{season}-10-20")
        n = 0
        for rnd in range(82):
            order = rng.permutation(TEAMS)
            played = season != current or rnd < completed_rounds
            for k in range(15):
                home, away = order[2 * k], order[2 * k + 1]
                margin = strength[home] - strength[away] + 2.5 + rng.normal(0, 12)
                hp = round(110 + margin / 2)
                ap = round(110 - margin / 2)
                if hp == ap:
                    hp += 1
                rows.append(
                    {
                        "game_id": f"002{season % 100:02d}{n:05d}",
                        "season": season,
                        "date": start + pd.Timedelta(days=2 * rnd),
                        "home_team": home,
                        "away_team": away,
                        "home_pts": float(hp) if played else np.nan,
                        "away_pts": float(ap) if played else np.nan,
                        "poss": 100.0 if played else np.nan,
                    }
                )
                n += 1
    return pd.DataFrame(rows, columns=GAME_COLUMNS)
