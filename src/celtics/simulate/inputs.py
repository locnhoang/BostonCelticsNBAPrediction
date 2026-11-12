"""Everything the season simulator needs, as numpy arrays."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from celtics.features.build import FEATURES, TeamState, matchup_matrix
from celtics.models.margin import MarginModel
from celtics.teams import IS_EAST, TEAM_INDEX, TEAMS


@dataclass
class SimInputs:
    teams: list[str]
    current_wins: np.ndarray
    current_losses: np.ndarray
    total_games: np.ndarray
    h2h_wins: np.ndarray
    conf_wins: np.ndarray
    conf_losses: np.ndarray
    rem_home: np.ndarray
    rem_away: np.ndarray
    rem_margin: np.ndarray
    same_conf_rem: np.ndarray
    extra_games: np.ndarray
    neutral_margin: np.ndarray
    matchup: np.ndarray
    sigma: float


def _idx(series: pd.Series) -> np.ndarray:
    return series.map(TEAM_INDEX).to_numpy(dtype=np.int64)


def prepare_sim_inputs(
    features: pd.DataFrame,
    model: MarginModel,
    state: TeamState,
    season: int,
    target_games: int | None = 82,
) -> SimInputs:
    n = len(TEAMS)
    cur = features[features["season"] == season]
    done = cur[cur["completed"]]
    rem = cur[~cur["completed"]]

    dh, da = _idx(done["home_team"]), _idx(done["away_team"])
    home_won = (done["margin"] > 0).to_numpy()
    w_idx = np.where(home_won, dh, da)
    l_idx = np.where(home_won, da, dh)
    current_wins = np.bincount(w_idx, minlength=n)
    current_losses = np.bincount(l_idx, minlength=n)
    h2h = np.zeros((n, n), dtype=np.int64)
    np.add.at(h2h, (w_idx, l_idx), 1)
    same_done = IS_EAST[dh] == IS_EAST[da]
    conf_wins = np.bincount(w_idx[same_done], minlength=n)
    conf_losses = np.bincount(l_idx[same_done], minlength=n)

    rh, ra = _idx(rem["home_team"]), _idx(rem["away_team"])
    rem_margin = model.predict_margin(rem[FEATURES]) if len(rem) else np.zeros(0)
    scheduled = (
        current_wins + current_losses + np.bincount(rh, minlength=n) + np.bincount(ra, minlength=n)
    )
    if target_games is None:
        extra = np.zeros(n, dtype=np.int64)
    else:
        extra = np.clip(target_games - scheduled, 0, None).astype(np.int64)

    matchup = matchup_matrix(model.predict_margin, state)
    neutral = ((matchup - matchup.T) / 2).sum(axis=1) / (n - 1)

    return SimInputs(
        teams=list(TEAMS),
        current_wins=current_wins,
        current_losses=current_losses,
        total_games=scheduled + extra,
        h2h_wins=h2h,
        conf_wins=conf_wins,
        conf_losses=conf_losses,
        rem_home=rh,
        rem_away=ra,
        rem_margin=np.asarray(rem_margin, dtype=float),
        same_conf_rem=IS_EAST[rh] == IS_EAST[ra],
        extra_games=extra,
        neutral_margin=neutral,
        matchup=matchup,
        sigma=float(model.sigma),
    )
