"""Monte Carlo simulation of the remaining league schedule."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from celtics.config import DEFAULT_TEAM_SD, N_SIMS
from celtics.simulate.inputs import SimInputs
from celtics.simulate.playin import play_in
from celtics.simulate.standings import division_winners, rank_conference
from celtics.teams import CONFERENCE_MEMBERS


@dataclass
class SimResult:
    """Per-simulation outcomes, one row per simulation and one column per team (TEAMS order).

    `seed` is the regular-season seed (1-15) before the play-in; `made_playoffs` accounts for
    the play-in. Placeholder (extra) games for unscheduled Cup games add to wins/losses but do
    not count toward conference record.
    """

    teams: list[str]
    wins: np.ndarray
    seed: np.ndarray
    made_playoffs: np.ndarray


def _count(idx: np.ndarray, n_sims: int, size: int) -> np.ndarray:
    """Per-simulation counts of each value in idx (shape (n_sims, k)) -> (n_sims, size)."""
    shifted = idx + (np.arange(n_sims)[:, None] * size)
    return np.bincount(shifted.ravel(), minlength=n_sims * size).reshape(n_sims, size)


def simulate_season(
    inp: SimInputs, n_sims: int = N_SIMS, team_sd: float = DEFAULT_TEAM_SD, seed: int = 0
) -> SimResult:
    rng = np.random.default_rng(seed)
    n = len(inp.teams)
    off = rng.normal(0.0, team_sd, size=(n_sims, n)) if team_sd > 0 else np.zeros((n_sims, n))

    noise = rng.normal(0.0, inp.sigma, size=(n_sims, len(inp.rem_home)))
    margin = inp.rem_margin[None, :] + off[:, inp.rem_home] - off[:, inp.rem_away] + noise
    home_win = margin > 0
    winners = np.where(home_win, inp.rem_home[None, :], inp.rem_away[None, :])
    losers = np.where(home_win, inp.rem_away[None, :], inp.rem_home[None, :])

    wins = inp.current_wins[None, :] + _count(winners, n_sims, n)
    if inp.extra_games.any():
        p = norm.cdf((inp.neutral_margin[None, :] + off) / inp.sigma)
        wins = wins + rng.binomial(inp.extra_games[None, :], p)
    losses = inp.total_games[None, :] - wins

    h2h = _count(winners * n + losers, n_sims, n * n).reshape(n_sims, n, n) + inp.h2h_wins[None]
    same = inp.same_conf_rem
    conf_w = inp.conf_wins[None, :] + _count(winners[:, same], n_sims, n)
    conf_l = inp.conf_losses[None, :] + _count(losers[:, same], n_sims, n)
    tiebreak = rng.random((n_sims, n))

    seeds = np.zeros((n_sims, n), dtype=np.int64)
    made = np.zeros((n_sims, n), dtype=bool)
    for s in range(n_sims):
        played = np.maximum(wins[s] + losses[s], 1)
        pct = np.round(wins[s] / played, 9)
        conf_games = conf_w[s] + conf_l[s]
        conf_pct = np.where(conf_games > 0, conf_w[s] / np.maximum(conf_games, 1), 0.5)
        div_winner = division_winners(pct, h2h[s], tiebreak[s])
        for members in CONFERENCE_MEMBERS:
            order = rank_conference(members, pct, h2h[s], conf_pct, div_winner, tiebreak[s])
            seeds[s, order] = np.arange(1, len(order) + 1)
            made[s, order[:6]] = True
            made[s, play_in(order[6:10], inp.matchup, off[s], inp.sigma, rng)] = True

    return SimResult(teams=list(inp.teams), wins=wins.astype(np.int64), seed=seeds, made_playoffs=made)
