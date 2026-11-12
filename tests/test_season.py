import numpy as np
import pytest

from celtics.simulate.inputs import SimInputs
from celtics.simulate.season import simulate_season
from celtics.teams import CONFERENCE_MEMBERS, IS_EAST, TEAMS


def even_inputs(rounds=82, margin=0.0, favorite=None, seed=0):
    rng = np.random.default_rng(seed)
    home, away = [], []
    for _ in range(rounds):
        order = rng.permutation(30)
        home += list(order[0::2])
        away += list(order[1::2])
    home, away = np.array(home), np.array(away)
    m = np.full(len(home), margin, dtype=float)
    if favorite is not None:
        m = np.where(home == favorite, 50.0, np.where(away == favorite, -50.0, m))
    zeros = np.zeros(30, dtype=np.int64)
    return SimInputs(
        teams=list(TEAMS),
        current_wins=zeros.copy(),
        current_losses=zeros.copy(),
        total_games=np.full(30, rounds),
        h2h_wins=np.zeros((30, 30), dtype=np.int64),
        conf_wins=zeros.copy(),
        conf_losses=zeros.copy(),
        rem_home=home,
        rem_away=away,
        rem_margin=m,
        same_conf_rem=IS_EAST[home] == IS_EAST[away],
        extra_games=zeros.copy(),
        neutral_margin=np.zeros(30),
        matchup=np.zeros((30, 30)),
        sigma=12.0,
    )


def test_equal_teams_average_41_wins():
    res = simulate_season(even_inputs(), n_sims=300, team_sd=0.0, seed=1)
    assert res.wins.mean() == pytest.approx(41, abs=0.5)


def test_wins_equal_losses_league_wide():
    inp = even_inputs()
    res = simulate_season(inp, n_sims=100, team_sd=2.0, seed=2)
    assert np.all(res.wins.sum(axis=1) == 15 * 82)


def test_seeds_and_playoff_counts():
    res = simulate_season(even_inputs(), n_sims=100, team_sd=2.0, seed=3)
    for members in CONFERENCE_MEMBERS:
        seeds = np.sort(res.seed[:, members], axis=1)
        assert np.all(seeds == np.arange(1, 16))
        assert np.all(res.made_playoffs[:, members].sum(axis=1) == 8)


def test_dominant_team_is_top_seed():
    res = simulate_season(even_inputs(favorite=0), n_sims=100, team_sd=0.0, seed=4)
    assert res.wins[:, 0].min() >= 80
    assert np.all(res.seed[:, 0] == 1)


def test_extra_games_add_wins():
    inp = even_inputs(rounds=80)
    inp.extra_games = np.full(30, 2)
    inp.total_games = np.full(30, 82)
    res = simulate_season(inp, n_sims=200, team_sd=0.0, seed=5)
    assert res.wins.mean() == pytest.approx(41, abs=0.5)
    assert res.wins.max() <= 82


def test_deterministic_with_seed():
    a = simulate_season(even_inputs(), n_sims=50, seed=9)
    b = simulate_season(even_inputs(), n_sims=50, seed=9)
    assert np.array_equal(a.wins, b.wins) and np.array_equal(a.seed, b.seed)
