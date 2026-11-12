import numpy as np

from celtics.simulate.playin import play_in
from celtics.simulate.standings import division_winners, h2h_pct, rank_conference
from celtics.teams import DIVISION_MEMBERS


def test_h2h_pct():
    h2h = np.zeros((30, 30), dtype=int)
    h2h[0, 1] = 3
    h2h[1, 0] = 1
    assert h2h_pct(0, [0, 1], h2h) == 0.75
    assert h2h_pct(2, [2, 3], h2h) == 0.5


def test_rank_by_pct_then_h2h_then_division_then_conf():
    members = [0, 1, 2, 3]
    pct = np.zeros(30)
    pct[[0, 1, 2, 3]] = [0.6, 0.6, 0.5, 0.5]
    h2h = np.zeros((30, 30), dtype=int)
    h2h[1, 0] = 2  # team 1 owns team 0
    conf_pct = np.zeros(30)
    conf_pct[3] = 0.9  # team 3 has better conference record
    div = np.zeros(30, dtype=bool)
    tiebreak = np.zeros(30)
    order = rank_conference(members, pct, h2h, conf_pct, div, tiebreak)
    assert order == [1, 0, 3, 2]

    div[2] = True  # division winner beats conference record
    assert rank_conference(members, pct, h2h, conf_pct, div, tiebreak) == [1, 0, 2, 3]


def test_random_tiebreak_is_last_resort():
    pct = np.full(30, 0.5)
    tiebreak = np.zeros(30)
    tiebreak[[0, 1]] = [0.9, 0.1]
    order = rank_conference(
        [0, 1], pct, np.zeros((30, 30), int), np.zeros(30), np.zeros(30, bool), tiebreak
    )
    assert order == [1, 0]


def test_division_winners_one_per_division():
    rng = np.random.default_rng(0)
    pct = np.round(rng.random(30), 3)
    winners = division_winners(pct, np.zeros((30, 30), int), rng.random(30))
    assert winners.sum() == 6


def test_play_in_bracket():
    rng = np.random.default_rng(0)
    matchup = np.zeros((30, 30))
    off = np.zeros(30)
    off[[7, 9]] = 100  # teams 7 and 9 always win
    seeds = [6, 7, 8, 9]  # team ids holding seeds 7, 8, 9, 10
    # 7v8: team 7 beats 6 -> seed 7 = team 7; 9v10: team 9 beats 8; 6 vs 9: team 9 wins
    assert play_in(seeds, matchup, off, 1.0, rng) == [7, 9]


def test_division_winner_breaks_tie_by_h2h():
    pct = np.zeros(30)
    a, b = DIVISION_MEMBERS[0][0], DIVISION_MEMBERS[0][1]
    pct[[a, b]] = 0.6
    h2h = np.zeros((30, 30), dtype=int)
    h2h[b, a] = 2  # b owns a
    winners = division_winners(pct, h2h, np.zeros(30))
    assert winners[b] and not winners[a]
