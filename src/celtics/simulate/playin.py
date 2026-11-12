"""Play-in tournament for seeds 7-10."""

import numpy as np


def play_in(
    seeds_7_to_10: list[int],
    matchup: np.ndarray,
    offsets: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> list[int]:
    s7, s8, s9, s10 = seeds_7_to_10

    def play(home: int, away: int) -> tuple[int, int]:
        margin = matchup[home, away] + offsets[home] - offsets[away] + rng.normal(0.0, sigma)
        return (home, away) if margin > 0 else (away, home)

    seed7, loser78 = play(s7, s8)
    winner910, _ = play(s9, s10)
    seed8, _ = play(loser78, winner910)
    return [seed7, seed8]
