"""Simplified NBA tiebreakers: h2h -> division winner -> conference record -> random."""

import numpy as np

from celtics.teams import DIVISION_MEMBERS


def h2h_pct(team: int, group: list[int], h2h: np.ndarray) -> float:
    others = [o for o in group if o != team]
    wins = int(h2h[team, others].sum())
    losses = int(h2h[others, team].sum())
    return wins / (wins + losses) if wins + losses else 0.5


def division_winners(pct: np.ndarray, h2h: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    winners = np.zeros(len(pct), dtype=bool)
    for members in DIVISION_MEMBERS:
        best = max(pct[t] for t in members)
        tied = [t for t in members if pct[t] == best]
        if len(tied) > 1:
            tied = sorted(tied, key=lambda t, g=tied: (-h2h_pct(t, g, h2h), tiebreak[t]))
        winners[tied[0]] = True
    return winners


def rank_conference(
    members: list[int],
    pct: np.ndarray,
    h2h: np.ndarray,
    conf_pct: np.ndarray,
    div_winner: np.ndarray,
    tiebreak: np.ndarray,
) -> list[int]:
    ordered = sorted(members, key=lambda t: -pct[t])
    result: list[int] = []
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and pct[ordered[j + 1]] == pct[ordered[i]]:
            j += 1
        group = ordered[i : j + 1]
        if len(group) > 1:
            group = sorted(
                group,
                key=lambda t, g=group: (
                    -h2h_pct(t, g, h2h),
                    -int(div_winner[t]),
                    -conf_pct[t],
                    tiebreak[t],
                ),
            )
        result.extend(group)
        i = j + 1
    return result
