"""Static team metadata. Abbreviations are stable from 2014-15 onward."""

import numpy as np

DIVISIONS: dict[str, list[str]] = {
    "Atlantic": ["BOS", "BKN", "NYK", "PHI", "TOR"],
    "Central": ["CHI", "CLE", "DET", "IND", "MIL"],
    "Southeast": ["ATL", "CHA", "MIA", "ORL", "WAS"],
    "Northwest": ["DEN", "MIN", "OKC", "POR", "UTA"],
    "Pacific": ["GSW", "LAC", "LAL", "PHX", "SAC"],
    "Southwest": ["DAL", "HOU", "MEM", "NOP", "SAS"],
}
EAST_DIVISIONS = ("Atlantic", "Central", "Southeast")

TEAMS: list[str] = sorted(t for members in DIVISIONS.values() for t in members)
TEAM_INDEX: dict[str, int] = {t: i for i, t in enumerate(TEAMS)}
DIVISION_OF: dict[str, str] = {t: d for d, members in DIVISIONS.items() for t in members}
CONFERENCE_OF: dict[str, str] = {
    t: ("East" if d in EAST_DIVISIONS else "West") for t, d in DIVISION_OF.items()
}
IS_EAST = np.array([CONFERENCE_OF[t] == "East" for t in TEAMS])
CONFERENCE_MEMBERS: list[list[int]] = [
    [TEAM_INDEX[t] for t in TEAMS if CONFERENCE_OF[t] == conf] for conf in ("East", "West")
]
DIVISION_MEMBERS: list[list[int]] = [
    [TEAM_INDEX[t] for t in members] for members in DIVISIONS.values()
]
