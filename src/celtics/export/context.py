"""Current-season team context: standings position, recent form, efficiency ratings."""

import pandas as pd

from celtics.teams import CONFERENCE_OF, TEAMS


def _team_lines(games: pd.DataFrame, season: int) -> pd.DataFrame:
    done = games[(games["season"] == season) & games["home_pts"].notna()]
    home = pd.DataFrame(
        {"team": done["home_team"], "date": done["date"], "game_id": done["game_id"],
         "pf": done["home_pts"], "pa": done["away_pts"], "poss": done["poss"]}
    )
    away = pd.DataFrame(
        {"team": done["away_team"], "date": done["date"], "game_id": done["game_id"],
         "pf": done["away_pts"], "pa": done["home_pts"], "poss": done["poss"]}
    )
    return pd.concat([home, away], ignore_index=True).sort_values(["date", "game_id"])


def _streak(won: list[bool]) -> str:
    if not won:
        return ""
    k = 1
    while k < len(won) and won[-k - 1] == won[-1]:
        k += 1
    return f"{'W' if won[-1] else 'L'}{k}"


def _rank(raw: dict[str, dict], key: str, higher_is_better: bool) -> dict[str, int]:
    """League rank 1..30; teams with no games go last, ties broken by abbreviation."""

    def sort_key(t: str):
        played = raw[t]["games"] > 0
        value = raw[t][key]
        return (not played, -value if higher_is_better else value, t)

    return {t: i + 1 for i, t in enumerate(sorted(raw, key=sort_key))}


def team_context(games: pd.DataFrame, season: int) -> dict[str, dict]:
    lines = _team_lines(games, season)
    raw: dict[str, dict] = {}
    for team in TEAMS:
        g = lines[lines["team"] == team]
        won = (g["pf"] > g["pa"]).tolist()
        poss = float(g["poss"].sum())
        off = 100.0 * float(g["pf"].sum()) / poss if poss else 0.0
        de = 100.0 * float(g["pa"].sum()) / poss if poss else 0.0
        last = won[-10:]
        raw[team] = {
            "games": len(won),
            "wins": sum(won),
            "losses": len(won) - sum(won),
            "last10": f"{sum(last)}-{len(last) - sum(last)}",
            "streak": _streak(won),
            "off": off,
            "def": de,
            "net": off - de,
        }

    off_rank = _rank(raw, "off", higher_is_better=True)
    def_rank = _rank(raw, "def", higher_is_better=False)
    net_rank = _rank(raw, "net", higher_is_better=True)

    conf_rank: dict[str, int] = {}
    games_back: dict[str, float] = {}
    for conf in ("East", "West"):
        members = [t for t in TEAMS if CONFERENCE_OF[t] == conf]

        def standing(t: str):
            r = raw[t]
            pct = r["wins"] / r["games"] if r["games"] else 0.0
            return (r["games"] == 0, -pct, -r["wins"], t)

        order = sorted(members, key=standing)
        lw, ll = raw[order[0]]["wins"], raw[order[0]]["losses"]
        for i, t in enumerate(order):
            conf_rank[t] = i + 1
            games_back[t] = ((lw - raw[t]["wins"]) + (raw[t]["losses"] - ll)) / 2

    return {
        t: {
            "wins": raw[t]["wins"],
            "losses": raw[t]["losses"],
            "conf_rank": conf_rank[t],
            "games_back": games_back[t],
            "last10": raw[t]["last10"],
            "streak": raw[t]["streak"],
            "off_rating": round(raw[t]["off"], 1),
            "def_rating": round(raw[t]["def"], 1),
            "net_rating": round(raw[t]["net"], 1),
            "off_rank": off_rank[t],
            "def_rank": def_rank[t],
            "net_rank": net_rank[t],
        }
        for t in TEAMS
    }
