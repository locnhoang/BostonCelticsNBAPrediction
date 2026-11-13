"""ESPN injury feed: parse into a frame and match injured players to their stats."""

import re
import unicodedata
from datetime import date

import pandas as pd

ESPN_TEAM: dict[str, str] = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

INJURY_COLUMNS = ["team", "name", "status", "comment", "return_date", "reported_at"]
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def normalize_name(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    words = re.sub(r"[^a-z0-9\s]", "", folded.lower()).split()
    return " ".join(w for w in words if w not in _SUFFIXES)


def _return_date(entry: dict) -> date | None:
    raw = (entry.get("details") or {}).get("returnDate")
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def parse_injuries(raw: dict) -> pd.DataFrame:
    rows = []
    for team_block in raw.get("injuries", []):
        team = ESPN_TEAM.get(team_block.get("displayName", ""))
        if team is None:
            continue
        for entry in team_block.get("injuries", []):
            name = (entry.get("athlete") or {}).get("displayName")
            if not name:
                continue
            status = "Out" if entry.get("status") in ("Out", "Suspension") else "Day-To-Day"
            rows.append(
                {
                    "team": team,
                    "name": name,
                    "status": status,
                    "comment": entry.get("shortComment") or "",
                    "return_date": _return_date(entry),
                    "reported_at": pd.to_datetime(entry.get("date"), utc=True, errors="coerce"),
                }
            )
    return pd.DataFrame(rows, columns=INJURY_COLUMNS)


def _first_initial_key(norm: str) -> tuple[str, str]:
    parts = norm.split()
    return (parts[-1], parts[0][:1]) if parts else ("", "")


def match_players(
    injuries: pd.DataFrame, players: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, list[str]]:
    """Attach player_id to each injury; return (matched, unmatched "Name (TEAM)" labels)."""
    roster = players[players["season"] == season][["team", "player_id", "player_name"]]
    roster = roster.drop_duplicates(["team", "player_id"]).assign(
        norm=lambda d: d["player_name"].map(normalize_name)
    )
    roster["key"] = roster["norm"].map(_first_initial_key)
    by_team = {t: g for t, g in roster.groupby("team")}
    league_unique = roster.drop_duplicates("player_id").groupby("norm")["player_id"].agg(list)

    ids: list[int | None] = []
    unmatched: list[str] = []
    for row in injuries.itertuples():
        norm = normalize_name(row.name)
        team_players = by_team.get(row.team)
        pid = None
        if team_players is not None:
            exact = team_players[team_players["norm"] == norm]
            if len(exact) == 1:
                pid = int(exact["player_id"].iloc[0])
            elif exact.empty:
                loose = team_players[team_players["key"] == _first_initial_key(norm)]
                if len(loose) == 1:
                    pid = int(loose["player_id"].iloc[0])
        if pid is None and norm in league_unique.index and len(league_unique[norm]) == 1:
            pid = int(league_unique[norm][0])
        ids.append(pid)
        if pid is None:
            unmatched.append(f"{row.name} ({row.team})")

    out = injuries.copy()
    out["player_id"] = pd.array(ids, dtype="Int64")
    return out[out["player_id"].notna()].reset_index(drop=True), unmatched
