"""ESPN injury feed: parse into a frame and match injured players to their stats."""

import logging
import re
import unicodedata
from datetime import date

import pandas as pd

from celtics.config import DTD_MIN_CASES, DTD_PRIOR

log = logging.getLogger(__name__)

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
    "Los Angeles Clippers": "LAC",
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


def unknown_teams(raw: dict) -> list[str]:
    """Team display names in the feed that ESPN_TEAM does not know (their injuries are skipped)."""
    names = {b.get("displayName") or "" for b in raw.get("injuries", [])}
    return sorted(n for n in names if n and n not in ESPN_TEAM)


def parse_injuries(raw: dict) -> pd.DataFrame:
    unknown = unknown_teams(raw)
    if unknown:
        log.warning("ESPN injury feed has unknown teams, skipped: %s", ", ".join(unknown))
    rows = []
    for team_block in raw.get("injuries", []):
        team = ESPN_TEAM.get(team_block.get("displayName") or "")
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
            elif exact.empty and norm not in league_unique.index:
                # a full name that exists elsewhere is a different player, not a loose match
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
    matched = out[out["player_id"].notna()]
    matched = matched.sort_values("status", key=lambda c: c != "Out", kind="stable")  # Out wins
    matched = matched.drop_duplicates("player_id", keep="first")
    return matched.reset_index(drop=True), unmatched


def update_dtd_log(
    log: list[dict],
    injuries_matched: pd.DataFrame,
    games: pd.DataFrame,
    players: pd.DataFrame,
    today: date,
    rotation_ids: set[int] | None = None,
) -> list[dict]:
    """Log today's Day-To-Day players and resolve earlier cases against the team's next game.

    With `rotation_ids`, only Day-To-Day players in the rotation are logged (the share that
    matters for p_dtd); existing entries are kept as they are.

    A case resolves once the team has a completed game dated on or after the case (the log is
    written before tip-off); the player played if he has a box-score row in it.
    """
    out = [dict(case) for case in log]
    seen = {(c["date"], c["player_id"]) for c in out}
    day = today.isoformat()
    dtd = injuries_matched[injuries_matched["status"] == "Day-To-Day"]
    if rotation_ids is not None:
        dtd = dtd[dtd["player_id"].astype(int).isin(rotation_ids)]
    for row in dtd.itertuples():
        key = (day, int(row.player_id))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "date": day,
                "team": row.team,
                "player_id": int(row.player_id),
                "name": row.name,
                "played": None,
            }
        )

    done = games[games["home_pts"].notna()].sort_values(["date", "game_id"])
    appeared = set(zip(players["game_id"], players["player_id"].astype(int), strict=True))
    for case in out:
        if case["played"] is not None:
            continue
        after = done[
            ((done["home_team"] == case["team"]) | (done["away_team"] == case["team"]))
            & (done["date"] >= pd.Timestamp(case["date"]))
        ]
        if not after.empty:
            case["played"] = (after["game_id"].iloc[0], case["player_id"]) in appeared
    return out


def learn_p_dtd(log: list[dict]) -> float:
    """Share of resolved Day-To-Day cases that did not play; the prior until enough cases."""
    resolved = [c for c in log if c["played"] is not None]
    if len(resolved) < DTD_MIN_CASES:
        return DTD_PRIOR
    return sum(1 for c in resolved if not c["played"]) / len(resolved)


def p_out(status: str, return_date, game_date: date, today: date, p_dtd: float, next_game: bool = True) -> float:
    """Probability a listed player misses a game.

    Next game: Out counts fully unless his return date falls between today and the game;
    Day-To-Day counts p_dtd. Later games: only Out players whose return date is after the game.
    """
    has_date = pd.notna(return_date)
    if not next_game:
        return 1.0 if status == "Out" and has_date and return_date > game_date else 0.0
    if status == "Out":
        return 0.0 if has_date and today <= return_date <= game_date else 1.0
    return p_dtd if status == "Day-To-Day" else 0.0
