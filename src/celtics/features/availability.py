"""Point-in-time player value, rotation absences and the historical availability feature.

Everything here is sequential: a game's values use only games strictly before it.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from celtics.config import (
    AVAIL_BASELINE_GAMES,
    EXPECTED_MIN_WINDOW,
    FIRST_SEASON,
    ROTATION_MIN_GAMES,
    ROTATION_MIN_MINUTES,
    ROTATION_WINDOW,
    VALUE_DECAY,
    VALUE_PRIOR_MINUTES,
)

STATE_COLUMNS = [
    "game_id", "team", "player_id", "player_name",
    "value_per_min", "expected_minutes", "game_value", "rotation", "played",
]  # fmt: skip


def game_score(players: pd.DataFrame) -> pd.Series:
    return (
        players["pts"]
        + 0.4 * players["fgm"]
        - 0.7 * players["fga"]
        - 0.4 * (players["fta"] - players["ftm"])
        + 0.7 * players["oreb"]
        + 0.3 * players["dreb"]
        + players["stl"]
        + 0.7 * players["ast"]
        + 0.7 * players["blk"]
        - 0.4 * players["pf"]
        - players["tov"]
    )


def _window_stats(
    history: list[dict[int, float]], window: int, team: str, last_team: dict[int, str]
) -> dict[int, tuple[int, float]]:
    """Per player: (games played, mean minutes) in the team's last `window` games.

    Players who have since played for another team are skipped (traded players count for
    their new team from his first game there).
    """
    total: dict[int, float] = defaultdict(float)
    count: dict[int, int] = defaultdict(int)
    for minutes in history[-window:]:
        for pid, mins in minutes.items():
            if last_team.get(pid) != team:
                continue
            total[pid] += mins
            count[pid] += 1
    return {pid: (count[pid], total[pid] / count[pid]) for pid in count}


def _is_rotation(stats: tuple[int, float] | None) -> bool:
    return stats is not None and stats[0] >= ROTATION_MIN_GAMES and stats[1] >= ROTATION_MIN_MINUTES


def _played_minutes(players: pd.DataFrame) -> dict[tuple[str, str], dict[int, float]]:
    """(game_id, team) -> {player_id: minutes} for players who actually played."""
    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    on_court = players[players["min"] > 0]
    for gid, team, pid, mins in zip(
        on_court["game_id"], on_court["team"], on_court["player_id"], on_court["min"], strict=True
    ):
        out[(gid, team)][int(pid)] = float(mins)
    return out


def _rotation_flags(players: pd.DataFrame) -> np.ndarray:
    """Per player row (positional): was he in the team's rotation going into that game?"""
    flags = np.zeros(len(players), dtype=bool)
    ordered = players.reset_index(drop=True).sort_values(["date", "game_id"])
    history: dict[str, list[dict[int, float]]] = defaultdict(list)
    last_team: dict[int, str] = {}
    current_season: int | None = None
    for _, game_rows in ordered.groupby(["date", "game_id"], sort=True):
        season = int(game_rows["season"].iloc[0])
        if season != current_season:
            history.clear()
            current_season = season
        for team, team_rows in game_rows.groupby("team", sort=False):
            stats = _window_stats(history[team], ROTATION_WINDOW, team, last_team)
            flags[team_rows.index] = [_is_rotation(stats.get(int(p))) for p in team_rows["player_id"]]
        for team, team_rows in game_rows.groupby("team", sort=False):
            here = {int(p): float(m) for p, m in zip(team_rows["player_id"], team_rows["min"], strict=True) if m > 0}
            history[team].append(here)
            last_team.update(dict.fromkeys(here, team))
    return flags


def season_baselines(players: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """season -> (league_avg, replacement) per minute, taken from the previous season.

    FIRST_SEASON (warm-up only) uses its own season. Any later season uses the previous
    season, or the nearest earlier one present, or (0, 0) if there is none; never its own.
    """
    if players.empty:
        return {}
    players = players.reset_index(drop=True)
    df = players[players["min"] > 0]
    gs = game_score(df)
    rot = pd.Series(_rotation_flags(players), index=players.index).loc[df.index]
    own: dict[int, tuple[float, float]] = {}
    for season in sorted(df["season"].unique()):
        in_season = df["season"] == season
        minutes = df.loc[in_season, "min"].sum()
        league_avg = gs[in_season].sum() / minutes if minutes > 0 else 0.0
        bench = in_season & ~rot
        bench_minutes = df.loc[bench, "min"].sum()
        replacement = gs[bench].sum() / bench_minutes if bench_minutes > 0 else league_avg
        own[int(season)] = (float(league_avg), float(replacement))
    out: dict[int, tuple[float, float]] = {}
    for s in own:
        if s == FIRST_SEASON:
            out[s] = own[s]
            continue
        earlier = [e for e in own if e < s]
        out[s] = own[max(earlier)] if earlier else (0.0, 0.0)  # never the season's own data
    return out


def player_states(games: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    players = players.reset_index(drop=True)
    baselines = season_baselines(players)
    played = _played_minutes(players)
    gscore = game_score(players)
    on_court = players["min"] > 0
    gs_lookup = {
        (g, t, int(p)): float(s)
        for g, t, p, s in zip(
            players.loc[on_court, "game_id"],
            players.loc[on_court, "team"],
            players.loc[on_court, "player_id"],
            gscore[on_court],
            strict=True,
        )
    }
    names = dict(zip(players["player_id"].astype(int), players["player_name"], strict=True))

    sums: dict[int, list[float]] = {}  # player -> [decayed game score, decayed minutes]
    history: dict[str, list[dict[int, float]]] = defaultdict(list)
    last_team: dict[int, str] = {}
    recent: dict[int, list[float]] = defaultdict(list)  # player's own last games played, any team
    current_season: int | None = None
    rows: list[tuple] = []

    done = games[games["home_pts"].notna()].sort_values(["date", "game_id"])
    for gid, season, home, away in zip(
        done["game_id"], done["season"], done["home_team"], done["away_team"], strict=True
    ):
        if season != current_season:
            history.clear()  # rotation and expected minutes reset each season
            current_season = season
        league_avg, replacement = baselines.get(int(season), (0.0, 0.0))
        for team in (home, away):
            hist = history[team]
            rot_stats = _window_stats(hist, ROTATION_WINDOW, team, last_team)
            here = played.get((gid, team), {})
            other = played.get((gid, away if team == home else home), {})
            for pid in rot_stats:
                if pid in other:  # playing for the other side today: traded, new team from this game
                    continue
                s, m = sums.get(pid, (0.0, 0.0))
                vpm = (VALUE_PRIOR_MINUTES * league_avg + s) / (VALUE_PRIOR_MINUTES + m)
                own_games = recent[pid][-EXPECTED_MIN_WINDOW:]
                expected = float(np.mean(own_games)) if own_games else 0.0
                rows.append(
                    (
                        gid, team, pid, names[pid], vpm, expected,
                        max(0.0, vpm - replacement) * expected,
                        _is_rotation(rot_stats.get(pid)), pid in here,
                    )
                )  # fmt: skip
        for team in (home, away):  # update only after both teams' states are fixed
            here = played.get((gid, team), {})
            history[team].append(here)
            for pid, mins in here.items():
                last_team[pid] = team
                recent[pid].append(mins)
                s, m = sums.get(pid, (0.0, 0.0))
                sums[pid] = [VALUE_DECAY * s + gs_lookup[(gid, team, pid)], VALUE_DECAY * m + mins]
    return pd.DataFrame(rows, columns=STATE_COLUMNS)


def team_missing(states: pd.DataFrame) -> pd.DataFrame:
    out_rotation = states[states["rotation"] & ~states["played"]]
    missing = out_rotation.groupby(["game_id", "team"])["game_value"].sum()
    return (
        states[["game_id", "team"]]
        .drop_duplicates()
        .merge(missing.rename("missing_value").reset_index(), on=["game_id", "team"], how="left")
        .fillna({"missing_value": 0.0})
        .reset_index(drop=True)
    )


def historical_avail(games: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """avail = baseline - missing: negative when more rotation value than usual is out."""
    missing = team_missing(player_states(games, players))
    lookup = {(g, t): v for g, t, v in zip(missing["game_id"], missing["team"], missing["missing_value"], strict=True)}
    done = games[games["home_pts"].notna()].sort_values(["date", "game_id"])
    recent: dict[str, list[float]] = defaultdict(list)
    avail: dict[str, dict[str, float]] = {}
    current_season: int | None = None
    for gid, season, home, away in zip(
        done["game_id"], done["season"], done["home_team"], done["away_team"], strict=True
    ):
        if season != current_season:
            recent.clear()
            current_season = season
        values = {}
        for side, team in (("home", home), ("away", away)):
            prior = recent[team][-AVAIL_BASELINE_GAMES:]
            baseline = float(np.mean(prior)) if prior else 0.0
            values[side] = baseline - lookup.get((gid, team), 0.0)
        avail[gid] = values
        for team in (home, away):
            recent[team].append(lookup.get((gid, team), 0.0))
    return pd.DataFrame(
        {
            "game_id": games["game_id"].to_numpy(),
            "home_avail": [avail.get(g, {}).get("home", 0.0) for g in games["game_id"]],
            "away_avail": [avail.get(g, {}).get("away", 0.0) for g in games["game_id"]],
        }
    )
