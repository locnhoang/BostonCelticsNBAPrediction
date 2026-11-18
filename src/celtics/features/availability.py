"""Point-in-time player value, rotation absences and the historical availability feature.

Everything here is sequential: a game's values use only games strictly before it.
"""

from collections import defaultdict
from datetime import date

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
from celtics.injuries import p_out

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
    """(game_id, team) -> {player_id: minutes} for players with a box-score row.

    A row means he played (DNPs have no row; 0-minute rows are brief appearances).
    """
    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for gid, team, pid, mins in zip(
        players["game_id"], players["team"], players["player_id"], players["min"], strict=True
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
            flags[team_rows.index] = [
                _is_rotation(stats.get(int(p))) for p in team_rows["player_id"]
            ]
        for team, team_rows in game_rows.groupby("team", sort=False):
            here = {
                int(p): float(m)
                for p, m in zip(team_rows["player_id"], team_rows["min"], strict=True)
            }
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


def _value_state(
    pid: int,
    sums: dict[int, list[float]],
    recent: dict[int, list[float]],
    league_avg: float,
    replacement: float,
) -> tuple[float, float, float]:
    """(value per minute, expected minutes, game value) for a player given the history so far."""
    s, m = sums.get(pid, (0.0, 0.0))
    vpm = (VALUE_PRIOR_MINUTES * league_avg + s) / (VALUE_PRIOR_MINUTES + m)
    own_games = recent[pid][-EXPECTED_MIN_WINDOW:]
    expected = float(np.mean(own_games)) if own_games else 0.0
    return vpm, expected, max(0.0, vpm - replacement) * expected


def _record_game(
    gid: str,
    teams: tuple[str, str],
    played: dict[tuple[str, str], dict[int, float]],
    gs_lookup: dict[tuple[str, str, int], float],
    history: dict[str, list[dict[int, float]]],
    last_team: dict[int, str],
    recent: dict[int, list[float]],
    sums: dict[int, list[float]],
) -> None:
    """Fold one completed game into the running state (after both teams' states are fixed)."""
    for team in teams:
        here = played.get((gid, team), {})
        history[team].append(here)
        for pid, mins in here.items():
            last_team[pid] = team
            recent[pid].append(mins)
            s, m = sums.get(pid, (0.0, 0.0))
            sums[pid] = [VALUE_DECAY * s + gs_lookup[(gid, team, pid)], VALUE_DECAY * m + mins]


def _gs_lookup(players: pd.DataFrame) -> dict[tuple[str, str, int], float]:
    gscore = game_score(players)
    return {
        (g, t, int(p)): float(s)
        for g, t, p, s in zip(
            players["game_id"], players["team"], players["player_id"], gscore, strict=True
        )
    }


def player_states(games: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    players = players.reset_index(drop=True)
    baselines = season_baselines(players)
    played = _played_minutes(players)
    gs_lookup = _gs_lookup(players)
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
                if (
                    pid in other
                ):  # playing for the other side today: traded, new team from this game
                    continue
                vpm, expected, value = _value_state(pid, sums, recent, league_avg, replacement)
                rows.append(
                    (
                        gid, team, pid, names[pid], vpm, expected, value,
                        _is_rotation(rot_stats.get(pid)), pid in here,
                    )
                )  # fmt: skip
        _record_game(gid, (home, away), played, gs_lookup, history, last_team, recent, sums)
    return pd.DataFrame(rows, columns=STATE_COLUMNS)


CURRENT_COLUMNS = [
    "team",
    "player_id",
    "player_name",
    "value_per_min",
    "expected_minutes",
    "game_value",
    "rotation",
]


def current_player_values(games: pd.DataFrame, players: pd.DataFrame, season: int) -> pd.DataFrame:
    """Each recent player's state after his latest completed game, under his latest team."""
    players = players.reset_index(drop=True)
    league_avg, replacement = season_baselines(players).get(season, (0.0, 0.0))
    played = _played_minutes(players)
    gs_lookup = _gs_lookup(players)
    names = dict(zip(players["player_id"].astype(int), players["player_name"], strict=True))

    sums: dict[int, list[float]] = {}
    history: dict[str, list[dict[int, float]]] = defaultdict(list)
    last_team: dict[int, str] = {}
    recent: dict[int, list[float]] = defaultdict(list)
    current_season: int | None = None
    done = games[games["home_pts"].notna() & (games["season"] <= season)].sort_values(
        ["date", "game_id"]
    )
    for gid, g_season, home, away in zip(
        done["game_id"], done["season"], done["home_team"], done["away_team"], strict=True
    ):
        if g_season != current_season:
            history.clear()
            current_season = g_season
        _record_game(gid, (home, away), played, gs_lookup, history, last_team, recent, sums)
    if current_season != season:
        history.clear()  # no completed game this season yet: last season's rotation does not carry over

    rows = []
    for team, hist in history.items():
        for pid, stats in _window_stats(hist, ROTATION_WINDOW, team, last_team).items():
            vpm, expected, value = _value_state(pid, sums, recent, league_avg, replacement)
            rows.append((team, pid, names[pid], vpm, expected, value, _is_rotation(stats)))
    return pd.DataFrame(rows, columns=CURRENT_COLUMNS)


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
    lookup = {
        (g, t): v
        for g, t, v in zip(
            missing["game_id"], missing["team"], missing["missing_value"], strict=True
        )
    }
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


def live_avail(
    games: pd.DataFrame,
    players: pd.DataFrame,
    injuries_matched: pd.DataFrame,
    season: int,
    today: date,
    p_dtd: float,
) -> pd.DataFrame:
    """The avail frame of `live_avail_and_baselines`."""
    return live_avail_and_baselines(games, players, injuries_matched, season, today, p_dtd)[0]


def live_avail_and_baselines(
    games: pd.DataFrame,
    players: pd.DataFrame,
    injuries_matched: pd.DataFrame,
    season: int,
    today: date,
    p_dtd: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """avail for the season's incomplete games from the injury report, plus each team's baseline.

    A team's next game is baseline - missing, counting Out players fully and Day-To-Day players
    at p_dtd. A team is healthy when missing = 0, i.e. avail equals its baseline. Later games
    assume the usual absences continue and only lower avail for known Out value beyond the
    baseline: avail = -max(0, known_out - baseline), where known_out counts Out players whose
    return_date is after the game. An Out player with no return date, or one already past,
    therefore counts for the next game only. Like the historical feature, only rotation players
    contribute.
    """
    columns = ["game_id", "home_avail", "away_avail"]
    upcoming = games[games["home_pts"].isna() & (games["season"] == season)].sort_values(
        ["date", "game_id"]
    )
    if upcoming.empty:
        return pd.DataFrame(columns=columns), {}

    current = current_player_values(games, players, season)
    value = {
        int(p): float(v)
        for p, v, r in zip(
            current["player_id"], current["game_value"], current["rotation"], strict=True
        )
        if r
    }
    missing = team_missing(player_states(games, players))
    lookup = {
        (g, t): v
        for g, t, v in zip(
            missing["game_id"], missing["team"], missing["missing_value"], strict=True
        )
    }

    done = games[games["home_pts"].notna() & (games["season"] == season)].sort_values(
        ["date", "game_id"]
    )
    baseline: dict[str, float] = {}
    for team in set(upcoming["home_team"]) | set(upcoming["away_team"]):
        mine = done[(done["home_team"] == team) | (done["away_team"] == team)]["game_id"].tolist()
        prior = [lookup.get((g, team), 0.0) for g in mine[-AVAIL_BASELINE_GAMES:]]
        baseline[team] = float(np.mean(prior)) if prior else 0.0

    by_team = {t: g for t, g in injuries_matched.groupby("team")} if len(injuries_matched) else {}
    next_game: dict[str, str] = {}
    for gid, home, away in zip(
        upcoming["game_id"], upcoming["home_team"], upcoming["away_team"], strict=True
    ):
        next_game.setdefault(home, gid)
        next_game.setdefault(away, gid)

    def missing_for(team: str, gid: str, game_date: date) -> float:
        total = 0.0
        for row in by_team.get(team, pd.DataFrame()).itertuples():
            v = value.get(int(row.player_id), 0.0)
            total += v * p_out(
                row.status, row.return_date, game_date, today, p_dtd, next_game=next_game[team] == gid
            )
        return total

    def avail_for(team: str, gid: str, game_date: date) -> float:
        missing = missing_for(team, gid, game_date)
        if next_game[team] == gid:
            return baseline[team] - missing
        return -max(0.0, missing - baseline[team])

    out = []
    for gid, d, home, away in zip(
        upcoming["game_id"],
        upcoming["date"],
        upcoming["home_team"],
        upcoming["away_team"],
        strict=True,
    ):
        game_date = d.date()
        out.append(
            (
                gid,
                avail_for(home, gid, game_date),
                avail_for(away, gid, game_date),
            )
        )
    return pd.DataFrame(out, columns=columns), baseline
