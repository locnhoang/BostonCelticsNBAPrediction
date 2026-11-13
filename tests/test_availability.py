import numpy as np
import pandas as pd

from celtics.config import FIRST_SEASON
from celtics.data.parse import playerlog_to_frame
from celtics.features.availability import (
    game_score,
    historical_avail,
    player_states,
    season_baselines,
    team_missing,
)
from celtics.features.build import AVAIL_FEATURES, FEATURES, build_features
from tests.factories import make_games, player_rows
from tests.synth import make_league

S = FIRST_SEASON


def _gid(season, n):
    return f"002{season % 100:02d}{n:05d}"


def _games(season, n, opp="NYK"):
    return make_games(
        [(_gid(season, i), season, str(pd.Timestamp(f"{season}-11-01") + pd.Timedelta(days=2 * i))[:10],
          "BOS", opp, 100, 90) for i in range(1, n + 1)]
    )  # fmt: skip


def _rows(season, n, skip=(), pts=10, name="A", pid=1, minutes=30):
    return [
        player_rows(_gid(season, i), str(pd.Timestamp(f"{season}-11-01") + pd.Timedelta(days=2 * i))[:10],
                    "BOS", pid, name, minutes, pts=pts)
        for i in range(1, n + 1) if i not in skip
    ]  # fmt: skip


def _frame(rows, season=S):
    return playerlog_to_frame(pd.DataFrame(rows), season)


def _bos(states, n):
    return states[(states["game_id"] == _gid(S, n)) & (states["team"] == "BOS")]


def test_game_score_formula():
    df = _frame([player_rows("0021400001", "2014-11-01", "BOS", 1, "A", 30, pts=20, fgm=8, fga=15, ftm=4,
                             fta=5, oreb=2, dreb=6, ast=5, stl=1, blk=1, tov=3, pf=2)])  # fmt: skip
    assert game_score(df).iloc[0] == np.float64(16.9) or abs(game_score(df).iloc[0] - 16.9) < 1e-9


def test_rotation_requires_min_games_and_minutes():
    rows = _rows(S, 11, pid=1, name="A")
    rows += [r for r in _rows(S, 11, pid=2, name="B", minutes=30) if int(r["GAME_ID"][-5:]) > 6]  # 5 games
    rows += [r for r in _rows(S, 11, pid=3, name="C", minutes=30) if int(r["GAME_ID"][-5:]) > 7]  # 4 games
    rows += [r for r in _rows(S, 11, pid=4, name="D", minutes=14) if int(r["GAME_ID"][-5:]) > 4]  # 7 @ 14
    rows += [r for r in _rows(S, 11, pid=5, name="E", minutes=20) if int(r["GAME_ID"][-5:]) > 4]  # 7 @ 20
    states = _bos(player_states(_games(S, 12), _frame(rows)), 12).set_index("player_id")
    assert states.loc[1, "rotation"] and states.loc[5, "rotation"] and states.loc[2, "rotation"]
    assert not states.loc[3, "rotation"]
    assert not states.loc[4, "rotation"]


def _a_out_at_12():
    return _games(S, 12), _frame(_rows(S, 12, skip={12}, pts=20))


def test_absent_rotation_player_counts_as_missing():
    games, players = _a_out_at_12()
    states = player_states(games, players)
    a = _bos(states, 12).set_index("player_id").loc[1]
    assert a["rotation"] and not a["played"] and a["game_value"] > 0
    missing = team_missing(states)
    got = missing[(missing["game_id"] == _gid(S, 12)) & (missing["team"] == "BOS")]["missing_value"].iloc[0]
    assert abs(got - a["game_value"]) < 1e-9


def test_values_are_point_in_time():
    seasons = [S, S + 1]
    games = pd.concat([_games(s, 12) for s in seasons], ignore_index=True)

    def players(pts_last):
        return pd.concat([_frame(_rows(S, 12)), _frame(_rows(S + 1, 11) + _rows(S + 1, 12, pts=pts_last)[-1:], S + 1)])

    base = player_states(games, players(10))
    changed = player_states(games, players(50))
    pd.testing.assert_frame_equal(base, changed)


def test_low_minutes_shrink_to_league_avg():
    rows = _rows(S, 12, pts=15)
    rows.append(player_rows(_gid(S, 3), "2014-11-07", "BOS", 9, "B", 2, pts=10))
    players = _frame(rows)
    states = player_states(_games(S, 12), players)
    b = _bos(states, 4).set_index("player_id").loc[9]
    league_avg = season_baselines(players)[S][0]
    assert abs(b["value_per_min"] - league_avg) < 0.1


def test_avail_sitting_negative_returning_positive():
    games = _games(S, 13)
    players = _frame(_rows(S, 13, skip={11, 12}, pts=20))
    avail = historical_avail(games, players).set_index("game_id")["home_avail"]
    assert avail[_gid(S, 12)] < 0
    assert avail[_gid(S, 13)] > 0


def test_baseline_with_no_history_is_zero():
    avail = historical_avail(_games(S, 3), _frame(_rows(S, 3)))
    assert avail["home_avail"].iloc[0] == 0.0
    assert not avail[["home_avail", "away_avail"]].isna().any().any()


def test_incomplete_games_get_zero_avail():
    games = _games(S, 3)
    games.loc[2, ["home_pts", "away_pts", "poss"]] = np.nan
    avail = historical_avail(games, _frame(_rows(S, 2)))
    assert avail["home_avail"].iloc[2] == 0.0


def test_build_features_without_players_has_zero_avail():
    out = build_features(make_league([2023, 2024], current=2024, completed_rounds=5))
    assert (out[AVAIL_FEATURES] == 0.0).all().all()
    assert FEATURES == [
        "elo_diff", "net_diff", "home_rest", "away_rest", "home_b2b", "away_b2b", "home_3in4", "away_3in4",
    ]  # fmt: skip


def test_build_features_with_players_adds_avail():
    games = _games(S, 13)
    players = _frame(_rows(S, 13, skip={12}, pts=20))
    out = build_features(games, players)
    row = out[out["game_id"] == _gid(S, 12)].iloc[0]
    assert row["home_avail"] < 0
