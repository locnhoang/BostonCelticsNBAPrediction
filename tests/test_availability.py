from datetime import date

import numpy as np
import pandas as pd
import pytest

from celtics.config import FIRST_SEASON
from celtics.data.parse import playerlog_to_frame
from celtics.features.availability import (
    current_player_values,
    game_score,
    historical_avail,
    live_avail,
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
    assert game_score(df).iloc[0] == pytest.approx(16.9)


def test_rotation_requires_min_games_and_minutes():
    rows = _rows(S, 11, pid=1, name="A")
    rows += [
        r for r in _rows(S, 11, pid=2, name="B", minutes=30) if int(r["GAME_ID"][-5:]) > 6
    ]  # 5 games
    rows += [
        r for r in _rows(S, 11, pid=3, name="C", minutes=30) if int(r["GAME_ID"][-5:]) > 7
    ]  # 4 games
    rows += [
        r for r in _rows(S, 11, pid=4, name="D", minutes=14) if int(r["GAME_ID"][-5:]) > 4
    ]  # 7 @ 14
    rows += [
        r for r in _rows(S, 11, pid=5, name="E", minutes=20) if int(r["GAME_ID"][-5:]) > 4
    ]  # 7 @ 20
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
    got = missing[(missing["game_id"] == _gid(S, 12)) & (missing["team"] == "BOS")][
        "missing_value"
    ].iloc[0]
    assert abs(got - a["game_value"]) < 1e-9


def test_values_are_point_in_time():
    games = pd.concat([_games(S, 12), _games(S + 1, 14)], ignore_index=True)
    k = 8

    def states(pts_k):
        rows = _rows(S + 1, 14)
        rows[k - 1] = _rows(S + 1, 14, pts=pts_k)[k - 1]
        return player_states(
            games, pd.concat([_frame(_rows(S, 12)), _frame(rows, S + 1)], ignore_index=True)
        )

    base, changed = states(10), states(50)
    ids = list(games.sort_values(["date", "game_id"])["game_id"])
    upto = set(ids[: ids.index(_gid(S + 1, k)) + 1])
    early = base["game_id"].isin(upto)
    pd.testing.assert_frame_equal(base[early], changed[early])
    assert not base[~early].equals(changed[~early])


def test_traded_player_counts_for_new_team_only():
    games = make_games(
        [(_gid(S, i), S, str(pd.Timestamp("2014-11-01") + pd.Timedelta(days=2 * i))[:10],
          "BOS", "NYK", 100, 90) for i in range(1, 15)]
    )  # fmt: skip
    rows = []
    for i in range(1, 15):
        date = str(pd.Timestamp("2014-11-01") + pd.Timedelta(days=2 * i))[:10]
        rows.append(player_rows(_gid(S, i), date, "BOS" if i <= 10 else "NYK", 1, "A", 30, pts=20))
        rows.append(player_rows(_gid(S, i), date, "BOS", 2, "B", 30, pts=10))
        rows.append(player_rows(_gid(S, i), date, "NYK", 3, "C", 30, pts=10))
    missing = team_missing(player_states(games, _frame(rows)))
    bos = missing[missing["team"] == "BOS"].set_index("game_id")["missing_value"]
    assert (bos[[_gid(S, i) for i in range(11, 15)]] == 0.0).all()


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


# ---- live availability ----

TODAY = date(S, 11, 26)


def _live_setup():
    """BOS, 12 completed games with A (id 1, 20 pts) and B (id 2, 10 pts); games 13-15 still to play."""
    done = _games(S, 12)
    todo = make_games(
        [(_gid(S, i), S, str(pd.Timestamp(f"{S}-11-01") + pd.Timedelta(days=2 * i))[:10], "BOS", "NYK", None, None)
         for i in (13, 14, 15)]
    )  # fmt: skip
    games = pd.concat([done, todo], ignore_index=True)
    bench = _rows(S, 12, pts=0, name="C", pid=3, minutes=10)  # keeps replacement level low
    players = _frame(_rows(S, 12, pts=20) + _rows(S, 12, pts=10, name="B", pid=2) + bench)
    return games, players


def _inj(rows):
    return pd.DataFrame(
        [
            {"team": t, "name": n, "status": st, "comment": "", "return_date": rd, "reported_at": pd.NaT, "player_id": pid}
            for t, n, st, rd, pid in rows
        ],
        columns=["team", "name", "status", "comment", "return_date", "reported_at", "player_id"],
    )  # fmt: skip


def _values(games, players):
    cur = current_player_values(games, players, S).set_index("player_id")
    return cur.loc[1, "game_value"], cur.loc[2, "game_value"]


def _avail(live, n):
    return live[live["game_id"] == _gid(S, n)].iloc[0]


def test_current_player_values_matches_next_game_state():
    games, players = _live_setup()
    cur = current_player_values(games, players, S).set_index("player_id")
    assert set(cur.index) == {1, 2, 3} and list(cur["rotation"].sort_index()) == [True, True, False]
    assert cur.loc[1, "game_value"] > cur.loc[2, "game_value"] > 0
    # equals the point-in-time state a 13th game would see
    as_if_played = games.copy()
    as_if_played.loc[as_if_played["game_id"] == _gid(S, 13), ["home_pts", "away_pts", "poss"]] = [100, 90, 100]
    nxt = _bos(player_states(as_if_played, players), 13).set_index("player_id")
    for pid in (1, 2, 3):
        for col in ("value_per_min", "expected_minutes", "game_value"):
            assert cur.loc[pid, col] == pytest.approx(nxt.loc[pid, col])


def test_live_avail_next_game_uses_out_and_dtd():
    games, players = _live_setup()
    g1, g2 = _values(games, players)
    inj = _inj([("BOS", "A", "Out", None, 1), ("BOS", "B", "Day-To-Day", None, 2)])
    live = live_avail(games, players, inj, S, TODAY, 0.5)
    assert list(live["game_id"]) == [_gid(S, n) for n in (13, 14, 15)]
    assert _avail(live, 13)["home_avail"] == pytest.approx(-(g1 + 0.5 * g2))
    assert _avail(live, 13)["away_avail"] == 0.0  # NYK: no injuries, no baseline


def test_live_avail_return_date_limits_later_games():
    games, players = _live_setup()
    g1, _ = _values(games, players)
    inj = _inj([("BOS", "A", "Out", date(S, 12, 1), 1), ("BOS", "B", "Day-To-Day", None, 2)])
    live = live_avail(games, players, inj, S, TODAY, 1.0)
    g1, g2 = _values(games, players)
    assert _avail(live, 13)["home_avail"] == pytest.approx(-(g1 + g2))
    assert _avail(live, 14)["home_avail"] == pytest.approx(
        -g1
    )  # Nov 29, before the return; Brown back
    assert _avail(live, 15)["home_avail"] == pytest.approx(0.0)  # Dec 1: on the return date


def test_out_without_return_date_counts_next_game_only():
    games, players = _live_setup()
    g1, _ = _values(games, players)
    for rd in (None, date(S, 11, 1)):  # missing, or already past
        live = live_avail(games, players, _inj([("BOS", "A", "Out", rd, 1)]), S, TODAY, 0.5)
        assert _avail(live, 13)["home_avail"] == pytest.approx(-g1)
        assert _avail(live, 14)["home_avail"] == pytest.approx(0.0)
        assert not live.isna().any().any()


def test_team_missing_from_feed_has_no_injuries():
    games, players = _live_setup()
    inj = _inj([("MIA", "Z", "Out", None, 99)])  # BOS and NYK absent from the feed
    live = live_avail(games, players, inj, S, TODAY, 0.5)
    assert (live[["home_avail", "away_avail"]] == 0.0).all().all()
    empty = live_avail(games, players, _inj([]), S, TODAY, 0.5)
    assert (empty[["home_avail", "away_avail"]] == 0.0).all().all()


def test_live_avail_baseline_is_mean_of_recent_missing():
    games, players = _live_setup()
    rows = _rows(S, 12, pts=20, skip={11, 12}) + _rows(S, 12, pts=10, name="B", pid=2)
    live = live_avail(games, _frame(rows), _inj([]), S, TODAY, 0.5)
    states = player_states(games, _frame(rows))
    miss = team_missing(states)
    bos = miss[miss["team"] == "BOS"].sort_values("game_id")["missing_value"].tail(10)
    assert bos.iloc[-1] > 0
    assert _avail(live, 13)["home_avail"] == pytest.approx(bos.mean())


def test_zero_minute_row_counts_as_played():
    games, players = _a_out_at_12()
    extra = _frame([player_rows(_gid(S, 12), "2014-11-25", "BOS", 1, "A", 0, pts=0)])
    states = player_states(games, pd.concat([players, extra], ignore_index=True))
    a = _bos(states, 12).set_index("player_id").loc[1]
    assert a["played"]
    missing = team_missing(states)
    bos12 = missing[(missing["game_id"] == _gid(S, 12)) & (missing["team"] == "BOS")]
    assert bos12["missing_value"].iloc[0] == 0.0


def test_out_back_on_or_before_next_game_not_counted():
    games, players = _live_setup()
    for rd in (TODAY, date(S, 11, 27)):  # next game (13) is Nov 27
        live = live_avail(games, players, _inj([("BOS", "A", "Out", rd, 1)]), S, TODAY, 0.5)
        assert _avail(live, 13)["home_avail"] == pytest.approx(0.0)
    g1, _ = _values(games, players)
    live = live_avail(games, players, _inj([("BOS", "A", "Out", date(S, 11, 28), 1)]), S, TODAY, 0.5)
    assert _avail(live, 13)["home_avail"] == pytest.approx(-g1)


def test_opening_day_ignores_last_season_rotation():
    games, players = _live_setup()
    games = games[games["home_pts"].notna()]
    opener = make_games([(_gid(S + 1, 1), S + 1, f"{S + 1}-10-25", "BOS", "NYK", None, None)])
    games = pd.concat([games, opener], ignore_index=True)
    assert current_player_values(games, players, S + 1).empty
    live = live_avail(games, players, _inj([("BOS", "A", "Out", None, 1)]), S + 1, date(S + 1, 10, 24), 0.5)
    assert live.iloc[0]["home_avail"] == 0.0 and live.iloc[0]["away_avail"] == 0.0
