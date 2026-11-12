import numpy as np
import pandas as pd
import pytest

from celtics.data.parse import (
    GAME_COLUMNS,
    gamelog_to_games,
    merge_schedule_results,
    schedule_to_games,
)
from tests.factories import gamelog_rows, schedule_payload


def test_gamelog_to_games_pairs_home_and_away():
    raw = pd.DataFrame(
        gamelog_rows("0022600002", "2026-10-23", "NYK", "BOS", 101, 110)
        + gamelog_rows("0022600001", "2026-10-22", "BOS", "PHI", 120, 99)
        + gamelog_rows("0062600001", "2026-12-16", "BOS", "OKC", 100, 98)  # Cup final: excluded
    )
    games = gamelog_to_games(raw, 2026)
    assert list(games.columns) == GAME_COLUMNS
    assert games["game_id"].tolist() == ["0022600001", "0022600002"]
    first = games.iloc[0]
    assert (first.home_team, first.away_team) == ("BOS", "PHI")
    assert (first.home_pts, first.away_pts) == (120.0, 99.0)
    assert first.poss == pytest.approx(100.0, abs=1e-3)
    assert first.season == 2026
    assert first.date == pd.Timestamp("2026-10-22")


def test_gamelog_to_games_empty():
    games = gamelog_to_games(pd.DataFrame(), 2026)
    assert games.empty and list(games.columns) == GAME_COLUMNS


def test_schedule_to_games_skips_tbd_and_non_regular_season():
    raw = schedule_payload(
        "2026-27",
        [
            ("0022600001", "2026-10-22", "BOS", "PHI"),
            ("0022600900", "2026-12-10", "", ""),  # TBD Cup knockout slot
            ("0012600001", "2026-10-05", "BOS", "NYK"),  # preseason
        ],
    )
    games = schedule_to_games(raw, 2026)
    assert games["game_id"].tolist() == ["0022600001"]
    assert games["home_pts"].isna().all() and games["poss"].isna().all()


def test_schedule_to_games_skips_nan_tricodes():
    raw = schedule_payload(
        "2026-27",
        [("0022600001", "2026-10-22", "BOS", "PHI"), ("0022600900", "2026-12-10", "", "")],
    )
    raw["homeTeam_teamTricode"] = raw["homeTeam_teamTricode"].astype(object)
    raw.loc[1, ["homeTeam_teamTricode", "awayTeam_teamTricode"]] = np.nan
    games = schedule_to_games(raw, 2026)
    assert games["game_id"].tolist() == ["0022600001"]


def test_schedule_to_games_rejects_wrong_season():
    raw = schedule_payload("2025-26", [("0022500001", "2025-10-21", "BOS", "PHI")])
    with pytest.raises(ValueError, match="2025-26"):
        schedule_to_games(raw, 2026)


def test_merge_prefers_results_and_keeps_unplayed():
    sched = schedule_to_games(
        schedule_payload(
            "2026-27",
            [("0022600001", "2026-10-22", "BOS", "PHI"), ("0022600003", "2026-10-25", "BOS", "MIA")],
        ),
        2026,
    )
    results = gamelog_to_games(
        pd.DataFrame(gamelog_rows("0022600001", "2026-10-22", "BOS", "PHI", 120, 99)), 2026
    )
    merged = merge_schedule_results(sched, results)
    assert merged["game_id"].tolist() == ["0022600001", "0022600003"]
    assert merged["home_pts"].tolist()[0] == 120.0
    assert pd.isna(merged["home_pts"].tolist()[1])


def test_neutral_site_game_is_kept():
    raw = pd.DataFrame(gamelog_rows("0022500147", "2025-11-01", "DAL", "DET", 100, 110, neutral=True))
    games = gamelog_to_games(raw, 2025)
    assert games["game_id"].tolist() == ["0022500147"]


def test_neutral_site_home_is_alphabetically_second():
    for first, second in (("DET", "DAL"), ("DAL", "DET")):
        raw = pd.DataFrame(gamelog_rows("0022500147", "2025-11-01", first, second, 100, 110, neutral=True))
        g = gamelog_to_games(raw, 2025).iloc[0]
        assert (g.home_team, g.away_team) == ("DET", "DAL")
        expected = (110.0, 100.0) if first == "DAL" else (100.0, 110.0)
        assert (g.home_pts, g.away_pts) == expected


@pytest.mark.parametrize("n_rows", [1, 3])
def test_gamelog_game_without_two_rows_raises(n_rows):
    rows = gamelog_rows("0022500147", "2025-11-01", "BOS", "PHI", 100, 90)
    rows = rows[:1] if n_rows == 1 else rows + rows[:1]
    with pytest.raises(ValueError, match="0022500147"):
        gamelog_to_games(pd.DataFrame(rows), 2025)


def test_merge_corrects_reversed_neutral_game_from_schedule():
    # results guessed DET as home; the schedule says DAL is home
    results = gamelog_to_games(
        pd.DataFrame(gamelog_rows("0022500147", "2025-11-01", "DAL", "DET", 100, 110, neutral=True)), 2025
    )
    assert (results.iloc[0].home_team, results.iloc[0].home_pts) == ("DET", 110.0)
    sched = schedule_to_games(
        schedule_payload("2025-26", [("0022500147", "2025-11-01", "DAL", "DET")]), 2025
    )
    merged = merge_schedule_results(sched, results).iloc[0]
    assert (merged.home_team, merged.away_team) == ("DAL", "DET")
    assert (merged.home_pts, merged.away_pts) == (100.0, 110.0)
    assert merged.poss == pytest.approx(100.0, abs=1e-3)


def test_merge_raises_when_teams_do_not_match():
    results = gamelog_to_games(
        pd.DataFrame(gamelog_rows("0022500147", "2025-11-01", "BOS", "PHI", 100, 90)), 2025
    )
    sched = schedule_to_games(
        schedule_payload("2025-26", [("0022500147", "2025-11-01", "BOS", "NYK")]), 2025
    )
    with pytest.raises(ValueError, match="0022500147"):
        merge_schedule_results(sched, results)
