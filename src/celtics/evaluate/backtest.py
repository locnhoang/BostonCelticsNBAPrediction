"""Walk-forward backtesting and season-level interval coverage."""

import numpy as np
import pandas as pd

from celtics.config import COVERAGE_CHECKPOINT_GAMES, FIRST_TRAIN_SEASON
from celtics.features.build import FEATURES, as_of, build_features, current_team_state
from celtics.models.margin import make_model
from celtics.simulate.inputs import prepare_sim_inputs
from celtics.simulate.season import simulate_season
from celtics.teams import TEAM_INDEX


def walk_forward(features: pd.DataFrame, model_name: str, test_seasons: list[int]) -> pd.DataFrame:
    out = []
    done = features[features["completed"]]
    for season in test_seasons:
        train = done[(done["season"] >= FIRST_TRAIN_SEASON) & (done["season"] < season)]
        test = done[done["season"] == season]
        if train.empty or test.empty:
            continue
        model = make_model(model_name).fit(train[FEATURES], train["margin"])
        out.append(
            pd.DataFrame(
                {
                    "game_id": test["game_id"].to_numpy(),
                    "season": season,
                    "prob": model.win_prob(test[FEATURES]),
                    "outcome": (test["margin"] > 0).astype(float).to_numpy(),
                }
            )
        )
    if not out:
        return pd.DataFrame(columns=["game_id", "season", "prob", "outcome"])
    return pd.concat(out, ignore_index=True)


def checkpoint_date(season_games: pd.DataFrame, n_games: int = COVERAGE_CHECKPOINT_GAMES) -> pd.Timestamp:
    long = pd.concat(
        [
            season_games[["date", "home_team"]].rename(columns={"home_team": "team"}),
            season_games[["date", "away_team"]].rename(columns={"away_team": "team"}),
        ]
    ).sort_values("date")
    nth = long.groupby("team")["date"].nth(n_games - 1).sort_values()
    return pd.Timestamp(nth.iloc[len(nth) // 2])


def _final_wins(season_games: pd.DataFrame) -> np.ndarray:
    done = season_games[season_games["home_pts"].notna()]
    winners = np.where(done["home_pts"] > done["away_pts"], done["home_team"], done["away_team"])
    wins = np.zeros(len(TEAM_INDEX), dtype=np.int64)
    for team in winners:
        wins[TEAM_INDEX[team]] += 1
    return wins


def season_coverage(
    games: pd.DataFrame,
    model_name: str,
    test_seasons: list[int],
    team_sds: list[float],
    n_sims: int,
    seed: int = 0,
) -> dict[float, float]:
    covered = dict.fromkeys(team_sds, 0)
    total = 0
    for season in test_seasons:
        season_games = games[games["season"] == season]
        if season_games.empty:
            continue
        actual = _final_wins(season_games)
        view = as_of(games[games["season"] <= season], checkpoint_date(season_games))
        feats = build_features(view)
        train = feats[
            feats["completed"] & (feats["season"] >= FIRST_TRAIN_SEASON) & (feats["season"] < season)
        ]
        model = make_model(model_name).fit(train[FEATURES], train["margin"])
        inp = prepare_sim_inputs(feats, model, current_team_state(view), season, target_games=None)
        total += len(actual)
        for sd in team_sds:
            res = simulate_season(inp, n_sims=n_sims, team_sd=sd, seed=seed)
            lo, hi = np.percentile(res.wins, [10, 90], axis=0)
            covered[sd] += int(((actual >= lo) & (actual <= hi)).sum())
    return {sd: (covered[sd] / total if total else 0.0) for sd in team_sds}
