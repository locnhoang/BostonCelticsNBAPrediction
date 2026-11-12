"""Assemble point-in-time model features and current team state."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from celtics.config import DEFAULT_REST, ELO_MEAN
from celtics.features.elo import compute_elo
from celtics.features.ratings import compute_net_ratings, compute_rest
from celtics.teams import TEAM_INDEX, TEAMS

FEATURES = [
    "elo_diff",
    "net_diff",
    "home_rest",
    "away_rest",
    "home_b2b",
    "away_b2b",
    "home_3in4",
    "away_3in4",
]


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    elo, _ = compute_elo(games)
    net, _ = compute_net_ratings(games)
    rest = compute_rest(games)
    df = games.merge(elo, on="game_id").merge(net, on="game_id").merge(rest, on="game_id")
    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]
    df["net_diff"] = df["home_net_pre"] - df["away_net_pre"]
    df["completed"] = df["home_pts"].notna()
    df["margin"] = df["home_pts"] - df["away_pts"]
    return df.sort_values(["date", "game_id"]).reset_index(drop=True)


def as_of(games: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    view = games.copy()
    future = view["date"] > cutoff
    view.loc[future, ["home_pts", "away_pts", "poss"]] = np.nan
    return view


@dataclass
class TeamState:
    elo: dict[str, float]
    net: dict[str, float]


def current_team_state(games: pd.DataFrame) -> TeamState:
    _, elo = compute_elo(games)
    _, net = compute_net_ratings(games)
    return TeamState(elo=elo, net=net)


def matchup_frame(state: TeamState, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = [
        {
            "elo_diff": state.elo.get(h, ELO_MEAN) - state.elo.get(a, ELO_MEAN),
            "net_diff": state.net.get(h, 0.0) - state.net.get(a, 0.0),
            "home_rest": float(DEFAULT_REST),
            "away_rest": float(DEFAULT_REST),
            "home_b2b": 0.0,
            "away_b2b": 0.0,
            "home_3in4": 0.0,
            "away_3in4": 0.0,
        }
        for h, a in pairs
    ]
    return pd.DataFrame(rows, columns=FEATURES)


def matchup_matrix(
    predict: Callable[[pd.DataFrame], np.ndarray], state: TeamState
) -> np.ndarray:
    pairs = [(h, a) for h in TEAMS for a in TEAMS if h != a]
    margins = predict(matchup_frame(state, pairs))
    matrix = np.zeros((len(TEAMS), len(TEAMS)))
    for (h, a), value in zip(pairs, margins, strict=True):
        matrix[TEAM_INDEX[h], TEAM_INDEX[a]] = value
    return matrix
