"""Walk-forward backtesting."""

import pandas as pd

from celtics.config import FIRST_TRAIN_SEASON
from celtics.features.build import FEATURES
from celtics.models.margin import make_model


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
