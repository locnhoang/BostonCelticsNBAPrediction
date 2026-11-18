"""Split a predicted margin into additive drivers (exact for linear models)."""

import pandas as pd

from celtics.config import DEFAULT_REST
from celtics.features.build import AVAIL_FEATURES, FEATURES
from celtics.models.margin import MarginModel

DRIVER_ORDER = ["Season strength", "Recent form", "Home court", "Rest & schedule", "Injuries & availability"]

# Two equal teams on normal rest: what the model predicts from home court alone.
BASELINE = {
    "elo_diff": 0.0,
    "net_diff": 0.0,
    "home_rest": float(DEFAULT_REST),
    "away_rest": float(DEFAULT_REST),
    "home_b2b": 0.0,
    "away_b2b": 0.0,
    "home_3in4": 0.0,
    "away_3in4": 0.0,
    "home_avail": 0.0,
    "away_avail": 0.0,
}
GROUPS = {
    "Season strength": ["elo_diff"],
    "Recent form": ["net_diff"],
    "Rest & schedule": ["home_rest", "away_rest", "home_b2b", "away_b2b", "home_3in4", "away_3in4"],
    "Injuries & availability": AVAIL_FEATURES,
}


def explain(model: MarginModel, X: pd.DataFrame) -> pd.DataFrame | None:
    """Home-perspective contribution of each driver; rows sum to model.predict_margin(X)."""
    if not getattr(model, "is_linear", False):
        return None
    base = pd.DataFrame([BASELINE] * len(X), columns=[*FEATURES, *AVAIL_FEATURES])
    home_court = model.predict_margin(base)
    parts = {"Home court": home_court}
    active = [label for label in DRIVER_ORDER if label == "Home court" or set(GROUPS[label]) <= set(model.features)]
    for label in active:
        if label == "Home court":
            continue
        cols = GROUPS[label]
        swapped = base.copy()
        swapped[cols] = X[cols].to_numpy()
        parts[label] = model.predict_margin(swapped) - home_court
    return pd.DataFrame(parts, index=X.index)[active]
