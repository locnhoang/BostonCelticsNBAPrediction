"""Point-margin models. Win probability = Phi(margin / sigma)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from celtics.config import ELO_HOME, ELO_PER_POINT
from celtics.features.build import AVAIL_FEATURES, FEATURES


def win_prob(margin, sigma: float) -> np.ndarray:
    return norm.cdf(np.asarray(margin, dtype=float) / sigma)


class MarginModel:
    name = "base"
    is_linear = False
    features: list[str] = FEATURES

    def __init__(self) -> None:
        self.sigma = 12.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MarginModel":
        self._fit(X[self.features], y)
        resid = np.asarray(y, dtype=float) - self.predict_margin(X)
        self.sigma = float(np.std(resid, ddof=1))
        return self

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    def predict_margin(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def win_prob(self, X: pd.DataFrame) -> np.ndarray:
        return win_prob(self.predict_margin(X), self.sigma)


class HomeCourtModel(MarginModel):
    name = "homecourt"
    is_linear = True

    def _fit(self, X, y):
        self.mean_ = float(np.mean(y))

    def predict_margin(self, X):
        return np.full(len(X), self.mean_)


class EloModel(MarginModel):
    name = "elo"
    is_linear = True

    def _fit(self, X, y):
        pass

    def predict_margin(self, X):
        return (X["elo_diff"].to_numpy(dtype=float) + ELO_HOME) / ELO_PER_POINT


class RidgeModel(MarginModel):
    name = "ridge"
    is_linear = True

    def _fit(self, X, y):
        self.pipe_ = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)

    def predict_margin(self, X):
        return self.pipe_.predict(X[self.features])


class RidgeAvailModel(RidgeModel):
    name = "ridge_avail"
    features = FEATURES + AVAIL_FEATURES


class GBMModel(MarginModel):
    name = "gbm"

    def _fit(self, X, y):
        self.model_ = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=0,
        ).fit(X, y)

    def predict_margin(self, X):
        return self.model_.predict(X[self.features])


MODELS: dict[str, type[MarginModel]] = {
    "homecourt": HomeCourtModel,
    "elo": EloModel,
    "ridge": RidgeModel,
    "ridge_avail": RidgeAvailModel,
    "gbm": GBMModel,
}


def make_model(name: str) -> MarginModel:
    return MODELS[name]()


def save_model(model: MarginModel, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> MarginModel:
    return joblib.load(path)
