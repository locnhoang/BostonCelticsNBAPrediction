"""Project-wide constants and small helpers."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

TEAM = "BOS"

FIRST_SEASON = 2014  # warm-up season for Elo / ratings
FIRST_TRAIN_SEASON = 2015
BACKTEST_SEASONS = list(range(2018, 2026))

MODEL_NAME = "ridge"  # switch to "gbm" only if it clearly wins the backtest
RETRAIN_DAYS = 7
GAMES_PER_TEAM = 82

N_SIMS = 10_000
DEFAULT_TEAM_SD = 2.5  # points; overridden by backtest.json "chosen_team_sd"
TEAM_SD_GRID = [0.0, 1.0, 2.0, 2.5, 3.0, 4.0]
COVERAGE_CHECKPOINT_GAMES = 15
BACKTEST_COVERAGE_SIMS = 2_000

ELO_K = 20.0
ELO_HOME = 100.0
ELO_MEAN = 1500.0
ELO_CARRYOVER = 0.75  # keep 75%, regress 25% to the mean between seasons
ELO_PER_POINT = 28.0

NET_DECAY = 0.97  # per-game decay of the net-rating average
NET_PRIOR_WEIGHT = 8.0  # prior counts as this many games, decaying like a game
NET_PRIOR_SHRINK = 0.67  # prior = 0.67 * last season's average net rating

ROTATION_WINDOW = 10
ROTATION_MIN_GAMES = 5
ROTATION_MIN_MINUTES = 15.0
VALUE_DECAY = 0.95
VALUE_PRIOR_MINUTES = 100.0
EXPECTED_MIN_WINDOW = 10
AVAIL_BASELINE_GAMES = 10
DTD_PRIOR = 0.5
DTD_MIN_CASES = 100
AVAIL_ADOPT_MARGIN = 0.001

REST_CAP = 4
DEFAULT_REST = 2


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def model_dir(self) -> Path:
        return self.root / "data" / "models"

    @property
    def injuries_dir(self) -> Path:
        return self.root / "data" / "injuries"

    @property
    def site_data(self) -> Path:
        return self.root / "site" / "data"


def season_label(season: int) -> str:
    return f"{season}-{(season + 1) % 100:02d}"


def current_season(today: date) -> int:
    return today.year if today.month >= 8 else today.year - 1
