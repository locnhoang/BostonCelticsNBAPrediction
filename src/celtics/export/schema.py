"""JSON contract between the pipeline and the website."""

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

Prob = Annotated[float, Field(ge=0.0, le=1.0)]
TOL = 5e-3  # values are rounded to 4 decimals, so sums drift slightly


def _sums_to_one(values, name: str) -> None:
    total = sum(values)
    if abs(total - 1.0) > TOL:
        raise ValueError(f"{name} sums to {total}, expected 1")


class TeamSummary(BaseModel):
    team: str
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    proj_wins_median: float
    proj_wins_p10: float
    proj_wins_p25: float
    proj_wins_p75: float
    proj_wins_p90: float
    p_50_wins: Prob
    p_playoffs: Prob
    p_top6: Prob
    p_playin: Prob
    p_miss: Prob
    p_seed1: Prob
    win_distribution: dict[str, Prob]

    @model_validator(mode="after")
    def _consistent(self):
        _sums_to_one([self.p_top6, self.p_playin, self.p_miss], "top6+playin+miss")
        _sums_to_one(self.win_distribution.values(), "win_distribution")
        if not self.proj_wins_p10 <= self.proj_wins_median <= self.proj_wins_p90:
            raise ValueError("percentiles out of order")
        return self


class GameRow(BaseModel):
    game_id: str
    date: str
    opponent: str
    home: bool
    rest_days: float
    opp_rest_days: float
    b2b: bool
    completed: bool
    win_prob: Prob | None
    pred_margin: float | None
    result: str | None
    pregame_win_prob: Prob | None


class ConferenceRow(BaseModel):
    team: str
    wins: int
    losses: int
    proj_wins: float
    p_playoffs: Prob
    p_top6: Prob
    p_seed1: Prob


class LiveMetrics(BaseModel):
    n: int
    log_loss: float
    brier: float
    accuracy: Prob


class Metrics(BaseModel):
    live: LiveMetrics | None


class Predictions(BaseModel):
    generated_at: str
    model_version: str
    season: str
    team_summary: TeamSummary
    games: list[GameRow]
    seed_probabilities: dict[str, Prob]
    conference_table: list[ConferenceRow]
    metrics: Metrics

    @model_validator(mode="after")
    def _seeds(self):
        if list(self.seed_probabilities) != [str(k) for k in range(1, 16)]:
            raise ValueError("seed_probabilities must have keys '1'..'15'")
        _sums_to_one(self.seed_probabilities.values(), "seed_probabilities")
        return self


class HistoryEntry(BaseModel):
    date: str
    record: str
    projected_wins_median: float
    p10: float
    p90: float
    playoff_prob: Prob
    top6_prob: Prob
    seed1_prob: Prob


class ModelScore(BaseModel):
    n: int
    log_loss: float
    brier: float
    accuracy: Prob


class CalibrationBin(BaseModel):
    bin_lo: Prob
    bin_hi: Prob
    mean_pred: Prob
    observed: Prob
    n: int


class BacktestReport(BaseModel):
    generated_at: str
    test_seasons: list[int]
    main_model: str
    models: dict[str, ModelScore]
    calibration: list[CalibrationBin]
    coverage: dict[str, Prob]
    chosen_team_sd: float
