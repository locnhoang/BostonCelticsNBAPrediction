"""JSON contract between the pipeline and the website."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, model_validator

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


class Driver(BaseModel):
    label: str
    value: float


class TeamContext(BaseModel):
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    conf_rank: int = Field(ge=1, le=15)
    games_back: float
    last10: str = Field(pattern=r"^\d+-\d+$")
    streak: str = Field(pattern=r"^([WL]\d+)?$")
    off_rating: float
    def_rating: float
    net_rating: float
    off_rank: int = Field(ge=1, le=30)
    def_rank: int = Field(ge=1, le=30)
    net_rank: int = Field(ge=1, le=30)


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
    correct: bool | None = None
    drivers: list[Driver] | None
    win_prob_healthy: Prob | None = None


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


class ModelRecord(BaseModel):
    since: str
    correct: int = Field(ge=0)
    wrong: int = Field(ge=0)


class Metrics(BaseModel):
    live: LiveMetrics | None
    record: ModelRecord | None = None


class InjuryPlayer(BaseModel):
    player_id: int | None
    name: str
    status: Literal["Available", "Out", "Day-To-Day"]
    comment: str
    return_date: str | None
    p_out: Prob
    expected_minutes: float = Field(ge=0)
    game_value: float = Field(ge=0)
    rotation: bool


class InjuryReport(BaseModel):
    available: bool
    as_of: str | None
    p_dtd: Prob
    unmatched: list[str]
    teams: dict[str, list[InjuryPlayer]]


class _TeamPage(BaseModel):
    """Fields and checks shared by a team's file and the legacy predictions file."""

    generated_at: str
    model_version: str
    season: str
    team_summary: TeamSummary
    games: list[GameRow]
    seed_probabilities: dict[str, Prob]
    conference_table: list[ConferenceRow]
    injury_report: InjuryReport

    @model_validator(mode="after")
    def _seeds(self):
        if list(self.seed_probabilities) != [str(k) for k in range(1, 16)]:
            raise ValueError("seed_probabilities must have keys '1'..'15'")
        _sums_to_one(self.seed_probabilities.values(), "seed_probabilities")
        return self


def _thirty(team_context: dict) -> None:
    if len(team_context) != 30:
        raise ValueError(f"team_context has {len(team_context)} teams, expected 30")


class Predictions(_TeamPage):
    metrics: Metrics
    team_context: dict[str, TeamContext]

    @model_validator(mode="after")
    def _thirty_teams(self):
        _thirty(self.team_context)
        return self


class TeamMetrics(BaseModel):
    record: ModelRecord


class TeamFile(_TeamPage):
    metrics: TeamMetrics


class League(BaseModel):
    generated_at: str
    model_version: str
    season: str
    completed_games: int = Field(ge=0)
    team_context: dict[str, TeamContext]

    @model_validator(mode="after")
    def _thirty_teams(self):
        _thirty(self.team_context)
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
    availability_note: str


class PregameEntry(BaseModel):
    home_win_prob: Prob | None
    pred_margin: float | None
    logged_at: str


PregameLog = TypeAdapter(dict[str, PregameEntry])


class SlateGame(BaseModel):
    game_id: str
    date: str
    home: str
    away: str
    home_record: str
    away_record: str
    completed: bool
    home_win_prob: Prob | None
    home_win_prob_healthy: Prob | None
    pred_margin: float | None
    pick: str | None
    home_pts: int | None
    away_pts: int | None
    pregame_home_win_prob: Prob | None
    correct: bool | None
    drivers: list[Driver] | None
    injuries: dict[str, list[InjuryPlayer]] | None


class SlateDay(BaseModel):
    date: str
    games: list[SlateGame]


class RecordCount(BaseModel):
    correct: int = Field(ge=0)
    wrong: int = Field(ge=0)


class SlateRecord(BaseModel):
    since: str
    yesterday: RecordCount
    total: RecordCount


class Slate(BaseModel):
    generated_at: str
    model_version: str
    days: list[SlateDay]
    record: SlateRecord


class DtdCase(BaseModel):
    date: str
    team: str
    player_id: int
    name: str
    played: bool | None


DtdLog = TypeAdapter(list[DtdCase])
