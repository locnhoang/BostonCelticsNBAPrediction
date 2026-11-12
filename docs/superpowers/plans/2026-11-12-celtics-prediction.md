# Celtics 2026-27 Prediction Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a daily-updating NBA model that predicts the Boston Celtics' remaining 2026-27 games, final win total, seed, and playoff/play-in odds, published as a static website.

**Architecture:** A Python package (`src/celtics`) fetches NBA data with `nba_api` and the NBA CDN schedule, caches it as parquet, builds point-in-time features (Elo, decayed net rating, rest), fits a point-margin regression, and runs a 10,000-run Monte Carlo of the full league's remaining schedule (standings, tiebreakers, play-in). It writes validated JSON to `site/data/`; a static HTML/JS page on GitHub Pages renders it. GitHub Actions runs the update daily, and a macOS `launchd` job is the fallback.

**Tech Stack:** Python 3.12, uv, pandas, pyarrow, numpy, scipy, scikit-learn (incl. HistGradientBoosting), joblib, pydantic v2, requests, nba_api, pytest, ruff. Frontend: plain HTML/CSS/JS, d3 7.9.0 and Observable Plot 0.6.16 from jsDelivr.

**Spec:** `docs/superpowers/specs/2026-11-12-celtics-prediction-design.md`

## Global Constraints

- Python `>=3.12`, managed with `uv`. All commands run as `uv run ...`.
- Data coverage: regular-season games from 2014-15 (season `2014`) through the current season. Season `2014` is Elo/rating warm-up only; model training starts at season `2015`.
- Seasons are identified by start year as `int` (2026 = 2026-27). Label format: `"2026-27"`.
- Regular-season games are exactly those whose `game_id` starts with `"002"` (this excludes the NBA Cup final, preseason, and playoffs).
- All model features are point-in-time: computed only from games before the game being predicted.
- Win probability = `Φ(margin / σ)`; σ = residual std of the fitted model.
- Simulation: 10,000 runs by default, full league, correlated team-strength offsets.
- Tiebreakers (simplified, documented): win % → head-to-head among tied teams → division winner → conference win % → random.
- Backtest test seasons: 2018 through 2025. Primary metric: log loss.
- Website: no NBA or team logos or other trademarked branding; Celtics green (`#007a33` light / `#3fae6a` dark) is the accent color only.
- CI never calls NBA.com; all tests run offline.
- The model is refit when the saved model is ≥7 days old (or when `--force-train` is passed).
- Before every commit run `uv run ruff check --fix . && uv run ruff check .`; this fixes import ordering automatically.

## File Map

```
pyproject.toml                      # deps, entry point, ruff/pytest config
.gitignore
src/celtics/__init__.py
src/celtics/config.py               # constants, Paths, season helpers
src/celtics/teams.py                # team / division / conference metadata
src/celtics/data/__init__.py
src/celtics/data/parse.py           # raw nba_api / CDN payloads → canonical games frame
src/celtics/data/fetch.py           # network calls + retries
src/celtics/data/cache.py           # parquet cache, refresh orchestration
src/celtics/data/validate.py        # data integrity checks
src/celtics/features/__init__.py
src/celtics/features/elo.py         # margin-of-victory Elo
src/celtics/features/ratings.py     # decayed net rating with prior; rest features
src/celtics/features/build.py       # feature matrix, as_of view, team state, matchup matrix
src/celtics/models/__init__.py
src/celtics/models/margin.py        # HomeCourt / Elo / Ridge / GBM margin models
src/celtics/evaluate/__init__.py
src/celtics/evaluate/metrics.py     # log loss, Brier, accuracy, calibration
src/celtics/evaluate/backtest.py    # walk-forward backtest, season coverage
src/celtics/simulate/__init__.py
src/celtics/simulate/inputs.py      # SimInputs from features + model
src/celtics/simulate/standings.py   # division winners, conference ranking
src/celtics/simulate/playin.py      # play-in bracket
src/celtics/simulate/season.py      # Monte Carlo engine
src/celtics/export/__init__.py
src/celtics/export/schema.py        # pydantic models for every JSON file
src/celtics/export/build.py         # JSON builders + atomic writer
src/celtics/pipeline.py             # run_update, run_backtest, load_or_train
src/celtics/cli.py                  # `celtics` command
tests/__init__.py
tests/factories.py                  # hand-built games, raw payloads
tests/synth.py                      # synthetic full league generator
tests/test_*.py
site/index.html, site/styles.css, site/app.js
site/data/.gitkeep
.github/workflows/ci.yml, daily.yml, pages.yml
scripts/run_update.sh, scripts/install_launchd.sh, scripts/com.celtics.update.plist.template
README.md
```

**Canonical games frame** (used everywhere): columns `game_id: str, season: int, date: datetime64[ns], home_team: str, away_team: str, home_pts: float, away_pts: float, poss: float`. A game is completed iff `home_pts` is not NaN. `poss` is the average of the two teams' possession estimates `FGA - OREB + TOV + 0.44*FTA`.

---

### Task 1: Project scaffolding, config, team metadata, CI

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/celtics/__init__.py`, `src/celtics/config.py`, `src/celtics/teams.py`, `tests/__init__.py`, `tests/test_config.py`, `.github/workflows/ci.yml`, `site/data/.gitkeep`

**Interfaces:**
- Produces: `celtics.config` constants listed below, `Paths`, `season_label(int) -> str`, `current_season(date) -> int`; `celtics.teams.TEAMS: list[str]` (sorted, 30), `TEAM_INDEX: dict[str,int]`, `DIVISION_OF`, `CONFERENCE_OF`, `IS_EAST: np.ndarray[bool]`, `CONFERENCE_MEMBERS: list[list[int]]` (East, West), `DIVISION_MEMBERS: list[list[int]]`.

- [ ] **Step 1: Install tooling**

Run:
```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
```
Expected: `uv --version` prints a version.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "celtics"
version = "0.1.0"
description = "Boston Celtics 2026-27 season prediction model"
requires-python = ">=3.12"
dependencies = [
    "nba_api>=1.10",
    "pandas>=2.2",
    "pyarrow>=17",
    "numpy>=2.0",
    "scipy>=1.14",
    "scikit-learn>=1.5",
    "joblib>=1.4",
    "pydantic>=2.9",
    "requests>=2.32",
]

[project.scripts]
celtics = "celtics.cli:main"

[dependency-groups]
dev = ["pytest>=8.3", "ruff>=0.7"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/celtics"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 3: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
/data/
.DS_Store
```

- [ ] **Step 4: Write the failing test `tests/test_config.py`** (also create empty `tests/__init__.py` and `src/celtics/__init__.py`)

```python
from datetime import date

from celtics.config import Paths, current_season, season_label
from celtics.teams import (
    CONFERENCE_MEMBERS,
    CONFERENCE_OF,
    DIVISION_MEMBERS,
    IS_EAST,
    TEAM_INDEX,
    TEAMS,
)


def test_season_label():
    assert season_label(2026) == "2026-27"
    assert season_label(2099) == "2099-00"


def test_current_season_rolls_over_in_august():
    assert current_season(date(2026, 11, 12)) == 2026
    assert current_season(date(2027, 3, 1)) == 2026
    assert current_season(date(2027, 8, 1)) == 2027


def test_thirty_teams_split_evenly():
    assert len(TEAMS) == 30 and TEAMS == sorted(TEAMS)
    assert TEAM_INDEX["BOS"] == TEAMS.index("BOS")
    assert sum(1 for t in TEAMS if CONFERENCE_OF[t] == "East") == 15
    assert IS_EAST.sum() == 15 and IS_EAST[TEAM_INDEX["BOS"]]
    assert [len(m) for m in CONFERENCE_MEMBERS] == [15, 15]
    assert sorted(len(m) for m in DIVISION_MEMBERS) == [5] * 6


def test_paths(tmp_path):
    p = Paths(tmp_path)
    assert p.raw_dir == tmp_path / "data" / "raw"
    assert p.model_dir == tmp_path / "data" / "models"
    assert p.site_data == tmp_path / "site" / "data"
```

- [ ] **Step 5: Run it to verify it fails**

Run: `uv sync && uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.config'`

- [ ] **Step 6: Write `src/celtics/config.py`**

```python
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
    def site_data(self) -> Path:
        return self.root / "site" / "data"


def season_label(season: int) -> str:
    return f"{season}-{(season + 1) % 100:02d}"


def current_season(today: date) -> int:
    return today.year if today.month >= 8 else today.year - 1
```

- [ ] **Step 7: Write `src/celtics/teams.py`**

```python
"""Static team metadata. Abbreviations are stable from 2014-15 onward."""

import numpy as np

DIVISIONS: dict[str, list[str]] = {
    "Atlantic": ["BOS", "BKN", "NYK", "PHI", "TOR"],
    "Central": ["CHI", "CLE", "DET", "IND", "MIL"],
    "Southeast": ["ATL", "CHA", "MIA", "ORL", "WAS"],
    "Northwest": ["DEN", "MIN", "OKC", "POR", "UTA"],
    "Pacific": ["GSW", "LAC", "LAL", "PHX", "SAC"],
    "Southwest": ["DAL", "HOU", "MEM", "NOP", "SAS"],
}
EAST_DIVISIONS = ("Atlantic", "Central", "Southeast")

TEAMS: list[str] = sorted(t for members in DIVISIONS.values() for t in members)
TEAM_INDEX: dict[str, int] = {t: i for i, t in enumerate(TEAMS)}
DIVISION_OF: dict[str, str] = {t: d for d, members in DIVISIONS.items() for t in members}
CONFERENCE_OF: dict[str, str] = {
    t: ("East" if d in EAST_DIVISIONS else "West") for t, d in DIVISION_OF.items()
}
IS_EAST = np.array([CONFERENCE_OF[t] == "East" for t in TEAMS])
CONFERENCE_MEMBERS: list[list[int]] = [
    [TEAM_INDEX[t] for t in TEAMS if CONFERENCE_OF[t] == conf] for conf in ("East", "West")
]
DIVISION_MEMBERS: list[list[int]] = [
    [TEAM_INDEX[t] for t in members] for members in DIVISIONS.values()
]
```

- [ ] **Step 8: Run tests and lint**

Run: `uv run pytest tests/test_config.py -v && uv run ruff check .`
Expected: 4 passed; `All checks passed!`

- [ ] **Step 9: Write `.github/workflows/ci.yml`** and `site/data/.gitkeep` (empty)

```yaml
name: CI
on:
  push:
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run pytest -q
```

- [ ] **Step 10: Commit** (includes `uv.lock` created by `uv sync`)

```bash
git add pyproject.toml uv.lock .gitignore src tests .github site/data/.gitkeep
git commit -m "chore: scaffold celtics package, config, team metadata, CI"
```

---

### Task 2: Parse raw NBA payloads into the canonical games frame

**Files:**
- Create: `src/celtics/data/__init__.py` (empty), `src/celtics/data/parse.py`, `tests/factories.py`, `tests/test_parse.py`

**Interfaces:**
- Produces:
  - `GAME_COLUMNS: list[str]`, `empty_games() -> pd.DataFrame`
  - `gamelog_to_games(raw: pd.DataFrame, season: int) -> pd.DataFrame`. `raw` is the `LeagueGameLog` team-level frame (one row per team per game; columns `GAME_ID, GAME_DATE, MATCHUP, TEAM_ABBREVIATION, PTS, FGA, OREB, TOV, FTA`).
  - `schedule_to_games(raw: dict, season: int) -> pd.DataFrame`. `raw` is the CDN `scheduleLeagueV2.json`; returns unplayed rows (NaN pts/poss); raises `ValueError` if `leagueSchedule.seasonYear != season_label(season)`.
  - `merge_schedule_results(schedule, results) -> pd.DataFrame`. Results win on conflicts; schedule supplies the unplayed games.
  - `tests/factories.py`: `gamelog_rows(game_id, date, home, away, home_pts, away_pts) -> list[dict]`, `schedule_payload(season_label, games: list[tuple[game_id, date, home, away]]) -> dict`, `make_games(rows: list[tuple]) -> pd.DataFrame`

- [ ] **Step 1: Write `tests/factories.py`**

```python
"""Hand-built inputs for unit tests."""

import numpy as np
import pandas as pd

from celtics.data.parse import GAME_COLUMNS


def gamelog_rows(game_id, date, home, away, home_pts, away_pts):
    """Two LeagueGameLog rows (home + away) with stats that give 100 possessions each."""
    base = {"GAME_ID": game_id, "GAME_DATE": date, "FGA": 90, "OREB": 10, "TOV": 15, "FTA": 11.363636}
    return [
        {**base, "TEAM_ABBREVIATION": home, "MATCHUP": f"{home} vs. {away}", "PTS": home_pts},
        {**base, "TEAM_ABBREVIATION": away, "MATCHUP": f"{away} @ {home}", "PTS": away_pts},
    ]


def schedule_payload(season_label, games):
    """Minimal scheduleLeagueV2.json. games: (game_id, 'YYYY-MM-DD', home, away)."""
    days: dict[str, list] = {}
    for game_id, date, home, away in games:
        days.setdefault(date, []).append(
            {
                "gameId": game_id,
                "gameDateEst": f"{date}T00:00:00Z",
                "gameStatus": 1,
                "homeTeam": {"teamTricode": home, "score": 0},
                "awayTeam": {"teamTricode": away, "score": 0},
            }
        )
    return {
        "leagueSchedule": {
            "seasonYear": season_label,
            "gameDates": [{"gameDate": d, "games": g} for d, g in days.items()],
        }
    }


def make_games(rows):
    """rows: (game_id, season, 'YYYY-MM-DD', home, away, home_pts|None, away_pts|None)."""
    records = []
    for game_id, season, date, home, away, hp, ap in rows:
        played = hp is not None
        records.append(
            {
                "game_id": game_id,
                "season": season,
                "date": pd.Timestamp(date),
                "home_team": home,
                "away_team": away,
                "home_pts": float(hp) if played else np.nan,
                "away_pts": float(ap) if played else np.nan,
                "poss": 100.0 if played else np.nan,
            }
        )
    return pd.DataFrame(records, columns=GAME_COLUMNS)
```

- [ ] **Step 2: Write the failing tests `tests/test_parse.py`**

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_parse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.data.parse'`

- [ ] **Step 4: Write `src/celtics/data/parse.py`**

```python
"""Convert raw NBA payloads into the canonical games frame."""

import numpy as np
import pandas as pd

from celtics.config import season_label

GAME_COLUMNS = ["game_id", "season", "date", "home_team", "away_team", "home_pts", "away_pts", "poss"]


def empty_games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": pd.Series(dtype=str),
            "season": pd.Series(dtype=np.int64),
            "date": pd.Series(dtype="datetime64[ns]"),
            "home_team": pd.Series(dtype=str),
            "away_team": pd.Series(dtype=str),
            "home_pts": pd.Series(dtype=float),
            "away_pts": pd.Series(dtype=float),
            "poss": pd.Series(dtype=float),
        }
    )


def _sorted(games: pd.DataFrame) -> pd.DataFrame:
    return games.sort_values(["date", "game_id"]).reset_index(drop=True)[GAME_COLUMNS]


def gamelog_to_games(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    if raw.empty:
        return empty_games()
    df = raw.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    df = df[df["GAME_ID"].str.startswith("002")].copy()
    df["poss"] = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]
    is_home = df["MATCHUP"].str.contains(" vs. ", regex=False)
    home = df[is_home].set_index("GAME_ID")
    away = df[~is_home].set_index("GAME_ID")
    ids = home.index.intersection(away.index)
    home, away = home.loc[ids], away.loc[ids]
    games = pd.DataFrame(
        {
            "game_id": ids.astype(str),
            "season": season,
            "date": pd.to_datetime(home["GAME_DATE"]).dt.normalize().to_numpy(),
            "home_team": home["TEAM_ABBREVIATION"].to_numpy(),
            "away_team": away["TEAM_ABBREVIATION"].to_numpy(),
            "home_pts": home["PTS"].astype(float).to_numpy(),
            "away_pts": away["PTS"].astype(float).to_numpy(),
            "poss": ((home["poss"] + away["poss"]) / 2).astype(float).to_numpy(),
        }
    )
    return _sorted(games)


def schedule_to_games(raw: dict, season: int) -> pd.DataFrame:
    sched = raw["leagueSchedule"]
    if sched.get("seasonYear") != season_label(season):
        raise ValueError(
            f"schedule is for {sched.get('seasonYear')}, expected {season_label(season)}"
        )
    rows = []
    for day in sched["gameDates"]:
        for g in day["games"]:
            game_id = str(g["gameId"])
            home = g["homeTeam"].get("teamTricode") or ""
            away = g["awayTeam"].get("teamTricode") or ""
            if not game_id.startswith("002") or not home or not away:
                continue
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "date": pd.Timestamp(g["gameDateEst"][:10]),
                    "home_team": home,
                    "away_team": away,
                    "home_pts": np.nan,
                    "away_pts": np.nan,
                    "poss": np.nan,
                }
            )
    if not rows:
        return empty_games()
    return _sorted(pd.DataFrame(rows).drop_duplicates("game_id"))


def merge_schedule_results(schedule: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    unplayed = schedule[~schedule["game_id"].isin(results["game_id"])]
    frames = [f for f in (results, unplayed) if not f.empty]
    if not frames:
        return empty_games()
    return _sorted(pd.concat(frames, ignore_index=True))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_parse.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/celtics/data tests/factories.py tests/test_parse.py
git commit -m "feat: parse nba_api game logs and CDN schedule into canonical games frame"
```

---

### Task 3: Fetching with retries and the parquet cache

**Files:**
- Create: `src/celtics/data/fetch.py`, `src/celtics/data/cache.py`, `tests/test_cache.py`

**Interfaces:**
- Consumes: `gamelog_to_games`, `schedule_to_games`, `merge_schedule_results`, `empty_games`, `GAME_COLUMNS` (Task 2); `FIRST_SEASON`, `current_season` (Task 1).
- Produces:
  - `FetchBlockedError(RuntimeError)`
  - `with_retries(fn, attempts=4, base_delay=2.0, sleep=time.sleep)`
  - `fetch_season_gamelog(season: int) -> pd.DataFrame`, `fetch_schedule() -> dict`
  - `season_path(raw_dir: Path, season: int) -> Path`
  - `load_all(raw_dir: Path) -> pd.DataFrame` (canonical frame of every cached season)
  - `refresh(raw_dir, today, fetch_gamelog=fetch_season_gamelog, fetch_schedule=fetch_schedule, pause=1.5) -> pd.DataFrame`. Fetches missing past seasons once, always refetches the current season, and returns `load_all`.

- [ ] **Step 1: Write the failing tests `tests/test_cache.py`**

```python
from datetime import date

import pandas as pd
import pytest

from celtics.data.cache import load_all, refresh, season_path
from celtics.data.fetch import FetchBlockedError, with_retries
from tests.factories import gamelog_rows, schedule_payload


def test_with_retries_succeeds_after_failures():
    calls = {"n": 0}
    sleeps = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("nope")
        return "ok"

    assert with_retries(flaky, attempts=4, base_delay=1.0, sleep=sleeps.append) == "ok"
    assert sleeps == [1.0, 2.0]


def test_with_retries_raises_fetch_blocked():
    def always_fails():
        raise TimeoutError("blocked")

    with pytest.raises(FetchBlockedError, match="blocked"):
        with_retries(always_fails, attempts=2, base_delay=0.0, sleep=lambda s: None)


def _fake_gamelog(season):
    gid = f"002{season % 100:02d}00001"
    return pd.DataFrame(gamelog_rows(gid, f"{season}-11-01", "BOS", "NYK", 110, 100))


def test_refresh_fetches_past_seasons_once_and_current_always(tmp_path, monkeypatch):
    monkeypatch.setattr("celtics.data.cache.FIRST_SEASON", 2024)
    fetched = []

    def fake_gamelog(season):
        fetched.append(season)
        return _fake_gamelog(season)

    def fake_schedule():
        return schedule_payload(
            "2026-27",
            [("0022600001", "2026-11-01", "BOS", "NYK"), ("0022600002", "2026-11-20", "MIA", "BOS")],
        )

    kwargs = dict(fetch_gamelog=fake_gamelog, fetch_schedule=fake_schedule, pause=0)
    games = refresh(tmp_path, date(2026, 11, 12), **kwargs)
    assert fetched == [2024, 2025, 2026]
    assert season_path(tmp_path, 2025).exists()
    assert games["season"].tolist() == [2024, 2025, 2026, 2026]
    assert games["home_pts"].isna().sum() == 1  # the unplayed Nov 20 game

    fetched.clear()
    refresh(tmp_path, date(2026, 11, 13), **kwargs)
    assert fetched == [2026]


def test_load_all_empty_dir(tmp_path):
    assert load_all(tmp_path).empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.data.cache'`

- [ ] **Step 3: Write `src/celtics/data/fetch.py`**

```python
"""Network access to NBA data. Everything here is retried and fails loudly."""

import time
from collections.abc import Callable

import pandas as pd
import requests

from celtics.config import season_label

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"


class FetchBlockedError(RuntimeError):
    """NBA data sources refused, rate-limited, or timed out after all retries."""


def with_retries[T](
    fn: Callable[[], T],
    attempts: int = 4,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # network stacks raise many exception types
            last = exc
            if i < attempts - 1:
                sleep(base_delay * 2**i)
    raise FetchBlockedError(f"fetch blocked or failing after {attempts} attempts: {last}") from last


def fetch_season_gamelog(season: int) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    def call() -> pd.DataFrame:
        endpoint = leaguegamelog.LeagueGameLog(
            season=season_label(season),
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="T",
            timeout=30,
        )
        return endpoint.get_data_frames()[0]

    return with_retries(call)


def fetch_schedule() -> dict:
    def call() -> dict:
        resp = requests.get(SCHEDULE_URL, timeout=30)
        resp.raise_for_status()
        return resp.json()

    return with_retries(call)
```

- [ ] **Step 4: Write `src/celtics/data/cache.py`**

```python
"""Local parquet cache: one file per season."""

import time
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd

from celtics.config import FIRST_SEASON, current_season
from celtics.data.fetch import fetch_schedule as _fetch_schedule
from celtics.data.fetch import fetch_season_gamelog
from celtics.data.parse import (
    GAME_COLUMNS,
    empty_games,
    gamelog_to_games,
    merge_schedule_results,
    schedule_to_games,
)


def season_path(raw_dir: Path, season: int) -> Path:
    return Path(raw_dir) / f"games_{season}.parquet"


def load_all(raw_dir: Path) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("games_*.parquet"))
    if not files:
        return empty_games()
    games = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    games["date"] = pd.to_datetime(games["date"])
    games["season"] = games["season"].astype("int64")
    return games.sort_values(["date", "game_id"]).reset_index(drop=True)[GAME_COLUMNS]


def refresh(
    raw_dir: Path,
    today: date,
    fetch_gamelog: Callable[[int], pd.DataFrame] = fetch_season_gamelog,
    fetch_schedule: Callable[[], dict] = _fetch_schedule,
    pause: float = 1.5,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    season = current_season(today)
    for past in range(FIRST_SEASON, season):
        path = season_path(raw_dir, past)
        if not path.exists():
            gamelog_to_games(fetch_gamelog(past), past).to_parquet(path, index=False)
            time.sleep(pause)
    results = gamelog_to_games(fetch_gamelog(season), season)
    schedule = schedule_to_games(fetch_schedule(), season)
    merge_schedule_results(schedule, results).to_parquet(season_path(raw_dir, season), index=False)
    return load_all(raw_dir)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cache.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/celtics/data/fetch.py src/celtics/data/cache.py tests/test_cache.py
git commit -m "feat: fetch NBA data with retries and cache seasons as parquet"
```

---

### Task 4: Data validation

**Files:**
- Create: `src/celtics/data/validate.py`, `tests/test_validate.py`

**Interfaces:**
- Consumes: `TEAMS` (Task 1), `make_games` (Task 2 factories).
- Produces: `DataValidationError(ValueError)`, `validate_games(games: pd.DataFrame, season: int, min_games: int = 80, max_games: int = 82) -> None`. Raises with every problem joined by `"; "`.

The rule is 80–82 games per team in the current season because the NBA adds each team's last 2 games (decided by NBA Cup results) in December. Missing games are filled in during simulation (Task 10).

- [ ] **Step 1: Write the failing tests `tests/test_validate.py`**

```python
import numpy as np
import pandas as pd
import pytest

from celtics.data.validate import DataValidationError, validate_games
from celtics.teams import TEAMS


def league_season(season=2026, games_per_team=82, played=10):
    """Round-robin-ish schedule where every team plays games_per_team games."""
    rows = []
    rng = np.random.default_rng(0)
    start = pd.Timestamp(f"{season}-10-20")
    for day in range(games_per_team):
        order = rng.permutation(TEAMS)
        for k in range(15):
            done = day < played
            rows.append(
                {
                    "game_id": f"002{season % 100:02d}{day:03d}{k:02d}",
                    "season": season,
                    "date": start + pd.Timedelta(days=2 * day),
                    "home_team": order[2 * k],
                    "away_team": order[2 * k + 1],
                    "home_pts": 110.0 if done else np.nan,
                    "away_pts": 100.0 if done else np.nan,
                    "poss": 100.0 if done else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_valid_season_passes():
    validate_games(league_season(), 2026)


def test_duplicate_ids_fail():
    g = league_season()
    g = pd.concat([g, g.iloc[[0]]])
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_games(g, 2026)


def test_unknown_team_fails():
    g = league_season()
    g.loc[0, "home_team"] = "SEA"
    with pytest.raises(DataValidationError, match="unknown teams"):
        validate_games(g, 2026)


def test_completed_game_missing_possessions_fails():
    g = league_season()
    g.loc[0, "poss"] = np.nan
    with pytest.raises(DataValidationError, match="missing"):
        validate_games(g, 2026)


def test_too_few_games_fails():
    with pytest.raises(DataValidationError, match="games per team"):
        validate_games(league_season(games_per_team=70), 2026)


def test_eighty_games_allowed_before_cup_games_are_scheduled():
    validate_games(league_season(games_per_team=80), 2026)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_validate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.data.validate'`

- [ ] **Step 3: Write `src/celtics/data/validate.py`**

```python
"""Integrity checks run before any modeling. A failure aborts the run."""

import pandas as pd

from celtics.teams import TEAMS


class DataValidationError(ValueError):
    """The fetched data is inconsistent; nothing should be published."""


def validate_games(
    games: pd.DataFrame, season: int, min_games: int = 80, max_games: int = 82
) -> None:
    problems: list[str] = []

    dup = games["game_id"].duplicated()
    if dup.any():
        problems.append(f"duplicate game_ids: {games.loc[dup, 'game_id'].tolist()[:5]}")

    seen = set(games["home_team"]) | set(games["away_team"])
    unknown = seen - set(TEAMS)
    if unknown:
        problems.append(f"unknown teams: {sorted(unknown)}")

    played = games["home_pts"].notna() | games["away_pts"].notna()
    incomplete = games[played & games[["home_pts", "away_pts", "poss"]].isna().any(axis=1)]
    if len(incomplete):
        problems.append(f"completed games missing scores/possessions: {incomplete['game_id'].tolist()[:5]}")

    cur = games[games["season"] == season]
    cur_teams = set(cur["home_team"]) | set(cur["away_team"])
    if cur_teams != set(TEAMS):
        problems.append(f"season {season} has {len(cur_teams)} teams, expected 30")
    counts = pd.concat([cur["home_team"], cur["away_team"]]).value_counts()
    bad = counts[(counts < min_games) | (counts > max_games)]
    if len(bad):
        problems.append(f"games per team out of [{min_games}, {max_games}]: {bad.to_dict()}")

    if problems:
        raise DataValidationError("; ".join(problems))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_validate.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/celtics/data/validate.py tests/test_validate.py
git commit -m "feat: validate games data before modeling"
```

---

### Task 5: Margin-of-victory Elo

**Files:**
- Create: `src/celtics/features/__init__.py` (empty), `src/celtics/features/elo.py`, `tests/test_elo.py`

**Interfaces:**
- Consumes: canonical games frame; `ELO_*` constants.
- Produces: `compute_elo(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]`. The frame has columns `game_id, home_elo_pre, away_elo_pre` (one row per game). The dict holds each team's rating after the last completed game, with between-season carryover already applied if the latest season has started (has any row). Unplayed games get pre-game ratings but trigger no update.

- [ ] **Step 1: Write the failing tests `tests/test_elo.py`**

```python
import pytest

from celtics.features.elo import compute_elo
from tests.factories import make_games


def test_first_game_update_matches_formula():
    games = make_games([("g1", 2020, "2020-12-22", "BOS", "NYK", 110, 100)])
    pre, ratings = compute_elo(games)
    assert pre.loc[0, "home_elo_pre"] == 1500 and pre.loc[0, "away_elo_pre"] == 1500
    expected_home = 1 / (1 + 10 ** (-100 / 400))
    multiplier = (13**0.8) / (7.5 + 0.006 * 100)
    delta = 20 * multiplier * (1 - expected_home)
    assert ratings["BOS"] == pytest.approx(1500 + delta)
    assert ratings["NYK"] == pytest.approx(1500 - delta)


def test_updates_are_zero_sum_and_used_as_next_pre_rating():
    games = make_games(
        [
            ("g1", 2020, "2020-12-22", "BOS", "NYK", 110, 100),
            ("g2", 2020, "2020-12-24", "NYK", "BOS", 90, 120),
        ]
    )
    pre, ratings = compute_elo(games)
    assert pre.loc[1, "away_elo_pre"] > 1500  # BOS carries its win into game 2
    assert sum(ratings.values()) == pytest.approx(3000)


def test_unplayed_game_has_pre_rating_but_no_update():
    games = make_games(
        [
            ("g1", 2020, "2020-12-22", "BOS", "NYK", 110, 100),
            ("g2", 2020, "2020-12-24", "BOS", "NYK", None, None),
        ]
    )
    pre, ratings = compute_elo(games)
    assert pre.loc[1, "home_elo_pre"] == pytest.approx(ratings["BOS"])


def test_season_carryover_regresses_25_percent():
    rows = [("g1", 2020, "2021-01-01", "BOS", "NYK", 150, 50)]
    rows.append(("g2", 2021, "2021-10-20", "BOS", "NYK", None, None))
    pre, ratings = compute_elo(make_games(rows))
    _, end_2020 = compute_elo(make_games(rows[:1]))
    assert pre.loc[1, "home_elo_pre"] == pytest.approx(0.75 * end_2020["BOS"] + 0.25 * 1500)
    assert ratings["BOS"] == pytest.approx(pre.loc[1, "home_elo_pre"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_elo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.features.elo'`

- [ ] **Step 3: Write `src/celtics/features/elo.py`**

```python
"""Margin-of-victory Elo (FiveThirtyEight-style)."""

from collections import defaultdict

import numpy as np
import pandas as pd

from celtics.config import ELO_CARRYOVER, ELO_HOME, ELO_K, ELO_MEAN


def expected_home(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(home_elo + ELO_HOME - away_elo) / 400))


def mov_multiplier(mov: float, winner_elo_edge: float) -> float:
    return ((abs(mov) + 3) ** 0.8) / (7.5 + 0.006 * winner_elo_edge)


def compute_elo(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    ratings: defaultdict[str, float] = defaultdict(lambda: ELO_MEAN)
    ordered = games.sort_values(["date", "game_id"])
    season = None
    out_ids, out_home, out_away = [], [], []
    for row in ordered.itertuples(index=False):
        if row.season != season:
            if season is not None:
                for team in list(ratings):
                    ratings[team] = ELO_CARRYOVER * ratings[team] + (1 - ELO_CARRYOVER) * ELO_MEAN
            season = row.season
        h, a = ratings[row.home_team], ratings[row.away_team]
        out_ids.append(row.game_id)
        out_home.append(h)
        out_away.append(a)
        if np.isnan(row.home_pts):
            continue
        mov = row.home_pts - row.away_pts
        exp = expected_home(h, a)
        home_won = mov > 0
        edge = (h + ELO_HOME - a) if home_won else (a - h - ELO_HOME)
        delta = ELO_K * mov_multiplier(mov, edge) * ((1.0 if home_won else 0.0) - exp)
        ratings[row.home_team] = h + delta
        ratings[row.away_team] = a - delta
    pre = pd.DataFrame({"game_id": out_ids, "home_elo_pre": out_home, "away_elo_pre": out_away})
    return pre, dict(ratings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_elo.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/celtics/features tests/test_elo.py
git commit -m "feat: margin-of-victory Elo with season carryover"
```

---

### Task 6: Decayed net rating with prior, and rest features

**Files:**
- Create: `src/celtics/features/ratings.py`, `tests/test_ratings.py`

**Interfaces:**
- Consumes: canonical games frame; `NET_*`, `REST_CAP` constants.
- Produces:
  - `compute_net_ratings(games) -> tuple[pd.DataFrame, dict[str, float]]`. The frame has columns `game_id, home_net_pre, away_net_pre`. The dict holds each team's current estimate for the latest season.

    Estimate = `(w0·λⁿ·prior + Σ λ^(n-i)·xᵢ) / (w0·λⁿ + Σ λ^(n-i))`, where `xᵢ = 100·(pts − opp_pts)/poss` and `prior = NET_PRIOR_SHRINK × last season's average net rating` (0 if unknown).
  - `compute_rest(games) -> pd.DataFrame` with columns `game_id, home_rest, away_rest, home_b2b, away_b2b, home_3in4, away_3in4` (floats). Rest = days since the team's previous game in the same season, capped at 4; the first game of a season gets 4. `3in4` means the game is the team's third within four calendar days.

- [ ] **Step 1: Write the failing tests `tests/test_ratings.py`**

```python
import pytest

from celtics.features.ratings import compute_net_ratings, compute_rest
from tests.factories import make_games


def test_first_game_uses_prior_zero_then_updates():
    games = make_games(
        [
            ("g1", 2020, "2020-12-22", "BOS", "NYK", 110, 100),
            ("g2", 2020, "2020-12-24", "BOS", "MIA", None, None),
        ]
    )
    pre, current = compute_net_ratings(games)
    assert pre.loc[0, "home_net_pre"] == 0.0
    # x = +10 per 100; num = 0.97*0 + 10, den = 0.97*8 + 1
    expected = 10 / (0.97 * 8 + 1)
    assert pre.loc[1, "home_net_pre"] == pytest.approx(expected)
    assert current["BOS"] == pytest.approx(expected)
    assert current["NYK"] == pytest.approx(-expected)


def test_prior_comes_from_last_season_average():
    games = make_games(
        [
            ("g1", 2020, "2021-01-01", "BOS", "NYK", 110, 100),
            ("g2", 2021, "2021-10-20", "BOS", "NYK", None, None),
        ]
    )
    pre, _ = compute_net_ratings(games)
    assert pre.loc[1, "home_net_pre"] == pytest.approx(0.67 * 10)
    assert pre.loc[1, "away_net_pre"] == pytest.approx(0.67 * -10)


def test_rest_back_to_back_and_three_in_four():
    games = make_games(
        [
            ("g1", 2020, "2020-12-01", "BOS", "NYK", 100, 90),
            ("g2", 2020, "2020-12-02", "MIA", "BOS", 100, 90),
            ("g3", 2020, "2020-12-04", "BOS", "CHI", 100, 90),
            ("g4", 2020, "2020-12-05", "BOS", "ATL", None, None),
        ]
    )
    rest = compute_rest(games).set_index("game_id")
    assert rest.loc["g1", "home_rest"] == 4  # season opener
    assert rest.loc["g2", "away_rest"] == 1 and rest.loc["g2", "away_b2b"] == 1
    assert rest.loc["g3", "home_rest"] == 2 and rest.loc["g3", "home_b2b"] == 0
    assert rest.loc["g4", "home_b2b"] == 1
    assert rest.loc["g4", "home_3in4"] == 1  # Dec 2, 4, 5
    assert rest.loc["g3", "home_3in4"] == 1  # Dec 1, 2, 4
    assert rest.loc["g2", "home_3in4"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ratings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.features.ratings'`

- [ ] **Step 3: Write `src/celtics/features/ratings.py`**

```python
"""Exponentially decayed net rating with a preseason prior, plus rest features."""

from collections import defaultdict

import numpy as np
import pandas as pd

from celtics.config import NET_DECAY, NET_PRIOR_SHRINK, NET_PRIOR_WEIGHT, REST_CAP


def _estimate(state: dict[str, float]) -> float:
    return state["num"] / state["den"] if state["den"] > 0 else 0.0


def compute_net_ratings(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    ordered = games.sort_values(["date", "game_id"])
    season = None
    state: dict[str, dict[str, float]] = {}
    totals: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    prev_avg: dict[str, float] = {}
    ids, home_pre, away_pre = [], [], []
    for row in ordered.itertuples(index=False):
        if row.season != season:
            prev_avg = {t: s / n for t, (s, n) in totals.items() if n > 0}
            totals = defaultdict(lambda: [0.0, 0.0])
            state = {}
            season = row.season
        for team in (row.home_team, row.away_team):
            if team not in state:
                prior = NET_PRIOR_SHRINK * prev_avg.get(team, 0.0)
                state[team] = {"num": NET_PRIOR_WEIGHT * prior, "den": NET_PRIOR_WEIGHT}
        ids.append(row.game_id)
        home_pre.append(_estimate(state[row.home_team]))
        away_pre.append(_estimate(state[row.away_team]))
        if np.isnan(row.home_pts):
            continue
        x = 100.0 * (row.home_pts - row.away_pts) / row.poss
        for team, value in ((row.home_team, x), (row.away_team, -x)):
            s = state[team]
            s["num"] = NET_DECAY * s["num"] + value
            s["den"] = NET_DECAY * s["den"] + 1.0
            totals[team][0] += value
            totals[team][1] += 1.0
    pre = pd.DataFrame({"game_id": ids, "home_net_pre": home_pre, "away_net_pre": away_pre})
    return pre, {t: _estimate(s) for t, s in state.items()}


def compute_rest(games: pd.DataFrame) -> pd.DataFrame:
    base = games[["game_id", "season", "date"]]
    long = pd.concat(
        [
            base.assign(team=games["home_team"], side="home"),
            base.assign(team=games["away_team"], side="away"),
        ],
        ignore_index=True,
    ).sort_values(["team", "date", "game_id"])
    grouped = long.groupby(["season", "team"])["date"]
    prev1, prev2 = grouped.shift(1), grouped.shift(2)
    long["rest"] = (long["date"] - prev1).dt.days.clip(upper=REST_CAP).fillna(REST_CAP).astype(float)
    long["b2b"] = (long["rest"] == 1).astype(float)
    long["three_in_four"] = ((long["date"] - prev2).dt.days <= 3).astype(float)
    wide = long.pivot(index="game_id", columns="side", values=["rest", "b2b", "three_in_four"])
    return pd.DataFrame(
        {
            "game_id": wide.index,
            "home_rest": wide[("rest", "home")].to_numpy(),
            "away_rest": wide[("rest", "away")].to_numpy(),
            "home_b2b": wide[("b2b", "home")].to_numpy(),
            "away_b2b": wide[("b2b", "away")].to_numpy(),
            "home_3in4": wide[("three_in_four", "home")].to_numpy(),
            "away_3in4": wide[("three_in_four", "away")].to_numpy(),
        }
    ).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ratings.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/celtics/features/ratings.py tests/test_ratings.py
git commit -m "feat: decayed net rating with prior and rest features"
```

---

### Task 7: Feature matrix, as-of view, team state, matchup matrix, synthetic league

**Files:**
- Create: `src/celtics/features/build.py`, `tests/synth.py`, `tests/test_build.py`

**Interfaces:**
- Consumes: `compute_elo` (Task 5), `compute_net_ratings`, `compute_rest` (Task 6), `TEAMS`, `TEAM_INDEX`.
- Produces:
  - `FEATURES = ["elo_diff", "net_diff", "home_rest", "away_rest", "home_b2b", "away_b2b", "home_3in4", "away_3in4"]`
  - `build_features(games) -> pd.DataFrame`: the canonical columns plus `FEATURES`, `completed: bool`, `margin: float` (home − away; NaN if unplayed), sorted by date and game_id.
  - `as_of(games, cutoff: pd.Timestamp) -> pd.DataFrame`: a copy where games after `cutoff` are blanked (unplayed).
  - `@dataclass TeamState(elo: dict[str, float], net: dict[str, float])`, `current_team_state(games) -> TeamState`
  - `matchup_frame(state, pairs: list[tuple[str, str]]) -> pd.DataFrame` (columns `FEATURES`, default rest)
  - `matchup_matrix(predict: Callable[[pd.DataFrame], np.ndarray], state) -> np.ndarray` of shape (30, 30): `[i, j]` is the predicted margin for home team `TEAMS[i]` vs away team `TEAMS[j]`, with zeros on the diagonal.
  - `tests/synth.py`: `make_league(seasons, current=None, completed_rounds=15, seed=0) -> pd.DataFrame`. Each season has 82 rounds two days apart, and each round pairs all 30 teams into 15 games. Teams have fixed hidden strengths. In the `current` season only the first `completed_rounds` rounds are played.

- [ ] **Step 1: Write `tests/synth.py`**

```python
"""Synthetic full-league data for integration-style tests (no network)."""

import numpy as np
import pandas as pd

from celtics.data.parse import GAME_COLUMNS
from celtics.teams import TEAMS


def make_league(seasons, current=None, completed_rounds=15, seed=0):
    rng = np.random.default_rng(seed)
    strength = {t: rng.normal(0, 5) for t in TEAMS}
    rows = []
    for season in seasons:
        start = pd.Timestamp(f"{season}-10-20")
        n = 0
        for rnd in range(82):
            order = rng.permutation(TEAMS)
            played = season != current or rnd < completed_rounds
            for k in range(15):
                home, away = order[2 * k], order[2 * k + 1]
                margin = strength[home] - strength[away] + 2.5 + rng.normal(0, 12)
                hp = round(110 + margin / 2)
                ap = round(110 - margin / 2)
                if hp == ap:
                    hp += 1
                rows.append(
                    {
                        "game_id": f"002{season % 100:02d}{n:05d}",
                        "season": season,
                        "date": start + pd.Timedelta(days=2 * rnd),
                        "home_team": home,
                        "away_team": away,
                        "home_pts": float(hp) if played else np.nan,
                        "away_pts": float(ap) if played else np.nan,
                        "poss": 100.0 if played else np.nan,
                    }
                )
                n += 1
    return pd.DataFrame(rows, columns=GAME_COLUMNS)
```

- [ ] **Step 2: Write the failing tests `tests/test_build.py`**

```python
import numpy as np
import pandas as pd

from celtics.features.build import (
    FEATURES,
    as_of,
    build_features,
    current_team_state,
    matchup_frame,
    matchup_matrix,
)
from celtics.teams import TEAM_INDEX
from tests.synth import make_league


def test_build_features_shape_and_columns():
    games = make_league([2022, 2023], current=2023)
    feats = build_features(games)
    assert len(feats) == len(games)
    assert set(FEATURES) <= set(feats.columns)
    assert feats["completed"].sum() == 15 * 82 + 15 * 15
    assert feats.loc[feats["completed"], "margin"].notna().all()
    assert feats.loc[~feats["completed"], "margin"].isna().all()
    assert not feats[FEATURES].isna().any().any()


def test_no_future_leakage():
    games = make_league([2022, 2023])
    base = build_features(games)
    changed = games.copy()
    last = changed.index[-1]
    changed.loc[last, ["home_pts", "away_pts"]] = [200.0, 50.0]
    after = build_features(changed)
    # every game's pre-game features are unaffected by the final game's result
    pd.testing.assert_frame_equal(base[FEATURES], after[FEATURES])


def test_as_of_blanks_future_games():
    games = make_league([2022])
    cutoff = pd.Timestamp("2022-11-01")
    view = as_of(games, cutoff)
    assert view.loc[view["date"] > cutoff, "home_pts"].isna().all()
    assert view.loc[view["date"] <= cutoff, "home_pts"].notna().all()
    assert games["home_pts"].notna().all()  # original untouched


def test_team_state_and_matchup_matrix():
    games = make_league([2022, 2023], current=2023)
    state = current_team_state(games)
    assert len(state.elo) == 30 and len(state.net) == 30
    frame = matchup_frame(state, [("BOS", "NYK")])
    assert list(frame.columns) == FEATURES
    assert frame.loc[0, "elo_diff"] == state.elo["BOS"] - state.elo["NYK"]

    def predict(X):
        return X["elo_diff"].to_numpy() / 28 + 3

    m = matchup_matrix(predict, state)
    b, n = TEAM_INDEX["BOS"], TEAM_INDEX["NYK"]
    assert m.shape == (30, 30) and np.all(np.diag(m) == 0)
    assert m[b, n] == (state.elo["BOS"] - state.elo["NYK"]) / 28 + 3
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.features.build'`

- [ ] **Step 4: Write `src/celtics/features/build.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_build.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/celtics/features/build.py tests/synth.py tests/test_build.py
git commit -m "feat: feature matrix, as-of view, team state, matchup matrix"
```

---

### Task 8: Margin models (home court, Elo, ridge, gradient boosting)

**Files:**
- Create: `src/celtics/models/__init__.py` (empty), `src/celtics/models/margin.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: `FEATURES` (Task 7); `ELO_HOME`, `ELO_PER_POINT`.
- Produces:
  - `win_prob(margin: np.ndarray | float, sigma: float) -> np.ndarray`
  - `class MarginModel` with `name: str`, `sigma: float`, `fit(X: pd.DataFrame, y: pd.Series) -> self` (sets `sigma` to the residual std, ddof=1), `predict_margin(X) -> np.ndarray`, `win_prob(X) -> np.ndarray`
  - Subclasses `HomeCourtModel` (`"homecourt"`), `EloModel` (`"elo"`), `RidgeModel` (`"ridge"`), `GBMModel` (`"gbm"`, scikit-learn HistGradientBoostingRegressor)
  - `MODELS: dict[str, type[MarginModel]]`, `make_model(name) -> MarginModel`, `save_model(model, path)`, `load_model(path) -> MarginModel`

- [ ] **Step 1: Write the failing tests `tests/test_models.py`**

```python
import numpy as np
import pandas as pd
import pytest

from celtics.features.build import FEATURES
from celtics.models.margin import MODELS, load_model, make_model, save_model, win_prob


def synthetic_xy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(0.0, index=range(n), columns=FEATURES)
    X["elo_diff"] = rng.normal(0, 150, n)
    X["net_diff"] = X["elo_diff"] / 28 + rng.normal(0, 2, n)
    X["home_rest"] = rng.integers(1, 5, n).astype(float)
    X["away_rest"] = rng.integers(1, 5, n).astype(float)
    y = pd.Series((X["elo_diff"] + 100) / 28 + rng.normal(0, 12, n))
    return X, y


def test_win_prob_symmetry():
    assert win_prob(0.0, 12.0) == pytest.approx(0.5)
    assert win_prob(5.0, 12.0) + win_prob(-5.0, 12.0) == pytest.approx(1.0)


def test_homecourt_predicts_mean_margin():
    X, y = synthetic_xy()
    m = make_model("homecourt").fit(X, y)
    assert np.allclose(m.predict_margin(X), y.mean())
    assert m.sigma == pytest.approx(y.std(), rel=1e-6)


def test_elo_model_formula():
    X, y = synthetic_xy()
    m = make_model("elo").fit(X, y)
    assert np.allclose(m.predict_margin(X), (X["elo_diff"] + 100) / 28)
    assert m.sigma == pytest.approx(12, rel=0.05)


@pytest.mark.parametrize("name", ["ridge", "gbm"])
def test_learned_models_recover_signal(name):
    X, y = synthetic_xy()
    m = make_model(name).fit(X, y)
    truth = (X["elo_diff"] + 100) / 28
    assert np.corrcoef(m.predict_margin(X), truth)[0, 1] > 0.9
    assert 11 < m.sigma < 13
    p = m.win_prob(X)
    assert ((p > 0) & (p < 1)).all()


def test_registry_and_roundtrip(tmp_path):
    assert set(MODELS) == {"homecourt", "elo", "ridge", "gbm"}
    X, y = synthetic_xy(500)
    m = make_model("ridge").fit(X, y)
    path = tmp_path / "ridge.joblib"
    save_model(m, path)
    loaded = load_model(path)
    assert np.allclose(loaded.predict_margin(X), m.predict_margin(X))
    assert loaded.sigma == m.sigma
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.models.margin'`

- [ ] **Step 3: Write `src/celtics/models/margin.py`**

```python
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
from celtics.features.build import FEATURES


def win_prob(margin, sigma: float) -> np.ndarray:
    return norm.cdf(np.asarray(margin, dtype=float) / sigma)


class MarginModel:
    name = "base"

    def __init__(self) -> None:
        self.sigma = 12.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MarginModel":
        self._fit(X[FEATURES], y)
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

    def _fit(self, X, y):
        self.mean_ = float(np.mean(y))

    def predict_margin(self, X):
        return np.full(len(X), self.mean_)


class EloModel(MarginModel):
    name = "elo"

    def _fit(self, X, y):
        pass

    def predict_margin(self, X):
        return (X["elo_diff"].to_numpy(dtype=float) + ELO_HOME) / ELO_PER_POINT


class RidgeModel(MarginModel):
    name = "ridge"

    def _fit(self, X, y):
        self.pipe_ = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)

    def predict_margin(self, X):
        return self.pipe_.predict(X[FEATURES])


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
        return self.model_.predict(X[FEATURES])


MODELS: dict[str, type[MarginModel]] = {
    "homecourt": HomeCourtModel,
    "elo": EloModel,
    "ridge": RidgeModel,
    "gbm": GBMModel,
}


def make_model(name: str) -> MarginModel:
    return MODELS[name]()


def save_model(model: MarginModel, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> MarginModel:
    return joblib.load(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/celtics/models tests/test_models.py
git commit -m "feat: home-court, Elo, ridge and gradient-boosting margin models"
```

---

### Task 9: Metrics and walk-forward backtest

**Files:**
- Create: `src/celtics/evaluate/__init__.py` (empty), `src/celtics/evaluate/metrics.py`, `src/celtics/evaluate/backtest.py`, `tests/test_metrics.py`, `tests/test_backtest.py`

**Interfaces:**
- Consumes: `build_features`, `FEATURES` (Task 7); `make_model` (Task 8); `FIRST_TRAIN_SEASON`.
- Produces:
  - `log_loss(p, y) -> float`, `brier(p, y) -> float`, `accuracy(p, y) -> float`
  - `calibration_table(p, y, n_bins=10) -> list[dict]` with keys `bin_lo, bin_hi, mean_pred, observed, n` (empty bins skipped)
  - `summarize(p, y) -> dict` with keys `n, log_loss, brier, accuracy`
  - `walk_forward(features, model_name, test_seasons) -> pd.DataFrame` with columns `game_id, season, prob, outcome` (outcome 1.0 if the home team won). For each test season, trains on completed games with `FIRST_TRAIN_SEASON <= season < test_season`.

- [ ] **Step 1: Write the failing tests `tests/test_metrics.py`**

```python
import math

import numpy as np
import pytest

from celtics.evaluate.metrics import accuracy, brier, calibration_table, log_loss, summarize


def test_coin_flip_scores():
    p = np.full(4, 0.5)
    y = np.array([1, 0, 1, 0])
    assert log_loss(p, y) == pytest.approx(math.log(2))
    assert brier(p, y) == pytest.approx(0.25)


def test_accuracy_threshold():
    assert accuracy(np.array([0.9, 0.2, 0.6]), np.array([1, 0, 0])) == pytest.approx(2 / 3)


def test_log_loss_clips_extremes():
    assert np.isfinite(log_loss(np.array([1.0, 0.0]), np.array([0, 1])))


def test_calibration_table_bins():
    p = np.array([0.05, 0.07, 0.95, 0.96])
    y = np.array([0, 1, 1, 1])
    table = calibration_table(p, y, n_bins=10)
    assert [row["n"] for row in table] == [2, 2]
    assert table[0]["observed"] == pytest.approx(0.5)
    assert table[1]["mean_pred"] == pytest.approx(0.955)


def test_summarize_keys():
    s = summarize(np.array([0.7, 0.3]), np.array([1, 0]))
    assert set(s) == {"n", "log_loss", "brier", "accuracy"} and s["n"] == 2
```

- [ ] **Step 2: Write the failing tests `tests/test_backtest.py`**

```python
import celtics.evaluate.backtest as bt
from celtics.features.build import build_features
from celtics.models.margin import make_model
from tests.synth import make_league


def test_walk_forward_only_trains_on_prior_seasons(monkeypatch):
    feats = build_features(make_league([2020, 2021, 2022, 2023]))
    seen = []

    def spy(name):
        model = make_model(name)
        original = model.fit

        def fit(X, y):
            seen.append(feats.loc[X.index, "season"].max())
            return original(X, y)

        model.fit = fit
        return model

    monkeypatch.setattr(bt, "make_model", spy)
    preds = bt.walk_forward(feats, "ridge", [2022, 2023])
    assert seen == [2021, 2022]
    assert len(preds) == 2 * 82 * 15
    assert set(preds["season"]) == {2022, 2023}
    assert preds["prob"].between(0, 1).all()
    assert set(preds["outcome"].unique()) <= {0.0, 1.0}


def test_walk_forward_beats_coin_flip_on_synthetic_league():
    feats = build_features(make_league([2020, 2021, 2022, 2023]))
    preds = bt.walk_forward(feats, "ridge", [2023])
    from celtics.evaluate.metrics import log_loss

    assert log_loss(preds["prob"], preds["outcome"]) < 0.69
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py tests/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.evaluate'`

- [ ] **Step 4: Write `src/celtics/evaluate/metrics.py`**

```python
"""Probabilistic forecast metrics."""

import numpy as np

EPS = 1e-15


def _arrays(p, y):
    return np.asarray(p, dtype=float), np.asarray(y, dtype=float)


def log_loss(p, y) -> float:
    p, y = _arrays(p, y)
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y) -> float:
    p, y = _arrays(p, y)
    return float(np.mean((p - y) ** 2))


def accuracy(p, y) -> float:
    p, y = _arrays(p, y)
    return float(np.mean((p > 0.5) == (y == 1)))


def calibration_table(p, y, n_bins: int = 10) -> list[dict]:
    p, y = _arrays(p, y)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "mean_pred": float(p[mask].mean()),
                "observed": float(y[mask].mean()),
                "n": int(mask.sum()),
            }
        )
    return rows


def summarize(p, y) -> dict:
    return {
        "n": int(len(np.asarray(p))),
        "log_loss": log_loss(p, y),
        "brier": brier(p, y),
        "accuracy": accuracy(p, y),
    }
```

- [ ] **Step 5: Write `src/celtics/evaluate/backtest.py`**

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py tests/test_backtest.py -v`
Expected: 7 passed

- [ ] **Step 7: Commit**

```bash
git add src/celtics/evaluate tests/test_metrics.py tests/test_backtest.py
git commit -m "feat: forecast metrics and walk-forward backtest"
```

---

### Task 10: Simulation inputs

**Files:**
- Create: `src/celtics/simulate/__init__.py` (empty), `src/celtics/simulate/inputs.py`, `tests/test_sim_inputs.py`

**Interfaces:**
- Consumes: `FEATURES`, `TeamState`, `matchup_matrix` (Task 7); `MarginModel` (Task 8); `TEAMS`, `TEAM_INDEX`, `IS_EAST`.
- Produces:
  - `@dataclass SimInputs` with fields: `teams: list[str]`, `current_wins`, `current_losses`, `total_games` (int arrays of shape (30,)), `h2h_wins` (int (30, 30), where `[i, j]` is the number of i's wins over j), `conf_wins`, `conf_losses` (int (30,)), `rem_home`, `rem_away` (int (n,)), `rem_margin` (float (n,)), `same_conf_rem` (bool (n,)), `extra_games` (int (30,)), `neutral_margin` (float (30,)), `matchup` (float (30, 30)), `sigma: float`
  - `prepare_sim_inputs(features, model, state, season, target_games: int | None = 82) -> SimInputs`. `extra_games` = max(0, target − scheduled games) and stays all zeros when `target_games is None`. `neutral_margin[i]` is the mean over j≠i of `(M[i,j] − M[j,i]) / 2`.

- [ ] **Step 1: Write the failing tests `tests/test_sim_inputs.py`**

```python
import numpy as np
import pytest

from celtics.config import FIRST_TRAIN_SEASON
from celtics.features.build import FEATURES, build_features, current_team_state
from celtics.models.margin import make_model
from celtics.simulate.inputs import prepare_sim_inputs
from celtics.teams import TEAM_INDEX
from tests.synth import make_league


@pytest.fixture(scope="module")
def setup():
    games = make_league([2022, 2023], current=2023, completed_rounds=15)
    feats = build_features(games)
    train = feats[feats["completed"] & (feats["season"] >= FIRST_TRAIN_SEASON)]
    model = make_model("ridge").fit(train[FEATURES], train["margin"])
    return games, feats, model


def test_counts_are_consistent(setup):
    games, feats, model = setup
    inp = prepare_sim_inputs(feats, model, current_team_state(games), 2023)
    assert inp.current_wins.sum() == inp.current_losses.sum() == 15 * 15
    assert inp.h2h_wins.sum() == 15 * 15
    assert len(inp.rem_home) == len(inp.rem_margin) == 67 * 15
    assert np.all(inp.total_games == 82)
    assert np.all(inp.extra_games == 0)
    assert (inp.conf_wins + inp.conf_losses).sum() <= 2 * 15 * 15
    assert inp.neutral_margin.sum() == pytest.approx(0, abs=1e-6)
    assert inp.sigma == model.sigma


def test_missing_scheduled_games_become_extra_games(setup):
    games, feats, model = setup
    rem = feats[(feats["season"] == 2023) & ~feats["completed"]]
    dropped = rem.iloc[0]
    trimmed = feats.drop(index=rem.index[0])
    inp = prepare_sim_inputs(trimmed, model, current_team_state(games), 2023)
    assert inp.extra_games.sum() == 2
    assert np.all(inp.total_games == 82)
    assert inp.extra_games[TEAM_INDEX[dropped["home_team"]]] == 1


def test_no_extra_games_when_target_is_none(setup):
    games, feats, model = setup
    trimmed = feats.drop(index=feats[~feats["completed"]].index[0])
    inp = prepare_sim_inputs(trimmed, model, current_team_state(games), 2023, target_games=None)
    assert inp.extra_games.sum() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sim_inputs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.simulate'`

- [ ] **Step 3: Write `src/celtics/simulate/inputs.py`**

```python
"""Everything the season simulator needs, as numpy arrays."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from celtics.features.build import FEATURES, TeamState, matchup_matrix
from celtics.models.margin import MarginModel
from celtics.teams import IS_EAST, TEAM_INDEX, TEAMS


@dataclass
class SimInputs:
    teams: list[str]
    current_wins: np.ndarray
    current_losses: np.ndarray
    total_games: np.ndarray
    h2h_wins: np.ndarray
    conf_wins: np.ndarray
    conf_losses: np.ndarray
    rem_home: np.ndarray
    rem_away: np.ndarray
    rem_margin: np.ndarray
    same_conf_rem: np.ndarray
    extra_games: np.ndarray
    neutral_margin: np.ndarray
    matchup: np.ndarray
    sigma: float


def _idx(series: pd.Series) -> np.ndarray:
    return series.map(TEAM_INDEX).to_numpy(dtype=np.int64)


def prepare_sim_inputs(
    features: pd.DataFrame,
    model: MarginModel,
    state: TeamState,
    season: int,
    target_games: int | None = 82,
) -> SimInputs:
    n = len(TEAMS)
    cur = features[features["season"] == season]
    done = cur[cur["completed"]]
    rem = cur[~cur["completed"]]

    dh, da = _idx(done["home_team"]), _idx(done["away_team"])
    home_won = (done["margin"] > 0).to_numpy()
    w_idx = np.where(home_won, dh, da)
    l_idx = np.where(home_won, da, dh)
    current_wins = np.bincount(w_idx, minlength=n)
    current_losses = np.bincount(l_idx, minlength=n)
    h2h = np.zeros((n, n), dtype=np.int64)
    np.add.at(h2h, (w_idx, l_idx), 1)
    same_done = IS_EAST[dh] == IS_EAST[da]
    conf_wins = np.bincount(w_idx[same_done], minlength=n)
    conf_losses = np.bincount(l_idx[same_done], minlength=n)

    rh, ra = _idx(rem["home_team"]), _idx(rem["away_team"])
    rem_margin = model.predict_margin(rem[FEATURES]) if len(rem) else np.zeros(0)
    scheduled = (
        current_wins + current_losses + np.bincount(rh, minlength=n) + np.bincount(ra, minlength=n)
    )
    if target_games is None:
        extra = np.zeros(n, dtype=np.int64)
    else:
        extra = np.clip(target_games - scheduled, 0, None).astype(np.int64)

    matchup = matchup_matrix(model.predict_margin, state)
    neutral = ((matchup - matchup.T) / 2).sum(axis=1) / (n - 1)

    return SimInputs(
        teams=list(TEAMS),
        current_wins=current_wins,
        current_losses=current_losses,
        total_games=scheduled + extra,
        h2h_wins=h2h,
        conf_wins=conf_wins,
        conf_losses=conf_losses,
        rem_home=rh,
        rem_away=ra,
        rem_margin=np.asarray(rem_margin, dtype=float),
        same_conf_rem=IS_EAST[rh] == IS_EAST[ra],
        extra_games=extra,
        neutral_margin=neutral,
        matchup=matchup,
        sigma=float(model.sigma),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sim_inputs.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/celtics/simulate tests/test_sim_inputs.py
git commit -m "feat: build simulation inputs from features and model"
```

---

### Task 11: Standings tiebreakers and play-in

**Files:**
- Create: `src/celtics/simulate/standings.py`, `src/celtics/simulate/playin.py`, `tests/test_standings.py`

**Interfaces:**
- Consumes: `DIVISION_MEMBERS` (Task 1).
- Produces:
  - `h2h_pct(team: int, group: list[int], h2h: np.ndarray) -> float` (0.5 if they haven't played)
  - `division_winners(pct: np.ndarray, h2h: np.ndarray, tiebreak: np.ndarray) -> np.ndarray[bool]` (30,)
  - `rank_conference(members: list[int], pct, h2h, conf_pct, div_winner, tiebreak) -> list[int]`. Sorts by win % descending. Tied groups are ordered by h2h % within the group, then division winner, then conference win %, then `tiebreak` ascending.
  - `play_in(seeds_7_to_10: list[int], matchup, offsets, sigma, rng) -> list[int]` returns `[seed7_team, seed8_team]`. The 7v8 winner takes seed 7. The 7v8 loser hosts the 9v10 winner, and that game's winner takes seed 8.

- [ ] **Step 1: Write the failing tests `tests/test_standings.py`**

```python
import numpy as np

from celtics.simulate.playin import play_in
from celtics.simulate.standings import division_winners, h2h_pct, rank_conference


def test_h2h_pct():
    h2h = np.zeros((30, 30), dtype=int)
    h2h[0, 1] = 3
    h2h[1, 0] = 1
    assert h2h_pct(0, [0, 1], h2h) == 0.75
    assert h2h_pct(2, [2, 3], h2h) == 0.5


def test_rank_by_pct_then_h2h_then_division_then_conf():
    members = [0, 1, 2, 3]
    pct = np.zeros(30)
    pct[[0, 1, 2, 3]] = [0.6, 0.6, 0.5, 0.5]
    h2h = np.zeros((30, 30), dtype=int)
    h2h[1, 0] = 2  # team 1 owns team 0
    conf_pct = np.zeros(30)
    conf_pct[3] = 0.9  # team 3 has better conference record
    div = np.zeros(30, dtype=bool)
    tiebreak = np.zeros(30)
    order = rank_conference(members, pct, h2h, conf_pct, div, tiebreak)
    assert order == [1, 0, 3, 2]

    div[2] = True  # division winner beats conference record
    assert rank_conference(members, pct, h2h, conf_pct, div, tiebreak) == [1, 0, 2, 3]


def test_random_tiebreak_is_last_resort():
    pct = np.full(30, 0.5)
    tiebreak = np.zeros(30)
    tiebreak[[0, 1]] = [0.9, 0.1]
    order = rank_conference(
        [0, 1], pct, np.zeros((30, 30), int), np.zeros(30), np.zeros(30, bool), tiebreak
    )
    assert order == [1, 0]


def test_division_winners_one_per_division():
    rng = np.random.default_rng(0)
    pct = np.round(rng.random(30), 3)
    winners = division_winners(pct, np.zeros((30, 30), int), rng.random(30))
    assert winners.sum() == 6


def test_play_in_bracket():
    rng = np.random.default_rng(0)
    matchup = np.zeros((30, 30))
    off = np.zeros(30)
    off[[7, 9]] = 100  # teams 7 and 9 always win
    seeds = [6, 7, 8, 9]  # team ids holding seeds 7, 8, 9, 10
    # 7v8: team 7 beats 6 -> seed 7 = team 7; 9v10: team 9 beats 8; 6 vs 9: team 9 wins
    assert play_in(seeds, matchup, off, 1.0, rng) == [7, 9]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_standings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.simulate.standings'`

- [ ] **Step 3: Write `src/celtics/simulate/standings.py`**

```python
"""Simplified NBA tiebreakers: h2h -> division winner -> conference record -> random."""

import numpy as np

from celtics.teams import DIVISION_MEMBERS


def h2h_pct(team: int, group: list[int], h2h: np.ndarray) -> float:
    others = [o for o in group if o != team]
    wins = int(h2h[team, others].sum())
    losses = int(h2h[others, team].sum())
    return wins / (wins + losses) if wins + losses else 0.5


def division_winners(pct: np.ndarray, h2h: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    winners = np.zeros(len(pct), dtype=bool)
    for members in DIVISION_MEMBERS:
        best = max(pct[t] for t in members)
        tied = [t for t in members if pct[t] == best]
        if len(tied) > 1:
            tied = sorted(tied, key=lambda t, g=tied: (-h2h_pct(t, g, h2h), tiebreak[t]))
        winners[tied[0]] = True
    return winners


def rank_conference(
    members: list[int],
    pct: np.ndarray,
    h2h: np.ndarray,
    conf_pct: np.ndarray,
    div_winner: np.ndarray,
    tiebreak: np.ndarray,
) -> list[int]:
    ordered = sorted(members, key=lambda t: -pct[t])
    result: list[int] = []
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and pct[ordered[j + 1]] == pct[ordered[i]]:
            j += 1
        group = ordered[i : j + 1]
        if len(group) > 1:
            group = sorted(
                group,
                key=lambda t, g=group: (
                    -h2h_pct(t, g, h2h),
                    -int(div_winner[t]),
                    -conf_pct[t],
                    tiebreak[t],
                ),
            )
        result.extend(group)
        i = j + 1
    return result
```

- [ ] **Step 4: Write `src/celtics/simulate/playin.py`**

```python
"""Play-in tournament for seeds 7-10."""

import numpy as np


def play_in(
    seeds_7_to_10: list[int],
    matchup: np.ndarray,
    offsets: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> list[int]:
    s7, s8, s9, s10 = seeds_7_to_10

    def play(home: int, away: int) -> tuple[int, int]:
        margin = matchup[home, away] + offsets[home] - offsets[away] + rng.normal(0.0, sigma)
        return (home, away) if margin > 0 else (away, home)

    seed7, loser78 = play(s7, s8)
    winner910, _ = play(s9, s10)
    seed8, _ = play(loser78, winner910)
    return [seed7, seed8]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_standings.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/celtics/simulate/standings.py src/celtics/simulate/playin.py tests/test_standings.py
git commit -m "feat: simplified NBA tiebreakers and play-in bracket"
```

---

### Task 12: Monte Carlo season simulation

**Files:**
- Create: `src/celtics/simulate/season.py`, `tests/test_season.py`

**Interfaces:**
- Consumes: `SimInputs` (Task 10); `division_winners`, `rank_conference`, `play_in` (Task 11); `CONFERENCE_MEMBERS`; `N_SIMS`, `DEFAULT_TEAM_SD`.
- Produces:
  - `@dataclass SimResult(teams: list[str], wins: np.ndarray (n_sims, 30) int, seed: np.ndarray (n_sims, 30) int 1–15 within conference, made_playoffs: np.ndarray (n_sims, 30) bool)`
  - `simulate_season(inp: SimInputs, n_sims: int = N_SIMS, team_sd: float = DEFAULT_TEAM_SD, seed: int = 0) -> SimResult`

Each simulated game's margin is `rem_margin + off[home] − off[away] + N(0, σ)`, where `off ~ N(0, team_sd)` is drawn once per team per simulation. Missing ("extra") games are Binomial with p = `Φ((neutral_margin + off)/σ)`.

- [ ] **Step 1: Write the failing tests `tests/test_season.py`**

```python
import numpy as np
import pytest

from celtics.simulate.inputs import SimInputs
from celtics.simulate.season import simulate_season
from celtics.teams import CONFERENCE_MEMBERS, IS_EAST, TEAMS


def even_inputs(rounds=82, margin=0.0, favorite=None, seed=0):
    rng = np.random.default_rng(seed)
    home, away = [], []
    for _ in range(rounds):
        order = rng.permutation(30)
        home += list(order[0::2])
        away += list(order[1::2])
    home, away = np.array(home), np.array(away)
    m = np.full(len(home), margin, dtype=float)
    if favorite is not None:
        m = np.where(home == favorite, 50.0, np.where(away == favorite, -50.0, m))
    zeros = np.zeros(30, dtype=np.int64)
    return SimInputs(
        teams=list(TEAMS),
        current_wins=zeros.copy(),
        current_losses=zeros.copy(),
        total_games=np.full(30, rounds),
        h2h_wins=np.zeros((30, 30), dtype=np.int64),
        conf_wins=zeros.copy(),
        conf_losses=zeros.copy(),
        rem_home=home,
        rem_away=away,
        rem_margin=m,
        same_conf_rem=IS_EAST[home] == IS_EAST[away],
        extra_games=zeros.copy(),
        neutral_margin=np.zeros(30),
        matchup=np.zeros((30, 30)),
        sigma=12.0,
    )


def test_equal_teams_average_41_wins():
    res = simulate_season(even_inputs(), n_sims=300, team_sd=0.0, seed=1)
    assert res.wins.mean() == pytest.approx(41, abs=0.5)


def test_wins_equal_losses_league_wide():
    inp = even_inputs()
    res = simulate_season(inp, n_sims=100, team_sd=2.0, seed=2)
    assert np.all(res.wins.sum(axis=1) == 15 * 82)


def test_seeds_and_playoff_counts():
    res = simulate_season(even_inputs(), n_sims=100, team_sd=2.0, seed=3)
    for members in CONFERENCE_MEMBERS:
        seeds = np.sort(res.seed[:, members], axis=1)
        assert np.all(seeds == np.arange(1, 16))
        assert np.all(res.made_playoffs[:, members].sum(axis=1) == 8)


def test_dominant_team_is_top_seed():
    res = simulate_season(even_inputs(favorite=0), n_sims=100, team_sd=0.0, seed=4)
    assert res.wins[:, 0].min() >= 80
    assert np.all(res.seed[:, 0] == 1)


def test_extra_games_add_wins():
    inp = even_inputs(rounds=80)
    inp.extra_games = np.full(30, 2)
    inp.total_games = np.full(30, 82)
    res = simulate_season(inp, n_sims=200, team_sd=0.0, seed=5)
    assert res.wins.mean() == pytest.approx(41, abs=0.5)
    assert res.wins.max() <= 82


def test_deterministic_with_seed():
    a = simulate_season(even_inputs(), n_sims=50, seed=9)
    b = simulate_season(even_inputs(), n_sims=50, seed=9)
    assert np.array_equal(a.wins, b.wins) and np.array_equal(a.seed, b.seed)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_season.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.simulate.season'`

- [ ] **Step 3: Write `src/celtics/simulate/season.py`**

```python
"""Monte Carlo simulation of the remaining league schedule."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from celtics.config import DEFAULT_TEAM_SD, N_SIMS
from celtics.simulate.inputs import SimInputs
from celtics.simulate.playin import play_in
from celtics.simulate.standings import division_winners, rank_conference
from celtics.teams import CONFERENCE_MEMBERS


@dataclass
class SimResult:
    teams: list[str]
    wins: np.ndarray
    seed: np.ndarray
    made_playoffs: np.ndarray


def _count(idx: np.ndarray, n_sims: int, size: int) -> np.ndarray:
    """Per-simulation counts of each value in idx (shape (n_sims, k)) -> (n_sims, size)."""
    shifted = idx + (np.arange(n_sims)[:, None] * size)
    return np.bincount(shifted.ravel(), minlength=n_sims * size).reshape(n_sims, size)


def simulate_season(
    inp: SimInputs, n_sims: int = N_SIMS, team_sd: float = DEFAULT_TEAM_SD, seed: int = 0
) -> SimResult:
    rng = np.random.default_rng(seed)
    n = len(inp.teams)
    off = rng.normal(0.0, team_sd, size=(n_sims, n)) if team_sd > 0 else np.zeros((n_sims, n))

    noise = rng.normal(0.0, inp.sigma, size=(n_sims, len(inp.rem_home)))
    margin = inp.rem_margin[None, :] + off[:, inp.rem_home] - off[:, inp.rem_away] + noise
    home_win = margin > 0
    winners = np.where(home_win, inp.rem_home[None, :], inp.rem_away[None, :])
    losers = np.where(home_win, inp.rem_away[None, :], inp.rem_home[None, :])

    wins = inp.current_wins[None, :] + _count(winners, n_sims, n)
    if inp.extra_games.any():
        p = norm.cdf((inp.neutral_margin[None, :] + off) / inp.sigma)
        wins = wins + rng.binomial(inp.extra_games[None, :], p)
    losses = inp.total_games[None, :] - wins

    h2h = _count(winners * n + losers, n_sims, n * n).reshape(n_sims, n, n) + inp.h2h_wins[None]
    same = inp.same_conf_rem
    conf_w = inp.conf_wins[None, :] + _count(winners[:, same], n_sims, n)
    conf_l = inp.conf_losses[None, :] + _count(losers[:, same], n_sims, n)
    tiebreak = rng.random((n_sims, n))

    seeds = np.zeros((n_sims, n), dtype=np.int64)
    made = np.zeros((n_sims, n), dtype=bool)
    for s in range(n_sims):
        played = np.maximum(wins[s] + losses[s], 1)
        pct = np.round(wins[s] / played, 9)
        conf_games = conf_w[s] + conf_l[s]
        conf_pct = np.where(conf_games > 0, conf_w[s] / np.maximum(conf_games, 1), 0.5)
        div_winner = division_winners(pct, h2h[s], tiebreak[s])
        for members in CONFERENCE_MEMBERS:
            order = rank_conference(members, pct, h2h[s], conf_pct, div_winner, tiebreak[s])
            seeds[s, order] = np.arange(1, len(order) + 1)
            made[s, order[:6]] = True
            made[s, play_in(order[6:10], inp.matchup, off[s], inp.sigma, rng)] = True

    return SimResult(teams=list(inp.teams), wins=wins.astype(np.int64), seed=seeds, made_playoffs=made)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_season.py -v`
Expected: 6 passed

- [ ] **Step 5: Check speed at full size**

Run:
```bash
uv run python -c "
import time
from tests.test_season import even_inputs
from celtics.simulate.season import simulate_season
t=time.time(); simulate_season(even_inputs(rounds=67), n_sims=10000); print(f'{time.time()-t:.1f}s')"
```
Expected: under 30s. If it's slower, profile before optimizing. Don't change semantics.

- [ ] **Step 6: Commit**

```bash
git add src/celtics/simulate/season.py tests/test_season.py
git commit -m "feat: Monte Carlo season simulation with correlated team strength"
```

---

### Task 13: Season-level coverage (tunes team uncertainty)

**Files:**
- Modify: `src/celtics/evaluate/backtest.py` (append functions)
- Create: `tests/test_coverage.py`

**Interfaces:**
- Consumes: `as_of`, `build_features`, `current_team_state` (Task 7); `prepare_sim_inputs` (Task 10); `simulate_season` (Task 12); `make_model`; `FIRST_TRAIN_SEASON`, `COVERAGE_CHECKPOINT_GAMES`.
- Produces:
  - `checkpoint_date(season_games, n_games=COVERAGE_CHECKPOINT_GAMES) -> pd.Timestamp`: the median, across teams, of the date of each team's nth game.
  - `season_coverage(games, model_name, test_seasons, team_sds, n_sims, seed=0) -> dict[float, float]`: for each sd, the share of team-seasons whose actual final wins fell inside the simulated 10th–90th percentile range.

- [ ] **Step 1: Write the failing tests `tests/test_coverage.py`**

```python
import pandas as pd

from celtics.evaluate.backtest import checkpoint_date, season_coverage
from tests.synth import make_league


def test_checkpoint_date_is_15th_game():
    games = make_league([2022])
    assert checkpoint_date(games[games["season"] == 2022]) == pd.Timestamp("2022-10-20") + pd.Timedelta(days=28)


def test_coverage_in_range_and_wider_with_more_uncertainty():
    games = make_league([2020, 2021, 2022])
    cov = season_coverage(games, "ridge", [2022], [0.0, 6.0], n_sims=300)
    assert set(cov) == {0.0, 6.0}
    assert all(0.0 <= v <= 1.0 for v in cov.values())
    assert cov[6.0] >= cov[0.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_coverage.py -v`
Expected: FAIL with `ImportError: cannot import name 'checkpoint_date'`

- [ ] **Step 3: Replace `src/celtics/evaluate/backtest.py` with the full file below** (`walk_forward` is unchanged from Task 9)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_coverage.py tests/test_backtest.py -v && uv run ruff check src`
Expected: 4 passed; `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/celtics/evaluate/backtest.py tests/test_coverage.py
git commit -m "feat: season-level interval coverage for tuning team uncertainty"
```

---

### Task 14: JSON schema and export builders

**Files:**
- Create: `src/celtics/export/__init__.py` (empty), `src/celtics/export/schema.py`, `src/celtics/export/build.py`, `tests/test_export.py`

**Interfaces:**
- Consumes: `SimResult` (Task 12); `SimInputs` (Task 10); `FEATURES` (Task 7); `MarginModel` (Task 8); `summarize` (Task 9); `TEAM_INDEX`, `CONFERENCE_OF`.
- Produces:
  - Schema (pydantic): `Predictions`, `HistoryEntry`, `BacktestReport` (each raises `pydantic.ValidationError` on bad data).
  - `team_summary(res, team, wins, losses) -> dict`
  - `seed_probabilities(res, team) -> dict[str, float]` with keys `"1"`–`"15"`
  - `conference_table(res, inp, conference) -> list[dict]`
  - `game_rows(features, model, season, team, pregame_log) -> list[dict]`
  - `update_pregame_log(log, rows, now_iso) -> dict`
  - `live_metrics(rows) -> dict | None`
  - `history_entry(day: date, summary) -> dict`, `upsert_history(history, entry) -> list[dict]`
  - `write_json_atomic(path, obj) -> None`, `read_json(path, default)`

`pregame_log` maps `game_id → {"win_prob", "pred_margin", "logged_at"}` from the team's perspective. Each update overwrites the entry for an unplayed game, and it stays frozen once the game is played. That gives completed games a real pre-game prediction.

- [ ] **Step 1: Write the failing tests `tests/test_export.py`**

```python
from datetime import date

import numpy as np
import pytest
from pydantic import ValidationError

from celtics.config import FIRST_TRAIN_SEASON
from celtics.export.build import (
    game_rows,
    history_entry,
    live_metrics,
    read_json,
    seed_probabilities,
    team_summary,
    update_pregame_log,
    upsert_history,
    write_json_atomic,
)
from celtics.export.schema import HistoryEntry, TeamSummary
from celtics.features.build import FEATURES, build_features
from celtics.models.margin import make_model
from celtics.simulate.season import SimResult
from celtics.teams import TEAMS
from tests.synth import make_league


def fake_result(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    seeds = np.tile(np.arange(1, 16), (n, 2))
    wins = rng.integers(30, 60, size=(n, 30))
    made = seeds <= 8
    return SimResult(teams=list(TEAMS), wins=wins, seed=seeds, made_playoffs=made)


def test_team_summary_is_consistent():
    s = team_summary(fake_result(), "BOS", 9, 4)
    TeamSummary.model_validate(s)
    assert s["p_top6"] + s["p_playin"] + s["p_miss"] == pytest.approx(1, abs=1e-3)
    assert sum(s["win_distribution"].values()) == pytest.approx(1, abs=1e-3)
    assert s["proj_wins_p10"] <= s["proj_wins_median"] <= s["proj_wins_p90"]


def test_seed_probabilities_sum_to_one():
    sp = seed_probabilities(fake_result(), "BOS")
    assert list(sp) == [str(k) for k in range(1, 16)]
    assert sum(sp.values()) == pytest.approx(1, abs=1e-3)


def test_schema_rejects_bad_probability():
    with pytest.raises(ValidationError):
        HistoryEntry.model_validate(
            {"date": "2026-11-12", "record": "9-4", "projected_wins_median": 50, "p10": 45,
             "p90": 55, "playoff_prob": 1.5, "top6_prob": 0.5, "seed1_prob": 0.1}
        )


@pytest.fixture(scope="module")
def season_rows():
    feats = build_features(make_league([2022, 2023], current=2023))
    train = feats[feats["completed"] & (feats["season"] >= FIRST_TRAIN_SEASON)]
    model = make_model("ridge").fit(train[FEATURES], train["margin"])
    return feats, model


def test_game_rows_perspective_and_pregame_log(season_rows):
    feats, model = season_rows
    rows = game_rows(feats, model, 2023, "BOS", pregame_log={})
    assert len(rows) == 82
    upcoming = [r for r in rows if not r["completed"]]
    done = [r for r in rows if r["completed"]]
    assert len(done) == 15 and all(r["result"][0] in "WL" for r in done)
    assert all(r["pregame_win_prob"] is None for r in done)  # nothing logged yet
    away = next(r for r in upcoming if not r["home"])
    assert (away["win_prob"] > 0.5) == (away["pred_margin"] > 0)

    log = update_pregame_log({}, rows, "2023-11-20T14:00:00")
    assert set(log) == {r["game_id"] for r in upcoming}
    first = upcoming[0]["game_id"]
    log[first]["win_prob"] = 0.42
    # once the game is completed the logged value is kept and shown
    feats2 = feats.copy()
    i = feats2.index[feats2["game_id"] == first][0]
    feats2.loc[i, ["home_pts", "away_pts", "completed", "margin"]] = [100.0, 90.0, True, 10.0]
    rows2 = game_rows(feats2, model, 2023, "BOS", pregame_log=log)
    shown = next(r for r in rows2 if r["game_id"] == first)
    assert shown["completed"] and shown["pregame_win_prob"] == 0.42
    log2 = update_pregame_log(log, rows2, "2023-11-21T14:00:00")
    assert log2[first]["win_prob"] == 0.42
    metrics = live_metrics(rows2)
    assert metrics["n"] == 1


def test_history_upsert_replaces_same_day():
    s = team_summary(fake_result(), "BOS", 9, 4)
    s["p_playoffs"] = 0.8
    e1 = history_entry(date(2026, 11, 12), s)
    HistoryEntry.model_validate(e1)
    hist = upsert_history([], e1)
    hist = upsert_history(hist, {**e1, "playoff_prob": 0.9})
    assert len(hist) == 1 and hist[0]["playoff_prob"] == 0.9
    hist = upsert_history(hist, {**e1, "date": "2026-11-11"})
    assert [h["date"] for h in hist] == ["2026-11-11", "2026-11-12"]


def test_atomic_write_roundtrip(tmp_path):
    path = tmp_path / "nested" / "x.json"
    write_json_atomic(path, {"a": 1})
    assert read_json(path, None) == {"a": 1}
    assert read_json(tmp_path / "missing.json", []) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.export'`

- [ ] **Step 3: Write `src/celtics/export/schema.py`**

```python
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
```

- [ ] **Step 4: Write `src/celtics/export/build.py`**

```python
"""Turn simulation output into website JSON."""

import json
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from celtics.evaluate.metrics import summarize
from celtics.features.build import FEATURES
from celtics.models.margin import MarginModel
from celtics.simulate.inputs import SimInputs
from celtics.simulate.season import SimResult
from celtics.teams import CONFERENCE_OF, TEAM_INDEX


def r4(x) -> float:
    return round(float(x), 4)


def team_summary(res: SimResult, team: str, wins: int, losses: int) -> dict:
    i = TEAM_INDEX[team]
    w = res.wins[:, i]
    seeds = res.seed[:, i]
    p10, p25, p50, p75, p90 = np.percentile(w, [10, 25, 50, 75, 90])
    counts = np.bincount(w) / len(w)
    return {
        "team": team,
        "wins": int(wins),
        "losses": int(losses),
        "proj_wins_median": float(p50),
        "proj_wins_p10": float(p10),
        "proj_wins_p25": float(p25),
        "proj_wins_p75": float(p75),
        "proj_wins_p90": float(p90),
        "p_50_wins": r4((w >= 50).mean()),
        "p_playoffs": r4(res.made_playoffs[:, i].mean()),
        "p_top6": r4((seeds <= 6).mean()),
        "p_playin": r4(((seeds >= 7) & (seeds <= 10)).mean()),
        "p_miss": r4((seeds >= 11).mean()),
        "p_seed1": r4((seeds == 1).mean()),
        "win_distribution": {str(k): r4(v) for k, v in enumerate(counts) if v > 0},
    }


def seed_probabilities(res: SimResult, team: str) -> dict[str, float]:
    seeds = res.seed[:, TEAM_INDEX[team]]
    return {str(k): r4((seeds == k).mean()) for k in range(1, 16)}


def conference_table(res: SimResult, inp: SimInputs, conference: str) -> list[dict]:
    rows = []
    for team, i in TEAM_INDEX.items():
        if CONFERENCE_OF[team] != conference:
            continue
        rows.append(
            {
                "team": team,
                "wins": int(inp.current_wins[i]),
                "losses": int(inp.current_losses[i]),
                "proj_wins": round(float(res.wins[:, i].mean()), 1),
                "p_playoffs": r4(res.made_playoffs[:, i].mean()),
                "p_top6": r4((res.seed[:, i] <= 6).mean()),
                "p_seed1": r4((res.seed[:, i] == 1).mean()),
            }
        )
    return sorted(rows, key=lambda r: -r["proj_wins"])


def game_rows(
    features: pd.DataFrame, model: MarginModel, season: int, team: str, pregame_log: dict
) -> list[dict]:
    cur = features[
        (features["season"] == season)
        & ((features["home_team"] == team) | (features["away_team"] == team))
    ].sort_values(["date", "game_id"])
    if cur.empty:
        return []
    margins = model.predict_margin(cur[FEATURES])
    probs = model.win_prob(cur[FEATURES])
    rows = []
    for row, m, p in zip(cur.itertuples(index=False), margins, probs, strict=True):
        home = row.home_team == team
        out = {
            "game_id": row.game_id,
            "date": row.date.date().isoformat(),
            "opponent": row.away_team if home else row.home_team,
            "home": bool(home),
            "rest_days": float(row.home_rest if home else row.away_rest),
            "opp_rest_days": float(row.away_rest if home else row.home_rest),
            "b2b": bool(row.home_b2b if home else row.away_b2b),
            "completed": bool(row.completed),
            "win_prob": None,
            "pred_margin": None,
            "result": None,
            "pregame_win_prob": None,
        }
        if row.completed:
            mine, theirs = (row.home_pts, row.away_pts) if home else (row.away_pts, row.home_pts)
            out["result"] = f"{'W' if mine > theirs else 'L'} {int(mine)}-{int(theirs)}"
            logged = pregame_log.get(row.game_id)
            out["pregame_win_prob"] = logged["win_prob"] if logged else None
        else:
            out["win_prob"] = r4(p if home else 1 - p)
            out["pred_margin"] = round(float(m if home else -m), 1)
        rows.append(out)
    return rows


def update_pregame_log(log: dict, rows: list[dict], now_iso: str) -> dict:
    updated = dict(log)
    for r in rows:
        if not r["completed"]:
            updated[r["game_id"]] = {
                "win_prob": r["win_prob"],
                "pred_margin": r["pred_margin"],
                "logged_at": now_iso,
            }
    return updated


def live_metrics(rows: list[dict]) -> dict | None:
    scored = [r for r in rows if r["completed"] and r["pregame_win_prob"] is not None]
    if not scored:
        return None
    p = [r["pregame_win_prob"] for r in scored]
    y = [1.0 if r["result"].startswith("W") else 0.0 for r in scored]
    s = summarize(p, y)
    return {"n": s["n"], "log_loss": r4(s["log_loss"]), "brier": r4(s["brier"]), "accuracy": r4(s["accuracy"])}


def history_entry(day: date, summary: dict) -> dict:
    return {
        "date": day.isoformat(),
        "record": f"{summary['wins']}-{summary['losses']}",
        "projected_wins_median": summary["proj_wins_median"],
        "p10": summary["proj_wins_p10"],
        "p90": summary["proj_wins_p90"],
        "playoff_prob": summary["p_playoffs"],
        "top6_prob": summary["p_top6"],
        "seed1_prob": summary["p_seed1"],
    }


def upsert_history(history: list[dict], entry: dict) -> list[dict]:
    kept = [h for h in history if h["date"] != entry["date"]]
    return sorted([*kept, entry], key=lambda h: h["date"])


def read_json(path: Path, default):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_export.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add src/celtics/export tests/test_export.py
git commit -m "feat: JSON schema and website export builders"
```

---

### Task 15: Pipeline orchestration and CLI

**Files:**
- Create: `src/celtics/pipeline.py`, `src/celtics/cli.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `load_or_train(features, paths, today, force=False) -> MarginModel`. Saves `data/models/<MODEL_NAME>.joblib` plus `<MODEL_NAME>.json` (`{"trained_on": "YYYY-MM-DD"}`) and refits when the model is missing, ≥7 days old, or `force` is set.
  - `read_team_sd(paths) -> float`: uses `chosen_team_sd` from `site/data/backtest.json`, else `DEFAULT_TEAM_SD`.
  - `run_update(paths, today, offline=False, force_train=False, n_sims=N_SIMS, now=None) -> dict`: writes `site/data/predictions.json`, `history.json`, `pregame_log.json`.
  - `run_backtest(paths, today, offline=False, test_seasons=BACKTEST_SEASONS, n_sims=BACKTEST_COVERAGE_SIMS, now=None) -> dict`: writes `site/data/backtest.json`.
  - CLI `celtics {fetch,train,update,backtest}`. Exit codes: 0 ok, 2 fetch blocked, 3 data validation failed.

- [ ] **Step 1: Write the failing tests `tests/test_pipeline.py`**

```python
import json
from datetime import date, datetime

import pytest

from celtics.cli import main
from celtics.config import Paths
from celtics.data.cache import season_path
from celtics.data.fetch import FetchBlockedError
from celtics.export.schema import BacktestReport, HistoryEntry, Predictions
from celtics.pipeline import run_backtest, run_update
from tests.synth import make_league

TODAY = date(2024, 11, 20)


@pytest.fixture
def repo(tmp_path):
    games = make_league([2021, 2022, 2023, 2024], current=2024, completed_rounds=15)
    raw = Paths(tmp_path).raw_dir
    raw.mkdir(parents=True)
    for season, g in games.groupby("season"):
        g.to_parquet(season_path(raw, season), index=False)
    return Paths(tmp_path)


def test_run_update_writes_valid_outputs(repo):
    run_update(repo, TODAY, offline=True, n_sims=200, now=datetime(2024, 11, 20, 14, 0))
    pred = json.loads((repo.site_data / "predictions.json").read_text())
    Predictions.model_validate(pred)
    assert pred["season"] == "2024-25"
    assert len(pred["games"]) == 82
    assert pred["team_summary"]["wins"] + pred["team_summary"]["losses"] == 15
    history = json.loads((repo.site_data / "history.json").read_text())
    assert len(history) == 1
    HistoryEntry.model_validate(history[0])
    log = json.loads((repo.site_data / "pregame_log.json").read_text())
    assert len(log) == 67
    assert (repo.model_dir / "ridge.joblib").exists()


def test_run_update_twice_same_day_keeps_one_history_entry(repo):
    run_update(repo, TODAY, offline=True, n_sims=100)
    run_update(repo, TODAY, offline=True, n_sims=100)
    assert len(json.loads((repo.site_data / "history.json").read_text())) == 1


def test_run_backtest_writes_report(repo):
    report = run_backtest(repo, TODAY, offline=True, test_seasons=[2022, 2023], n_sims=100)
    BacktestReport.model_validate(report)
    assert set(report["models"]) == {"homecourt", "elo", "ridge", "gbm"}
    assert report["chosen_team_sd"] in [0.0, 1.0, 2.0, 2.5, 3.0, 4.0]
    assert (repo.site_data / "backtest.json").exists()


def test_cli_update_offline(repo, monkeypatch):
    monkeypatch.chdir(repo.root)
    assert main(["--date", "2024-11-20", "update", "--offline", "--sims", "100"]) == 0


def test_cli_reports_blocked_fetch(repo, monkeypatch, capsys):
    monkeypatch.chdir(repo.root)

    def blocked(*args, **kwargs):
        raise FetchBlockedError("403")

    monkeypatch.setattr("celtics.pipeline.refresh", blocked)
    assert main(["--date", "2024-11-20", "update"]) == 2
    assert "fetch blocked" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.cli'`

- [ ] **Step 3: Write `src/celtics/pipeline.py`**

```python
"""End-to-end orchestration: data -> features -> model -> simulation -> JSON."""

from datetime import UTC, date, datetime

from celtics.config import (
    BACKTEST_COVERAGE_SIMS,
    BACKTEST_SEASONS,
    DEFAULT_TEAM_SD,
    FIRST_TRAIN_SEASON,
    GAMES_PER_TEAM,
    MODEL_NAME,
    N_SIMS,
    RETRAIN_DAYS,
    TEAM,
    TEAM_SD_GRID,
    Paths,
    current_season,
    season_label,
)
from celtics.data.cache import load_all, refresh
from celtics.data.validate import validate_games
from celtics.evaluate.backtest import season_coverage, walk_forward
from celtics.evaluate.metrics import calibration_table, summarize
from celtics.export.build import (
    conference_table,
    game_rows,
    history_entry,
    live_metrics,
    read_json,
    seed_probabilities,
    team_summary,
    update_pregame_log,
    upsert_history,
    write_json_atomic,
)
from celtics.export.schema import BacktestReport, HistoryEntry, Predictions
from celtics.features.build import FEATURES, build_features, current_team_state
from celtics.models.margin import MODELS, MarginModel, load_model, make_model, save_model
from celtics.simulate.inputs import prepare_sim_inputs
from celtics.simulate.season import simulate_season
from celtics.teams import CONFERENCE_OF, TEAM_INDEX


def _now(now: datetime | None) -> str:
    return (now or datetime.now(UTC)).isoformat(timespec="seconds")


def _load_games(paths: Paths, today: date, offline: bool):
    return load_all(paths.raw_dir) if offline else refresh(paths.raw_dir, today)


def load_or_train(features, paths: Paths, today: date, force: bool = False) -> MarginModel:
    model_path = paths.model_dir / f"{MODEL_NAME}.joblib"
    meta_path = paths.model_dir / f"{MODEL_NAME}.json"
    meta = read_json(meta_path, None)
    if not force and model_path.exists() and meta:
        trained_on = date.fromisoformat(meta["trained_on"])
        if (today - trained_on).days < RETRAIN_DAYS:
            model = load_model(model_path)
            model.trained_on = meta["trained_on"]
            return model
    train = features[features["completed"] & (features["season"] >= FIRST_TRAIN_SEASON)]
    model = make_model(MODEL_NAME).fit(train[FEATURES], train["margin"])
    save_model(model, model_path)
    write_json_atomic(meta_path, {"trained_on": today.isoformat()})
    model.trained_on = today.isoformat()
    return model


def read_team_sd(paths: Paths) -> float:
    report = read_json(paths.site_data / "backtest.json", None)
    return float(report["chosen_team_sd"]) if report else DEFAULT_TEAM_SD


def run_update(
    paths: Paths,
    today: date,
    offline: bool = False,
    force_train: bool = False,
    n_sims: int = N_SIMS,
    now: datetime | None = None,
) -> dict:
    season = current_season(today)
    games = _load_games(paths, today, offline)
    validate_games(games, season)
    feats = build_features(games)
    model = load_or_train(feats, paths, today, force=force_train)
    inputs = prepare_sim_inputs(feats, model, current_team_state(games), season, GAMES_PER_TEAM)
    result = simulate_season(
        inputs, n_sims=n_sims, team_sd=read_team_sd(paths), seed=int(today.strftime("%Y%m%d"))
    )

    generated_at = _now(now)
    log = read_json(paths.site_data / "pregame_log.json", {})
    rows = game_rows(feats, model, season, TEAM, log)
    log = update_pregame_log(log, rows, generated_at)
    i = TEAM_INDEX[TEAM]
    summary = team_summary(result, TEAM, inputs.current_wins[i], inputs.current_losses[i])
    predictions = {
        "generated_at": generated_at,
        "model_version": f"{MODEL_NAME}-{model.trained_on}",
        "season": season_label(season),
        "team_summary": summary,
        "games": rows,
        "seed_probabilities": seed_probabilities(result, TEAM),
        "conference_table": conference_table(result, inputs, CONFERENCE_OF[TEAM]),
        "metrics": {"live": live_metrics(rows)},
    }
    Predictions.model_validate(predictions)
    entry = history_entry(today, summary)
    HistoryEntry.model_validate(entry)
    history = upsert_history(read_json(paths.site_data / "history.json", []), entry)

    write_json_atomic(paths.site_data / "predictions.json", predictions)
    write_json_atomic(paths.site_data / "history.json", history)
    write_json_atomic(paths.site_data / "pregame_log.json", log)
    return predictions


def run_backtest(
    paths: Paths,
    today: date,
    offline: bool = False,
    test_seasons: list[int] = BACKTEST_SEASONS,
    n_sims: int = BACKTEST_COVERAGE_SIMS,
    now: datetime | None = None,
) -> dict:
    games = _load_games(paths, today, offline)
    feats = build_features(games)
    scores, main_preds = {}, None
    for name in MODELS:
        preds = walk_forward(feats, name, test_seasons)
        scores[name] = summarize(preds["prob"], preds["outcome"])
        if name == MODEL_NAME:
            main_preds = preds
    coverage = season_coverage(games, MODEL_NAME, test_seasons, TEAM_SD_GRID, n_sims=n_sims)
    chosen = min(coverage, key=lambda sd: (abs(coverage[sd] - 0.8), sd))
    report = {
        "generated_at": _now(now),
        "test_seasons": list(test_seasons),
        "main_model": MODEL_NAME,
        "models": scores,
        "calibration": calibration_table(main_preds["prob"], main_preds["outcome"]),
        "coverage": {f"{sd:.1f}": round(v, 4) for sd, v in coverage.items()},
        "chosen_team_sd": chosen,
    }
    BacktestReport.model_validate(report)
    write_json_atomic(paths.site_data / "backtest.json", report)
    return report
```

- [ ] **Step 4: Write `src/celtics/cli.py`**

```python
"""`celtics` command-line entry point."""

import argparse
import sys
from datetime import date
from pathlib import Path

from celtics.config import N_SIMS, Paths
from celtics.data.cache import load_all, refresh
from celtics.data.fetch import FetchBlockedError
from celtics.data.validate import DataValidationError
from celtics.features.build import build_features
from celtics.pipeline import load_or_train, run_backtest, run_update


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="celtics", description="Celtics season prediction model")
    parser.add_argument(
        "--date", type=date.fromisoformat, default=None, help="treat this YYYY-MM-DD as today"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="refresh the local data cache")
    sub.add_parser("train", help="refit the model now")
    update = sub.add_parser("update", help="daily run: fetch, simulate, write site JSON")
    update.add_argument("--offline", action="store_true", help="use cached data only")
    update.add_argument("--force-train", action="store_true")
    update.add_argument("--sims", type=int, default=N_SIMS)
    backtest = sub.add_parser("backtest", help="walk-forward backtest -> site/data/backtest.json")
    backtest.add_argument("--offline", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = Paths(Path.cwd())
    today = args.date or date.today()
    try:
        if args.cmd == "fetch":
            games = refresh(paths.raw_dir, today)
            print(f"cached {len(games)} games")
        elif args.cmd == "train":
            load_or_train(build_features(load_all(paths.raw_dir)), paths, today, force=True)
            print("model trained")
        elif args.cmd == "update":
            pred = run_update(
                paths, today, offline=args.offline, force_train=args.force_train, n_sims=args.sims
            )
            s = pred["team_summary"]
            print(
                f"{s['team']} {s['wins']}-{s['losses']} | proj {s['proj_wins_median']:.0f} wins "
                f"(80%: {s['proj_wins_p10']:.0f}-{s['proj_wins_p90']:.0f}) | "
                f"playoffs {s['p_playoffs']:.0%}"
            )
        elif args.cmd == "backtest":
            report = run_backtest(paths, today, offline=args.offline)
            for name, m in report["models"].items():
                print(f"{name:10s} log_loss={m['log_loss']:.4f} brier={m['brier']:.4f} acc={m['accuracy']:.3f}")
            print(f"coverage={report['coverage']} chosen_team_sd={report['chosen_team_sd']}")
    except FetchBlockedError as exc:
        print(f"fetch blocked: {exc}", file=sys.stderr)
        return 2
    except DataValidationError as exc:
        print(f"data validation failed: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 5 passed

- [ ] **Step 6: Run the full suite and lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all tests pass; `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add src/celtics/pipeline.py src/celtics/cli.py tests/test_pipeline.py
git commit -m "feat: update/backtest pipeline and celtics CLI"
```

---

### Task 16: Static website

**Files:**
- Create: `site/index.html`, `site/styles.css`, `site/app.js`

**Interfaces:**
- Consumes: `site/data/predictions.json`, `history.json`, `backtest.json` (schemas from Task 14).

Chart rules (from the dataviz guidance): every chart has a single series, so there is no legend and the accent color is the only series color; 2px lines; hairline grid; a hover tooltip on every chart; a "Show as table" twin for every chart; dark mode uses its own palette; text is never colored with the series color.

- [ ] **Step 1: Write `site/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Celtics 2026-27 Projection</title>
  <meta name="description" content="Daily-updated win probabilities and season projection for the Boston Celtics.">
  <link rel="stylesheet" href="styles.css">
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.16/dist/plot.umd.min.js"></script>
</head>
<body>
  <main>
    <header>
      <h1>Boston Celtics 2026-27 projection</h1>
      <p id="updated" class="muted">Loading…</p>
    </header>

    <div id="headline" class="tiles"></div>

    <section>
      <h2>Projected wins over time</h2>
      <p class="muted">Median projection with the 80% range, one point per daily update.</p>
      <div id="chart-history" class="chart"></div>
      <details><summary>Show as table</summary><div id="table-history" class="table-wrap"></div></details>
    </section>

    <section>
      <h2>Upcoming games</h2>
      <div id="upcoming" class="table-wrap"></div>
    </section>

    <section>
      <h2>Results vs. pre-game prediction</h2>
      <div id="completed" class="table-wrap"></div>
    </section>

    <section>
      <h2>Final win total</h2>
      <p class="muted">Share of 10,000 simulated seasons ending at each win total.</p>
      <div id="chart-wins" class="chart"></div>
      <details><summary>Show as table</summary><div id="table-wins" class="table-wrap"></div></details>
    </section>

    <section>
      <h2>Seed probabilities</h2>
      <p class="muted">Seeds 7–10 go to the play-in; 11–15 miss the postseason.</p>
      <div id="chart-seeds" class="chart"></div>
      <details><summary>Show as table</summary><div id="table-seeds" class="table-wrap"></div></details>
    </section>

    <section>
      <h2>Eastern Conference projection</h2>
      <div id="east" class="table-wrap"></div>
    </section>

    <section>
      <h2>Model report card</h2>
      <p id="backtest-note" class="muted"></p>
      <div id="models" class="table-wrap"></div>
      <p id="live" class="muted"></p>
      <h3>Calibration</h3>
      <p class="muted">When the model says X%, does the team win X% of the time? Points on the diagonal are perfectly calibrated.</p>
      <div id="chart-calibration" class="chart"></div>
      <details><summary>Show as table</summary><div id="table-calibration" class="table-wrap"></div></details>
    </section>

    <section>
      <h2>How it works</h2>
      <p>Each game is predicted as a point margin from team strength (a margin-of-victory Elo rating and a recency-weighted net rating per 100 possessions), home court, and rest (days off, back-to-backs, three games in four nights). The margin becomes a win probability using the model's historical error.</p>
      <p>The rest of the league schedule is simulated 10,000 times. Each simulation shifts every team's strength by a random amount reflecting how uncertain we are about it, then plays out every game, applies tiebreakers (head-to-head, division winner, conference record), and runs the play-in.</p>
      <p>The model is trained on every regular-season game since 2015-16 and tested walk-forward: each test season is predicted using only data from before it.</p>
    </section>

    <footer class="muted">
      Data from NBA.com via nba_api. Not affiliated with the NBA or the Boston Celtics.
      <a href="https://github.com/locnhoang/BostonCelticsNBAPrediction">Source code</a>
    </footer>
  </main>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `site/styles.css`**

```css
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --text: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.1);
  --accent: #007a33;
  --accent-soft: rgba(0, 122, 51, 0.1);
}

@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface: #1a1a19;
    --text: #f0efec;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255, 255, 255, 0.1);
    --accent: #3fae6a;
    --accent-soft: rgba(63, 174, 106, 0.16);
  }
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--page);
  color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}

main { max-width: 960px; margin: 0 auto; padding: 32px 16px 64px; }
header h1 { font-size: 1.75rem; margin: 0 0 4px; }
h2 { font-size: 1.1rem; margin: 0 0 4px; }
h3 { font-size: 0.95rem; margin: 20px 0 4px; }
.muted { color: var(--muted); font-size: 0.85rem; margin: 0 0 12px; }

section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  margin-top: 16px;
}

.tiles {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin-top: 20px;
}
.tile {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
}
.tile-label { color: var(--text-secondary); font-size: 0.8rem; }
.tile-value { font-size: 1.8rem; font-weight: 650; line-height: 1.2; }
.tile-note { color: var(--muted); font-size: 0.75rem; min-height: 1.1em; }

.chart { width: 100%; }
.table-wrap { overflow-x: auto; }

table { width: 100%; border-collapse: collapse; font-size: 0.9rem; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--text-secondary); font-weight: 600; }
tr.highlight td { font-weight: 650; background: var(--accent-soft); }

.prob { display: inline-flex; align-items: center; gap: 8px; }
.prob-track { width: 80px; height: 6px; background: var(--grid); border-radius: 3px; overflow: hidden; }
.prob-fill { display: block; height: 100%; background: var(--accent); border-radius: 3px; }

details { margin-top: 8px; }
summary { cursor: pointer; color: var(--text-secondary); font-size: 0.85rem; }

footer { margin-top: 24px; }
footer a { color: var(--text-secondary); }
```

- [ ] **Step 3: Write `site/app.js`**

```js
"use strict";

const pct = (p) => (p == null ? "—" : `${Math.round(p * 100)}%`);
const pct1 = (p) => `${(p * 100).toFixed(1)}%`;
const signed = (m) => (m > 0 ? `+${m.toFixed(1)}` : m.toFixed(1));
const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

function table(headers, rows, highlighted = []) {
  return el(
    "table",
    {},
    el("thead", {}, el("tr", {}, ...headers.map((h) => el("th", {}, h)))),
    el(
      "tbody",
      {},
      ...rows.map((cells, i) =>
        el("tr", highlighted[i] ? { class: "highlight" } : {}, ...cells.map((c) => el("td", {}, c))),
      ),
    ),
  );
}

function theme() {
  return {
    accent: token("--accent"),
    surface: token("--surface"),
    ink: token("--text-secondary"),
    rule: token("--axis"),
  };
}

function plotStyle(t) {
  return { background: t.surface, color: t.ink, fontSize: "12px", fontFamily: "inherit" };
}

function renderHeadline(pred) {
  const s = pred.team_summary;
  const median = Math.round(s.proj_wins_median);
  const tiles = [
    ["Current record", `${s.wins}-${s.losses}`, ""],
    ["Projected record", `${median}-${82 - median}`, `80% range ${Math.round(s.proj_wins_p10)}–${Math.round(s.proj_wins_p90)} wins`],
    ["Make playoffs", pct(s.p_playoffs), "after the play-in"],
    ["Top-6 seed", pct(s.p_top6), "skips the play-in"],
    ["#1 seed in East", pct(s.p_seed1), ""],
    ["50+ wins", pct(s.p_50_wins), ""],
  ];
  document.getElementById("headline").replaceChildren(
    ...tiles.map(([label, value, note]) =>
      el(
        "div",
        { class: "tile" },
        el("div", { class: "tile-label" }, label),
        el("div", { class: "tile-value" }, value),
        el("div", { class: "tile-note" }, note),
      ),
    ),
  );
  document.getElementById("updated").textContent =
    `Updated ${new Date(pred.generated_at).toLocaleString()} · ${pred.season} · model ${pred.model_version}`;
}

function renderHistory(history, t) {
  const box = document.getElementById("chart-history");
  const data = history.map((d) => ({ ...d, day: new Date(`${d.date}T12:00:00Z`) }));
  if (!data.length) {
    box.replaceChildren(el("p", { class: "muted" }, "The trend appears after the first daily updates."));
    return;
  }
  const tip = (d) =>
    `${d.date}\nMedian: ${d.projected_wins_median} wins\n80% range: ${d.p10}–${d.p90}\nPlayoffs: ${pct(d.playoff_prob)}`;
  box.replaceChildren(
    Plot.plot({
      width: box.clientWidth,
      height: 260,
      marginLeft: 40,
      style: plotStyle(t),
      x: { type: "utc", label: null },
      y: { label: "Projected wins", grid: true, nice: true },
      marks: [
        Plot.areaY(data, { x: "day", y1: "p10", y2: "p90", fill: t.accent, fillOpacity: 0.15 }),
        Plot.lineY(data, { x: "day", y: "projected_wins_median", stroke: t.accent, strokeWidth: 2 }),
        data.length < 5 ? Plot.dot(data, { x: "day", y: "projected_wins_median", fill: t.accent, r: 4 }) : null,
        Plot.ruleX(data, Plot.pointerX({ x: "day", stroke: t.rule })),
        Plot.dot(data, Plot.pointerX({ x: "day", y: "projected_wins_median", fill: t.accent, stroke: t.surface, strokeWidth: 2, r: 5 })),
        Plot.tip(data, Plot.pointerX({ x: "day", y: "projected_wins_median", title: tip })),
      ],
    }),
  );
  document.getElementById("table-history").replaceChildren(
    table(
      ["Date", "Record", "Median wins", "80% range", "Playoffs", "Top 6", "#1 seed"],
      data.map((d) => [d.date, d.record, String(d.projected_wins_median), `${d.p10}–${d.p90}`, pct(d.playoff_prob), pct(d.top6_prob), pct(d.seed1_prob)]),
    ),
  );
}

function probBar(p) {
  return el(
    "span",
    { class: "prob" },
    el("span", { class: "prob-track" }, el("span", { class: "prob-fill", style: `width:${Math.round(p * 100)}%` })),
    el("span", {}, pct(p)),
  );
}

function renderSchedule(pred) {
  const where = (g) => `${g.home ? "vs" : "@"} ${g.opponent}`;
  const rest = (g) => (g.b2b ? "Back-to-back" : `${Math.round(g.rest_days)}d rest`);
  const upcoming = pred.games.filter((g) => !g.completed);
  const done = pred.games.filter((g) => g.completed).reverse();
  document.getElementById("upcoming").replaceChildren(
    upcoming.length
      ? table(
          ["Date", "Opponent", "Rest", "Win probability", "Proj. margin"],
          upcoming.map((g) => [g.date, where(g), rest(g), probBar(g.win_prob), signed(g.pred_margin)]),
        )
      : el("p", { class: "muted" }, "Regular season complete."),
  );
  document.getElementById("completed").replaceChildren(
    done.length
      ? table(
          ["Date", "Opponent", "Result", "Pre-game win prob."],
          done.map((g) => [g.date, where(g), g.result, pct(g.pregame_win_prob)]),
        )
      : el("p", { class: "muted" }, "No games played yet."),
  );
}

function renderWins(pred, t) {
  const box = document.getElementById("chart-wins");
  const entries = Object.entries(pred.team_summary.win_distribution).map(([w, p]) => ({ wins: +w, p }));
  const byWins = new Map(entries.map((d) => [d.wins, d.p]));
  const lo = d3.min(entries, (d) => d.wins);
  const hi = d3.max(entries, (d) => d.wins);
  const data = d3.range(lo, hi + 1).map((w) => ({ wins: w, p: byWins.get(w) ?? 0 }));
  box.replaceChildren(
    Plot.plot({
      width: box.clientWidth,
      height: 240,
      marginLeft: 44,
      style: plotStyle(t),
      x: { type: "band", label: "Final wins", padding: 0.1, tickFormat: (w) => (w % 5 === 0 ? w : "") },
      y: { label: "Share of simulations", grid: true, tickFormat: "%" },
      marks: [
        Plot.barY(data, { x: "wins", y: "p", fill: t.accent }),
        Plot.ruleY([0], { stroke: t.rule }),
        Plot.tip(data, Plot.pointerX({ x: "wins", y: "p", title: (d) => `${d.wins} wins: ${pct1(d.p)}` })),
      ],
    }),
  );
  document.getElementById("table-wins").replaceChildren(
    table(["Wins", "Probability"], entries.map((d) => [String(d.wins), pct1(d.p)])),
  );
}

function renderSeeds(pred, t) {
  const box = document.getElementById("chart-seeds");
  const sp = pred.seed_probabilities;
  const data = d3.range(1, 11).map((k) => ({ label: String(k), p: sp[String(k)] }));
  data.push({ label: "11–15", p: d3.sum(d3.range(11, 16), (k) => sp[String(k)]) });
  box.replaceChildren(
    Plot.plot({
      width: box.clientWidth,
      height: 220,
      marginLeft: 44,
      style: plotStyle(t),
      x: { type: "band", label: "East seed", padding: 0.2, domain: data.map((d) => d.label) },
      y: { label: "Probability", grid: true, tickFormat: "%" },
      marks: [
        Plot.barY(data, { x: "label", y: "p", fill: t.accent }),
        Plot.ruleY([0], { stroke: t.rule }),
        Plot.tip(data, Plot.pointerX({ x: "label", y: "p", title: (d) => `Seed ${d.label}: ${pct1(d.p)}` })),
      ],
    }),
  );
  document.getElementById("table-seeds").replaceChildren(
    table(["Seed", "Probability"], data.map((d) => [d.label, pct1(d.p)])),
  );
}

function renderEast(pred) {
  const rows = pred.conference_table;
  document.getElementById("east").replaceChildren(
    table(
      ["Team", "Record", "Proj. wins", "Playoffs", "Top 6", "#1 seed"],
      rows.map((r) => [r.team, `${r.wins}-${r.losses}`, r.proj_wins.toFixed(1), pct(r.p_playoffs), pct(r.p_top6), pct(r.p_seed1)]),
      rows.map((r) => r.team === pred.team_summary.team),
    ),
  );
}

const MODEL_NAMES = { homecourt: "Home court only", elo: "Elo only", ridge: "Ridge regression", gbm: "Gradient boosting" };

function renderReportCard(pred, backtest, t) {
  const live = pred.metrics.live;
  document.getElementById("live").textContent = live
    ? `This season so far: ${live.n} Celtics games, ${pct1(live.accuracy)} correct, log loss ${live.log_loss.toFixed(3)}.`
    : "Live accuracy appears once predicted games have been played.";
  if (!backtest) {
    document.getElementById("backtest-note").textContent = "Backtest results not available yet.";
    return;
  }
  const seasons = backtest.test_seasons;
  document.getElementById("backtest-note").textContent =
    `Walk-forward backtest, ${seasons[0]}-${String(seasons[0] + 1).slice(2)} through ${seasons.at(-1)}-${String(seasons.at(-1) + 1).slice(2)} seasons. Lower log loss and Brier are better.`;
  const entries = Object.entries(backtest.models);
  document.getElementById("models").replaceChildren(
    table(
      ["Model", "Log loss", "Brier", "Accuracy", "Games"],
      entries.map(([k, m]) => [
        (MODEL_NAMES[k] ?? k) + (k === backtest.main_model ? " (live model)" : ""),
        m.log_loss.toFixed(4),
        m.brier.toFixed(4),
        pct1(m.accuracy),
        m.n.toLocaleString(),
      ]),
      entries.map(([k]) => k === backtest.main_model),
    ),
  );
  const box = document.getElementById("chart-calibration");
  const cal = backtest.calibration;
  const size = Math.min(box.clientWidth, 380);
  box.replaceChildren(
    Plot.plot({
      width: size,
      height: size,
      marginLeft: 44,
      style: plotStyle(t),
      x: { domain: [0, 1], label: "Predicted home win probability", tickFormat: "%", grid: true },
      y: { domain: [0, 1], label: "Actual home win rate", tickFormat: "%", grid: true },
      marks: [
        Plot.line([[0, 0], [1, 1]], { stroke: t.rule }),
        Plot.dot(cal, { x: "mean_pred", y: "observed", r: 5, fill: t.accent, stroke: t.surface, strokeWidth: 2 }),
        Plot.tip(cal, Plot.pointer({ x: "mean_pred", y: "observed", title: (d) => `Predicted ${pct1(d.mean_pred)}\nActual ${pct1(d.observed)}\n${d.n.toLocaleString()} games` })),
      ],
    }),
  );
  document.getElementById("table-calibration").replaceChildren(
    table(["Bucket", "Mean predicted", "Actual", "Games"], cal.map((d) => [`${pct(d.bin_lo)}–${pct(d.bin_hi)}`, pct1(d.mean_pred), pct1(d.observed), d.n.toLocaleString()])),
  );
}

async function loadJson(path) {
  const resp = await fetch(path, { cache: "no-store" });
  if (!resp.ok) throw new Error(`${path}: HTTP ${resp.status}`);
  return resp.json();
}

let state = null;

function renderAll() {
  if (!state) return;
  const t = theme();
  const { pred, history, backtest } = state;
  renderHeadline(pred);
  renderHistory(history, t);
  renderSchedule(pred);
  renderWins(pred, t);
  renderSeeds(pred, t);
  renderEast(pred);
  renderReportCard(pred, backtest, t);
}

async function main() {
  try {
    const [pred, history, backtest] = await Promise.all([
      loadJson("data/predictions.json"),
      loadJson("data/history.json").catch(() => []),
      loadJson("data/backtest.json").catch(() => null),
    ]);
    state = { pred, history, backtest };
    renderAll();
  } catch (err) {
    document.getElementById("updated").textContent = `Could not load predictions (${err.message}).`;
    return;
  }
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
  let timer;
  window.addEventListener("resize", () => {
    clearTimeout(timer);
    timer = setTimeout(renderAll, 150);
  });
}

main();
```

- [ ] **Step 4: Generate sample data and look at the page**

Run:
```bash
uv run python - <<'EOF'
from datetime import date
from pathlib import Path
import tempfile, shutil
from celtics.config import Paths
from celtics.data.cache import season_path
from celtics.pipeline import run_backtest, run_update
from tests.synth import make_league

tmp = Path(tempfile.mkdtemp())
p = Paths(tmp)
p.raw_dir.mkdir(parents=True)
for s, g in make_league([2021, 2022, 2023, 2024], current=2024).groupby("season"):
    g.to_parquet(season_path(p.raw_dir, s), index=False)
run_backtest(p, date(2024, 11, 20), offline=True, test_seasons=[2023], n_sims=200)
run_update(p, date(2024, 11, 20), offline=True, n_sims=2000)
run_update(p, date(2024, 11, 21), offline=True, n_sims=2000)
for f in ["predictions.json", "history.json", "backtest.json"]:
    shutil.copy(p.site_data / f, Path("site/data") / f)
EOF
cd site && python3 -m http.server 8000
```
Open http://localhost:8000. Check in both light and dark mode (toggle macOS appearance) and at phone width (~400px in the browser dev tools):
- every chart renders with the green accent and readable axes
- hover tooltips work on the history, distribution, seed and calibration charts
- "Show as table" opens a table for each chart
- BOS is highlighted in the East table
- no horizontal page scroll at 400px (tables scroll inside their own box)

Fix anything that looks wrong, then delete the sample data: `rm site/data/predictions.json site/data/history.json site/data/backtest.json`.

- [ ] **Step 5: Commit**

```bash
git add site/index.html site/styles.css site/app.js
git commit -m "feat: static projection website"
```

---

### Task 17: Daily automation (GitHub Actions + Pages + launchd fallback)

**Files:**
- Create: `.github/workflows/pages.yml`, `.github/workflows/daily.yml`, `scripts/run_update.sh`, `scripts/com.celtics.update.plist.template`, `scripts/install_launchd.sh`

- [ ] **Step 1: Write `.github/workflows/pages.yml`**

```yaml
name: Deploy site
on:
  push:
    branches: [main]
    paths: ["site/**"]
  workflow_dispatch:
  workflow_call:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: false
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
        with:
          ref: main
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Write `.github/workflows/daily.yml`**

Commits pushed with `GITHUB_TOKEN` don't trigger other workflows, so this workflow calls `pages.yml` directly after it commits.

```yaml
name: Daily update
on:
  schedule:
    - cron: "0 14 * * *" # 9:00 ET (EST) / 10:00 ET (EDT), after all West Coast games
  workflow_dispatch:
permissions:
  contents: write
  pages: write
  id-token: write
concurrency:
  group: daily-update
  cancel-in-progress: false
jobs:
  update:
    runs-on: ubuntu-latest
    outputs:
      changed: ${{ steps.commit.outputs.changed }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - uses: actions/cache@v4
        with:
          path: data
          key: nba-data-${{ github.run_id }}
          restore-keys: nba-data-
      - run: uv sync --frozen
      - name: Run pipeline
        run: uv run celtics update
      - id: commit
        name: Commit site data
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add site/data
          if git diff --cached --quiet; then
            echo "changed=false" >> "$GITHUB_OUTPUT"
          else
            git commit -m "data: daily update $(date -u +%F)"
            git push
            echo "changed=true" >> "$GITHUB_OUTPUT"
          fi
  deploy:
    needs: update
    if: needs.update.outputs.changed == 'true'
    uses: ./.github/workflows/pages.yml
```

If `celtics update` exits with code 2 ("fetch blocked"), the job fails, nothing is committed, and the site keeps its last good data. GitHub emails the repo owner about failed scheduled runs.

- [ ] **Step 3: Write `scripts/run_update.sh`**

```bash
#!/usr/bin/env bash
# Local fallback for the daily update (NBA.com does not block home IPs).
set -euo pipefail
cd "$(dirname "$0")/.."
git pull --rebase --quiet
uv run celtics update
git add site/data
if ! git diff --cached --quiet; then
  git commit -m "data: local update $(date +%F)"
  git push
fi
```

Run: `chmod +x scripts/run_update.sh`

- [ ] **Step 4: Write `scripts/com.celtics.update.plist.template`**

`__REPO__` and `__HOME__` are filled in by `install_launchd.sh`.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.celtics.update</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>__REPO__/scripts/run_update.sh</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>__HOME__/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>10</integer>
    <key>Minute</key>
    <integer>30</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>__HOME__/Library/Logs/celtics-update.log</string>
  <key>StandardErrorPath</key>
  <string>__HOME__/Library/Logs/celtics-update.log</string>
</dict>
</plist>
```

- [ ] **Step 5: Write `scripts/install_launchd.sh`**

```bash
#!/usr/bin/env bash
# Install (or reinstall) the daily local update job. Usage: scripts/install_launchd.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$HOME/Library/LaunchAgents/com.celtics.update.plist"
mkdir -p "$HOME/Library/LaunchAgents"
sed -e "s|__REPO__|$REPO|g" -e "s|__HOME__|$HOME|g" \
  "$REPO/scripts/com.celtics.update.plist.template" > "$TARGET"
launchctl unload "$TARGET" 2>/dev/null || true
launchctl load "$TARGET"
echo "Installed $TARGET (runs daily at 10:30). Logs: ~/Library/Logs/celtics-update.log"
```

Run: `chmod +x scripts/install_launchd.sh && bash -n scripts/run_update.sh && bash -n scripts/install_launchd.sh && plutil -lint scripts/com.celtics.update.plist.template`
Expected: no syntax errors; `OK` from plutil.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/pages.yml .github/workflows/daily.yml scripts
git commit -m "ci: daily update workflow, Pages deploy, launchd fallback"
```

---

### Task 18: First real run, model choice, README

**Files:**
- Modify: `README.md`, `src/celtics/config.py` (only if gradient boosting wins), `docs/superpowers/specs/2026-11-12-celtics-prediction-design.md` (record the results)
- Create: `site/data/backtest.json`, `site/data/predictions.json`, `site/data/history.json`, `site/data/pregame_log.json` (generated)

- [ ] **Step 1: Fetch real data (from your Mac, not the cloud)**

Run: `uv run celtics fetch`
Expected: `cached N games`, with N ≈ 15,000. The first run makes 13 slow requests to NBA.com, so allow a few minutes. If it fails with "fetch blocked", wait a few minutes and run it again; cached seasons are skipped.

- [ ] **Step 2: Run the backtest**

Run: `uv run celtics backtest --offline`
Expected: one line per model, then coverage. Sanity checks:
- ridge log loss is below elo's, which is below homecourt's (homecourt ≈ 0.68)
- ridge accuracy is about 0.64–0.68 and log loss about 0.60–0.63
- `chosen_team_sd` is between 1.0 and 4.0

If ridge doesn't beat elo, stop and investigate (likely a leakage or feature bug) before going further.

- [ ] **Step 3: Choose the live model**

If `gbm` log loss is lower than `ridge` by at least 0.002, set `MODEL_NAME = "gbm"` in `src/celtics/config.py` and rerun `uv run celtics backtest --offline`. Otherwise keep `ridge`.

- [ ] **Step 4: First real update**

Run: `uv run celtics update --offline --force-train`
Expected: a line like `BOS 9-4 | proj 52 wins (80%: 47-57) | playoffs 93%`. Check that the record matches the Celtics' actual record today. Then serve the site and look at it:
```bash
cd site && python3 -m http.server 8000
```

- [ ] **Step 5: Rewrite `README.md`**

````markdown
# Boston Celtics 2026-27 Prediction Model

Daily-updated projections for the Boston Celtics: win probability for every remaining game, the distribution of final wins, seed probabilities, and playoff / play-in odds.

**Live site:** https://locnhoang.github.io/BostonCelticsNBAPrediction/

## How it works

1. **Data:** every regular-season game since 2014-15 from NBA.com (`nba_api`) plus the NBA CDN schedule, cached locally as parquet.
2. **Features** (all point-in-time, so there's no leakage): margin-of-victory Elo, recency-weighted net rating per 100 possessions with a preseason prior, home court, rest days, back-to-backs, three-in-four.
3. **Model:** ridge regression predicts the point margin; win probability = Φ(margin / σ). It's compared against Elo-only, home-court-only, and gradient boosting in a walk-forward backtest (2018-19 → 2025-26).
4. **Simulation:** 10,000 runs of the full league's remaining schedule, with correlated team-strength uncertainty (tuned so 80% intervals cover about 80% of outcomes), simplified tiebreakers, and the play-in.
5. **Site:** static HTML/JS on GitHub Pages reading JSON written by the pipeline.

## Backtest results

| Model | Log loss | Brier | Accuracy |
|---|---|---|---|
| Home court only | FROM_BACKTEST | | |
| Elo only | | | |
| Ridge (live) | | | |
| Gradient boosting | | | |

## Usage

```bash
uv sync
uv run celtics fetch        # refresh data cache
uv run celtics backtest     # walk-forward backtest -> site/data/backtest.json
uv run celtics update       # daily run -> site/data/*.json
uv run pytest               # tests (offline)
```

## Automation

- `.github/workflows/daily.yml` runs `celtics update` at 14:00 UTC and deploys the site.
- NBA.com sometimes blocks cloud IPs. If the daily job fails with "fetch blocked", install the local fallback with `scripts/install_launchd.sh` (runs daily at 10:30 from your Mac).

## Limitations

- Team-level only: injuries and trades show up only through results (player-level adjustments are planned).
- Tiebreakers are simplified (head-to-head → division winner → conference record → random).
- The two NBA Cup–determined games per team are simulated against an average opponent until the NBA schedules them.

Not affiliated with the NBA or the Boston Celtics.
````

Replace the `FROM_BACKTEST` row with the actual numbers printed in Step 2 (round to 4 decimals for log loss and Brier, 1 decimal % for accuracy), and fill in all four rows.

- [ ] **Step 6: Record results in the spec**

Append a `## Results (first run)` section to the spec with the backtest table, the chosen model, and `chosen_team_sd`.

- [ ] **Step 7: Commit and push**

```bash
git add README.md docs site/data src/celtics/config.py
git commit -m "docs: README with backtest results; first real predictions"
git push origin main
```

- [ ] **Step 8: Turn on GitHub Pages and test the daily job**

These steps are for the repo owner, in the GitHub UI:
1. Repo → Settings → Pages → Source: **GitHub Actions**.
2. Actions → "Deploy site" → Run workflow. Confirm the site loads at `https://locnhoang.github.io/BostonCelticsNBAPrediction/`.
3. Actions → "Daily update" → Run workflow.
   - If it succeeds, automation is live.
   - If it fails with `fetch blocked`, run `scripts/install_launchd.sh` on the Mac to make the local job the daily path.
