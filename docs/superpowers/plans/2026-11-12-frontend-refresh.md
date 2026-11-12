# Front-End Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the site so it leads with the next game. It adds team context and a "why the model leans this way" breakdown, shows team logos, and drops the 50+ wins and top-6 displays.

**Architecture:**
- **Pipeline:** two new export pieces.
  - `explain()` splits each upcoming game's predicted margin into four additive drivers. It is exact for linear models.
  - `team_context()` gives standings, form and efficiency ranks for all 30 teams.
  - Both are schema-validated and added to `predictions.json`.
- **Logos:** downloaded once by a script and committed under `site/logos/`.
- **Website:** the static site (`index.html`, `styles.css`, `app.js`) is rewritten against the new JSON fields.

**Tech Stack:** Python 3.12 (uv, pandas 3.1, scikit-learn, pydantic 2, requests), plain HTML/CSS/JS, d3 7.9.0 + Observable Plot 0.6.16 via jsDelivr.

**Spec:** `docs/superpowers/specs/2026-11-12-frontend-refresh-design.md`

## Global Constraints

- **Scope:** no changes to the model, features, simulation or backtest. This plan only adds export fields and front end.
- **Validation:** every new JSON field is validated by the pydantic schema before writing.
- **`drivers`:** labels and order are exactly `Season strength`, `Recent form`, `Home court`, `Rest & schedule`. Values are from the tracked team's (BOS) perspective, rounded to 0.1. They are `null` for completed games and when the live model is not linear.
- **`team_context`:** keyed by all 30 team abbreviations, computed from current-season completed games only. Teams with no completed games rank after every team that has played, ordered by abbreviation.
- **Display:**
  - `p_50_wins`, `p_top6` and `top6_prob` stay in the JSON but are no longer displayed anywhere.
  - Logos are `site/logos/{ABBR}.png`, at most 128px, valid PNG, committed. If a logo image fails to load, a team-color badge is shown instead.
  - Footer credit (exact text): "Team logos are trademarks of the NBA and its teams. Not affiliated with the NBA or the Boston Celtics."
  - Celtics green is the only accent. Driver bars are green when they help Boston and neutral gray when they hurt.
  - Light and dark mode. Every chart has a table view. No horizontal page scroll at 400px.
  - Tables hide secondary columns below 560px. Win distribution and seeds sit side by side at 760px and wider.
- **Environment:**
  - Run everything with `uv run ...` after `export PATH="$HOME/.local/bin:$PATH"`.
  - Before every commit: `uv run ruff check --fix . && uv run ruff check .` and `uv run pytest -q`.
  - Never `git add` the untracked `.claude/` directory. Never push.

## File Map

```
src/celtics/models/margin.py      # + is_linear class attribute
src/celtics/models/explain.py     # NEW: explain(model, X) -> DataFrame | None
src/celtics/export/context.py     # NEW: team_context(games, season) -> dict
src/celtics/export/build.py       # game_rows adds "drivers"
src/celtics/export/schema.py      # + Driver, TeamContext; GameRow.drivers; Predictions.team_context
src/celtics/pipeline.py           # run_update adds team_context
scripts/fetch_logos.py            # NEW: one-time logo download
site/logos/*.png                  # NEW: 30 logos (committed)
site/index.html, site/styles.css, site/app.js   # rewritten
tests/test_explain.py, tests/test_context.py, tests/test_logos.py   # NEW
tests/test_export.py, tests/test_pipeline.py    # extended
```

---

### Task 1: Prediction breakdown (`explain`)

**Files:**
- Modify: `src/celtics/models/margin.py` (add `is_linear` to the classes)
- Create: `src/celtics/models/explain.py`
- Test: `tests/test_explain.py`

**Interfaces:**
- Consumes: `FEATURES` from `celtics.features.build`; `DEFAULT_REST` from `celtics.config`; `MarginModel` and `make_model` from `celtics.models.margin`.
- Produces:
  - `MarginModel.is_linear: bool` class attribute (False on the base and `GBMModel`; True on `HomeCourtModel`, `EloModel`, `RidgeModel`).
  - `celtics.models.explain.DRIVER_ORDER: list[str]`.
  - `celtics.models.explain.explain(model, X: pd.DataFrame) -> pd.DataFrame | None`. It returns home-perspective contributions with columns `DRIVER_ORDER` and the same index as `X`. Each row sums to `model.predict_margin(X)`. It returns `None` when `model.is_linear` is false.

- [ ] **Step 1: Write the failing test `tests/test_explain.py`**

```python
import numpy as np
import pandas as pd
import pytest

from celtics.features.build import FEATURES
from celtics.models.explain import DRIVER_ORDER, explain
from celtics.models.margin import make_model


def synthetic_xy(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(0.0, index=range(n), columns=FEATURES)
    X["elo_diff"] = rng.normal(0, 150, n)
    X["net_diff"] = X["elo_diff"] / 28 + rng.normal(0, 2, n)
    X["home_rest"] = rng.integers(1, 5, n).astype(float)
    X["away_rest"] = rng.integers(1, 5, n).astype(float)
    X["home_b2b"] = (X["home_rest"] == 1).astype(float)
    X["away_b2b"] = (X["away_rest"] == 1).astype(float)
    y = pd.Series((X["elo_diff"] + 100) / 28 + 0.8 * (X["home_rest"] - X["away_rest"]) + rng.normal(0, 12, n))
    return X, y


def test_ridge_parts_sum_to_prediction():
    X, y = synthetic_xy()
    model = make_model("ridge").fit(X, y)
    parts = explain(model, X.iloc[:50])
    assert list(parts.columns) == DRIVER_ORDER
    assert list(parts.index) == list(X.index[:50])
    np.testing.assert_allclose(parts.sum(axis=1), model.predict_margin(X.iloc[:50]), atol=1e-9)


def test_home_court_is_equal_teams_on_normal_rest():
    X, y = synthetic_xy()
    model = make_model("elo").fit(X, y)
    parts = explain(model, X.iloc[:3])
    assert parts["Home court"].tolist() == pytest.approx([100 / 28] * 3)
    assert parts["Season strength"].tolist() == pytest.approx((X["elo_diff"].iloc[:3] / 28).tolist())
    assert parts["Recent form"].abs().max() == 0
    assert parts["Rest & schedule"].abs().max() == 0


def test_rest_part_is_zero_on_normal_rest():
    X, y = synthetic_xy()
    model = make_model("ridge").fit(X, y)
    row = X.iloc[[0]].copy()
    row[["home_rest", "away_rest"]] = 2.0
    row[["home_b2b", "away_b2b", "home_3in4", "away_3in4"]] = 0.0
    assert explain(model, row)["Rest & schedule"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_nonlinear_model_has_no_breakdown():
    X, y = synthetic_xy(500)
    model = make_model("gbm").fit(X, y)
    assert model.is_linear is False
    assert explain(model, X.iloc[:5]) is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_explain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.models.explain'`

- [ ] **Step 3: Add `is_linear` in `src/celtics/models/margin.py`**

In `class MarginModel`, add `is_linear = False` on the line after `name = "base"`. In `HomeCourtModel`, `EloModel` and `RidgeModel`, add `is_linear = True` on the line after each class's `name = ...` line. Leave `GBMModel` alone; it inherits `False`. For example:

```python
class MarginModel:
    name = "base"
    is_linear = False
```

```python
class RidgeModel(MarginModel):
    name = "ridge"
    is_linear = True
```

- [ ] **Step 4: Write `src/celtics/models/explain.py`**

```python
"""Split a predicted margin into additive drivers (exact for linear models)."""

import pandas as pd

from celtics.config import DEFAULT_REST
from celtics.features.build import FEATURES
from celtics.models.margin import MarginModel

DRIVER_ORDER = ["Season strength", "Recent form", "Home court", "Rest & schedule"]

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
}
GROUPS = {
    "Season strength": ["elo_diff"],
    "Recent form": ["net_diff"],
    "Rest & schedule": ["home_rest", "away_rest", "home_b2b", "away_b2b", "home_3in4", "away_3in4"],
}


def explain(model: MarginModel, X: pd.DataFrame) -> pd.DataFrame | None:
    """Home-perspective contribution of each driver; rows sum to model.predict_margin(X)."""
    if not getattr(model, "is_linear", False):
        return None
    base = pd.DataFrame([BASELINE] * len(X), columns=FEATURES)
    home_court = model.predict_margin(base)
    parts = {"Home court": home_court}
    for label, cols in GROUPS.items():
        swapped = base.copy()
        swapped[cols] = X[cols].to_numpy()
        parts[label] = model.predict_margin(swapped) - home_court
    return pd.DataFrame(parts, index=X.index)[DRIVER_ORDER]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_explain.py tests/test_models.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/celtics/models/margin.py src/celtics/models/explain.py tests/test_explain.py
git commit -m "feat(models): additive driver breakdown for linear margin models"
```

---

### Task 2: Team context

**Files:**
- Create: `src/celtics/export/context.py`
- Test: `tests/test_context.py`

**Interfaces:**
- Consumes: the canonical games frame (`game_id, season, date, home_team, away_team, home_pts, away_pts, poss`); `TEAMS` and `CONFERENCE_OF` from `celtics.teams`; `make_games` from `tests.factories`.
- Produces: `celtics.export.context.team_context(games: pd.DataFrame, season: int) -> dict[str, dict]`. It has 30 keys, and each value has the keys `wins, losses, conf_rank, games_back, last10, streak, off_rating, def_rating, net_rating, off_rank, def_rank, net_rank`.

- [ ] **Step 1: Write the failing test `tests/test_context.py`**

```python
import pytest

from celtics.export.context import team_context
from celtics.teams import TEAMS
from tests.factories import make_games

GAMES = make_games(
    [
        ("g1", 2026, "2026-10-22", "BOS", "NYK", 110, 100),
        ("g2", 2026, "2026-10-24", "NYK", "BOS", 105, 100),
        ("g3", 2026, "2026-10-26", "BOS", "MIA", 120, 100),
        ("g4", 2026, "2026-10-28", "BOS", "PHI", 99, 98),
        ("g5", 2026, "2026-10-30", "BOS", "ATL", None, None),  # unplayed: ignored
        ("old", 2025, "2026-03-01", "BOS", "NYK", 50, 150),  # other season: ignored
    ]
)


def test_record_form_and_ratings():
    ctx = team_context(GAMES, 2026)
    assert set(ctx) == set(TEAMS)
    bos = ctx["BOS"]
    assert (bos["wins"], bos["losses"]) == (3, 1)
    assert bos["last10"] == "3-1" and bos["streak"] == "W2"
    assert bos["off_rating"] == pytest.approx(107.2, abs=0.05)  # 429 / 400 poss
    assert bos["def_rating"] == pytest.approx(100.8, abs=0.05)  # 403 / 400 poss
    assert bos["net_rating"] == pytest.approx(6.5, abs=0.05)
    assert ctx["NYK"]["streak"] == "W1"
    assert ctx["ATL"] == {
        "wins": 0, "losses": 0, "conf_rank": 5, "games_back": 1.0, "last10": "0-0", "streak": "",
        "off_rating": 0.0, "def_rating": 0.0, "net_rating": 0.0,
        "off_rank": 5, "def_rank": 5, "net_rank": 5,
    }


def test_conference_rank_and_games_back():
    ctx = team_context(GAMES, 2026)
    assert ctx["BOS"]["conf_rank"] == 1 and ctx["BOS"]["games_back"] == 0.0
    assert ctx["NYK"]["conf_rank"] == 2 and ctx["NYK"]["games_back"] == 1.0
    assert ctx["MIA"]["conf_rank"] == 3 and ctx["MIA"]["games_back"] == 1.5  # 0-1 tie broken by abbreviation
    assert ctx["PHI"]["conf_rank"] == 4
    assert ctx["DAL"]["conf_rank"] == 1 and ctx["DEN"]["conf_rank"] == 2  # West: nobody played; abbreviation order


def test_league_ranks_direction():
    ctx = team_context(GAMES, 2026)
    # offense: BOS 107.25, NYK 102.5, MIA 100, PHI 98 -> higher is better
    assert [ctx[t]["off_rank"] for t in ("BOS", "NYK", "MIA", "PHI")] == [1, 2, 3, 4]
    # defense: PHI 99, BOS 100.75, NYK 105, MIA 120 -> lower is better
    assert [ctx[t]["def_rank"] for t in ("PHI", "BOS", "NYK", "MIA")] == [1, 2, 3, 4]
    # net: BOS +6.5, PHI -1, NYK -2.5, MIA -20
    assert [ctx[t]["net_rank"] for t in ("BOS", "PHI", "NYK", "MIA")] == [1, 2, 3, 4]


def test_last10_uses_latest_ten_games():
    rows = [
        (f"g{i}", 2026, f"2026-11-{i + 1:02d}", "BOS", "NYK", 100 if i < 2 else 110, 105)
        for i in range(12)
    ]  # two losses, then ten wins
    ctx = team_context(make_games(rows), 2026)
    assert ctx["BOS"]["last10"] == "10-0" and ctx["BOS"]["streak"] == "W10"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'celtics.export.context'`

- [ ] **Step 3: Write `src/celtics/export/context.py`**

```python
"""Current-season team context: standings position, recent form, efficiency ratings."""

import pandas as pd

from celtics.teams import CONFERENCE_OF, TEAMS


def _team_lines(games: pd.DataFrame, season: int) -> pd.DataFrame:
    done = games[(games["season"] == season) & games["home_pts"].notna()]
    home = pd.DataFrame(
        {"team": done["home_team"], "date": done["date"], "game_id": done["game_id"],
         "pf": done["home_pts"], "pa": done["away_pts"], "poss": done["poss"]}
    )
    away = pd.DataFrame(
        {"team": done["away_team"], "date": done["date"], "game_id": done["game_id"],
         "pf": done["away_pts"], "pa": done["home_pts"], "poss": done["poss"]}
    )
    return pd.concat([home, away], ignore_index=True).sort_values(["date", "game_id"])


def _streak(won: list[bool]) -> str:
    if not won:
        return ""
    k = 1
    while k < len(won) and won[-k - 1] == won[-1]:
        k += 1
    return f"{'W' if won[-1] else 'L'}{k}"


def _rank(raw: dict[str, dict], key: str, higher_is_better: bool) -> dict[str, int]:
    """League rank 1..30; teams with no games go last, ties broken by abbreviation."""

    def sort_key(t: str):
        played = raw[t]["games"] > 0
        value = raw[t][key]
        return (not played, -value if higher_is_better else value, t)

    return {t: i + 1 for i, t in enumerate(sorted(raw, key=sort_key))}


def team_context(games: pd.DataFrame, season: int) -> dict[str, dict]:
    lines = _team_lines(games, season)
    raw: dict[str, dict] = {}
    for team in TEAMS:
        g = lines[lines["team"] == team]
        won = (g["pf"] > g["pa"]).tolist()
        poss = float(g["poss"].sum())
        off = 100.0 * float(g["pf"].sum()) / poss if poss else 0.0
        de = 100.0 * float(g["pa"].sum()) / poss if poss else 0.0
        last = won[-10:]
        raw[team] = {
            "games": len(won),
            "wins": sum(won),
            "losses": len(won) - sum(won),
            "last10": f"{sum(last)}-{len(last) - sum(last)}",
            "streak": _streak(won),
            "off": off,
            "def": de,
            "net": off - de,
        }

    off_rank = _rank(raw, "off", higher_is_better=True)
    def_rank = _rank(raw, "def", higher_is_better=False)
    net_rank = _rank(raw, "net", higher_is_better=True)

    conf_rank: dict[str, int] = {}
    games_back: dict[str, float] = {}
    for conf in ("East", "West"):
        members = [t for t in TEAMS if CONFERENCE_OF[t] == conf]

        def standing(t: str):
            r = raw[t]
            pct = r["wins"] / r["games"] if r["games"] else 0.0
            return (r["games"] == 0, -pct, -r["wins"], t)

        order = sorted(members, key=standing)
        lw, ll = raw[order[0]]["wins"], raw[order[0]]["losses"]
        for i, t in enumerate(order):
            conf_rank[t] = i + 1
            games_back[t] = ((lw - raw[t]["wins"]) + (raw[t]["losses"] - ll)) / 2

    return {
        t: {
            "wins": raw[t]["wins"],
            "losses": raw[t]["losses"],
            "conf_rank": conf_rank[t],
            "games_back": games_back[t],
            "last10": raw[t]["last10"],
            "streak": raw[t]["streak"],
            "off_rating": round(raw[t]["off"], 1),
            "def_rating": round(raw[t]["def"], 1),
            "net_rating": round(raw[t]["net"], 1),
            "off_rank": off_rank[t],
            "def_rank": def_rank[t],
            "net_rank": net_rank[t],
        }
        for t in TEAMS
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_context.py -v`
Expected: 4 passed. If `ruff` flags the function defined inside the loop (B023), pass `raw` explicitly, e.g. `key=lambda t, raw=raw: ...`. Keep the behavior unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/celtics/export/context.py tests/test_context.py
git commit -m "feat(export): team context (standings, form, efficiency ranks)"
```

---

### Task 3: Wire drivers and team context into the export, schema and pipeline

**Files:**
- Modify: `src/celtics/export/schema.py` (add `Driver`, `TeamContext`; `GameRow.drivers`; `Predictions.team_context`)
- Modify: `src/celtics/export/build.py` (`game_rows` adds `drivers`)
- Modify: `src/celtics/pipeline.py` (`run_update` adds `team_context`)
- Test: `tests/test_export.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `explain`, `DRIVER_ORDER` (Task 1); `team_context` (Task 2).
- Produces:
  - In `predictions.json`, each game row gains `drivers: [{"label": str, "value": float}] | null`.
  - A new top-level `team_context: {ABBR: {...}}` holds 30 teams.
  - Schema classes `Driver` and `TeamContext`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_export.py`:

```python
def test_game_rows_drivers_sum_and_flip(season_rows):
    feats, model = season_rows
    rows = game_rows(feats, model, 2023, "BOS", pregame_log={})
    upcoming = [r for r in rows if not r["completed"]]
    for r in upcoming:
        assert [d["label"] for d in r["drivers"]] == [
            "Season strength", "Recent form", "Home court", "Rest & schedule"
        ]
        assert sum(d["value"] for d in r["drivers"]) == pytest.approx(r["pred_margin"], abs=0.25)
    home = next(r for r in upcoming if r["home"])
    away = next(r for r in upcoming if not r["home"])
    court = {r["game_id"]: next(d["value"] for d in r["drivers"] if d["label"] == "Home court") for r in (home, away)}
    assert court[home["game_id"]] > 0 > court[away["game_id"]]
    assert all(r["drivers"] is None for r in rows if r["completed"])


def test_game_rows_no_drivers_for_nonlinear_model(season_rows):
    feats, _ = season_rows
    train = feats[feats["completed"] & (feats["season"] >= FIRST_TRAIN_SEASON)]
    gbm = make_model("gbm").fit(train[FEATURES], train["margin"])
    rows = game_rows(feats, gbm, 2023, "BOS", pregame_log={})
    assert all(r["drivers"] is None for r in rows)


def test_schema_rejects_bad_team_context():
    from celtics.export.schema import TeamContext

    good = {
        "wins": 3, "losses": 1, "conf_rank": 1, "games_back": 0.0, "last10": "3-1", "streak": "W2",
        "off_rating": 107.2, "def_rating": 100.8, "net_rating": 6.5,
        "off_rank": 1, "def_rank": 2, "net_rank": 1,
    }
    TeamContext.model_validate(good)
    for bad in ({"conf_rank": 16}, {"net_rank": 0}, {"streak": "X3"}, {"last10": "ten"}):
        with pytest.raises(ValidationError):
            TeamContext.model_validate({**good, **bad})
```

In `tests/test_pipeline.py`, add these assertions to `test_run_update_writes_valid_outputs`, right after `Predictions.model_validate(pred)`:

```python
    assert len(pred["team_context"]) == 30
    upcoming = [g for g in pred["games"] if not g["completed"]]
    assert upcoming and all(len(g["drivers"]) == 4 for g in upcoming)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_export.py tests/test_pipeline.py -v`
Expected: the new tests FAIL with `KeyError: 'drivers'`, an `ImportError` for `TeamContext`, and `KeyError: 'team_context'`.

- [ ] **Step 3: Update `src/celtics/export/schema.py`**

Add these classes above `class GameRow`:

```python
class Driver(BaseModel):
    label: str
    value: float


class TeamContext(BaseModel):
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    conf_rank: int = Field(ge=1, le=15)
    games_back: float = Field(ge=0)
    last10: str = Field(pattern=r"^\d+-\d+$")
    streak: str = Field(pattern=r"^([WL]\d+)?$")
    off_rating: float
    def_rating: float
    net_rating: float
    off_rank: int = Field(ge=1, le=30)
    def_rank: int = Field(ge=1, le=30)
    net_rank: int = Field(ge=1, le=30)
```

Add this as the last field of `class GameRow`:

```python
    drivers: list[Driver] | None
```

In `class Predictions`, add the field `team_context: dict[str, TeamContext]` after `metrics: Metrics`, and extend the existing `_seeds` validator so it also checks the team count. The validator becomes:

```python
    @model_validator(mode="after")
    def _seeds(self):
        if list(self.seed_probabilities) != [str(k) for k in range(1, 16)]:
            raise ValueError("seed_probabilities must have keys '1'..'15'")
        _sums_to_one(self.seed_probabilities.values(), "seed_probabilities")
        if len(self.team_context) != 30:
            raise ValueError(f"team_context has {len(self.team_context)} teams, expected 30")
        return self
```

- [ ] **Step 4: Update `game_rows` in `src/celtics/export/build.py`**

Add the import `from celtics.models.explain import DRIVER_ORDER, explain` next to the other `celtics` imports. Then replace the body of `game_rows` from `margins = model.predict_margin(cur[FEATURES])` to the end of the function with:

```python
    margins = model.predict_margin(cur[FEATURES])
    probs = model.win_prob(cur[FEATURES])
    parts = explain(model, cur[FEATURES])
    rows = []
    for idx, row, m, p in zip(cur.index, cur.itertuples(index=False), margins, probs, strict=True):
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
            "drivers": None,
        }
        if row.completed:
            mine, theirs = (row.home_pts, row.away_pts) if home else (row.away_pts, row.home_pts)
            out["result"] = f"{'W' if mine > theirs else 'L'} {int(mine)}-{int(theirs)}"
            logged = pregame_log.get(row.game_id)
            out["pregame_win_prob"] = logged["win_prob"] if logged else None
        else:
            out["win_prob"] = r4(p if home else 1 - p)
            out["pred_margin"] = round(float(m if home else -m), 1)
            if parts is not None:
                sign = 1.0 if home else -1.0
                out["drivers"] = [
                    {"label": label, "value": round(sign * float(parts.at[idx, label]), 1)}
                    for label in DRIVER_ORDER
                ]
        rows.append(out)
    return rows
```

- [ ] **Step 5: Update `run_update` in `src/celtics/pipeline.py`**

Add the import `from celtics.export.context import team_context` next to the other `celtics.export` imports. In the `predictions = {...}` dict in `run_update`, add this entry after `"metrics": {"live": live_metrics(rows)},`:

```python
        "team_context": team_context(games, season),
```

- [ ] **Step 6: Run the full suite**

Run: `uv run ruff check --fix . && uv run ruff check . && uv run pytest -q`
Expected: all pass. Any existing test that builds a full `Predictions` payload by hand must now include `team_context`. Search with `grep -rn "Predictions.model_validate" tests/`; if a hand-built payload exists, add `"team_context": team_context(<its games>, <season>)`, or skip that schema call with a comment.

- [ ] **Step 7: Commit**

```bash
git add src/celtics/export/schema.py src/celtics/export/build.py src/celtics/pipeline.py tests/test_export.py tests/test_pipeline.py
git commit -m "feat(export): drivers on upcoming games and team_context in predictions.json"
```

---

### Task 4: Team logos

**Files:**
- Create: `scripts/fetch_logos.py`
- Create: `site/logos/{ABBR}.png` × 30 (generated, committed)
- Test: `tests/test_logos.py`

**Interfaces:**
- Consumes: `TEAMS` from `celtics.teams`.
- Produces: `site/logos/{ABBR}.png` for every team. Task 5 references them as `logos/{ABBR}.png`.

The source is ESPN's public image server, verified reachable from this machine: `https://a.espncdn.com/combiner/i?img=/i/teamlogos/nba/500/{code}.png&h=128&w=128` returns a 128×128 PNG. NBA.com's CDN returns 403 here. ESPN codes differ from NBA abbreviations for 7 teams: GSW→gs, NOP→no, NYK→ny, SAS→sa, UTA→utah, WAS→wsh, and the rest are lowercase.

- [ ] **Step 1: Write the failing test `tests/test_logos.py`**

```python
from pathlib import Path

import pytest

from celtics.teams import TEAMS

LOGOS = Path(__file__).resolve().parent.parent / "site" / "logos"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("team", TEAMS)
def test_logo_is_small_valid_png(team):
    path = LOGOS / f"{team}.png"
    assert path.exists(), f"missing {path}"
    data = path.read_bytes()
    assert data.startswith(PNG_SIGNATURE)
    assert len(data) < 64_000
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    assert width <= 128 and height <= 128
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_logos.py -q`
Expected: 30 failed (`missing .../site/logos/ATL.png`, and so on).

- [ ] **Step 3: Write `scripts/fetch_logos.py`**

```python
"""One-time download of 128px team logos into site/logos/{ABBR}.png.

Team logos are trademarks of the NBA and its teams; they are shown on the site
under the credit in the footer. Run: uv run python scripts/fetch_logos.py
"""

import sys
from pathlib import Path

import requests

from celtics.teams import TEAMS

URL = "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nba/500/{code}.png&h=128&w=128"
ESPN_CODE = {"GSW": "gs", "NOP": "no", "NYK": "ny", "SAS": "sa", "UTA": "utah", "WAS": "wsh"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
OUT = Path(__file__).resolve().parent.parent / "site" / "logos"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for team in TEAMS:
        code = ESPN_CODE.get(team, team.lower())
        resp = requests.get(URL.format(code=code), timeout=30)
        if resp.status_code != 200 or not resp.content.startswith(PNG_SIGNATURE):
            failed.append(f"{team} ({code}): HTTP {resp.status_code}")
            continue
        (OUT / f"{team}.png").write_bytes(resp.content)
        print(f"{team}: {len(resp.content)} bytes")
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the script, then the test**

Run: `uv run python scripts/fetch_logos.py && uv run pytest tests/test_logos.py -q`
Expected: 30 lines of `ABBR: N bytes`, then 30 passed. If a team fails, check its ESPN code by opening `https://a.espncdn.com/i/teamlogos/nba/500/<code>.png`. Fix only `ESPN_CODE`.

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_logos.py site/logos tests/test_logos.py
git commit -m "feat(site): add 128px team logos and the script that fetched them"
```

---

### Task 5: Rebuild the website

**Files:**
- Rewrite: `site/index.html`, `site/styles.css`, `site/app.js`

**Interfaces:**
- Consumes: `predictions.json` (incl. `team_context`, `games[].drivers`), `history.json`, `backtest.json`, and `site/logos/{ABBR}.png`.

- [ ] **Step 1: Replace `site/index.html`**

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
    <header class="page-header">
      <span id="hdr-logo"></span>
      <div>
        <h1 id="hdr-title">Boston Celtics</h1>
        <p id="hdr-meta" class="hdr-meta"></p>
        <p id="updated" class="muted">Loading…</p>
      </div>
    </header>

    <section aria-labelledby="h-next">
      <h2 id="h-next">Next game</h2>
      <p class="subtitle">Win chance from the model, and what it is based on.</p>
      <div id="next-card"></div>
      <div id="drivers-wrap">
        <h3>Why the model leans this way</h3>
        <p class="subtitle">Points each factor adds to Boston's projected margin. Green helps, gray hurts.</p>
        <div id="chart-drivers" class="chart"></div>
        <details><summary>Show as table</summary><div id="table-drivers" class="table-wrap"></div></details>
      </div>
    </section>

    <section aria-labelledby="h-outlook">
      <h2 id="h-outlook">Season outlook</h2>
      <p class="subtitle">From 10,000 simulations of the rest of the season.</p>
      <div id="outlook" class="tiles"></div>
    </section>

    <section aria-labelledby="h-context">
      <h2 id="h-context">Team context</h2>
      <p class="subtitle">Where Boston stands right now, from this season's games.</p>
      <div id="context"></div>
    </section>

    <section aria-labelledby="h-schedule">
      <h2 id="h-schedule">Schedule</h2>
      <div class="tabs" role="tablist">
        <button role="tab" id="tab-upcoming" aria-controls="upcoming" aria-selected="true">Upcoming</button>
        <button role="tab" id="tab-results" aria-controls="completed" aria-selected="false">Results</button>
      </div>
      <p class="subtitle">Select an upcoming game to see its breakdown.</p>
      <div id="upcoming" class="table-wrap" role="tabpanel" aria-labelledby="tab-upcoming"></div>
      <div id="completed" class="table-wrap" role="tabpanel" aria-labelledby="tab-results" hidden></div>
    </section>

    <section aria-labelledby="h-history">
      <h2 id="h-history">Projected wins over time</h2>
      <p class="subtitle">Median projection with the 80% range, one point per daily update.</p>
      <div id="chart-history" class="chart"></div>
      <details><summary>Show as table</summary><div id="table-history" class="table-wrap"></div></details>
    </section>

    <div class="two-col">
      <section aria-labelledby="h-wins">
        <h2 id="h-wins">Final win total</h2>
        <p class="subtitle">Share of simulated seasons ending at each win total.</p>
        <div id="chart-wins" class="chart"></div>
        <details><summary>Show as table</summary><div id="table-wins" class="table-wrap"></div></details>
      </section>
      <section aria-labelledby="h-seeds">
        <h2 id="h-seeds">East seed</h2>
        <p class="subtitle">Seeds 7–10 go to the play-in; 11–15 miss the postseason.</p>
        <div id="chart-seeds" class="chart"></div>
        <details><summary>Show as table</summary><div id="table-seeds" class="table-wrap"></div></details>
      </section>
    </div>

    <section aria-labelledby="h-east">
      <h2 id="h-east">Eastern Conference projection</h2>
      <p class="subtitle">Current record and simulated outlook for every East team.</p>
      <div id="east" class="table-wrap"></div>
    </section>

    <section aria-labelledby="h-report">
      <h2 id="h-report">Model report card</h2>
      <p id="live" class="lead"></p>
      <p id="backtest-note" class="subtitle"></p>
      <div id="models" class="table-wrap"></div>
      <details>
        <summary>Calibration</summary>
        <p class="subtitle">When the model says X%, does the home team win X% of the time? Points on the diagonal are perfectly calibrated.</p>
        <div id="chart-calibration" class="chart"></div>
        <div id="table-calibration" class="table-wrap"></div>
      </details>
    </section>

    <section aria-labelledby="h-how">
      <h2 id="h-how">How it works</h2>
      <p id="how-example" class="lead" hidden></p>
      <p>Each game is predicted as a point margin from four things: season-long team strength (a margin-of-victory Elo rating), recent form (a recency-weighted net rating per 100 possessions), home court, and rest (days off, back-to-backs, three games in four nights). The margin becomes a win probability using how much NBA results vary from game to game.</p>
      <p>The rest of the league schedule is simulated 10,000 times. Each simulation shifts every team's strength by a random amount reflecting how uncertain we are about it, then plays out every game, applies tiebreakers (head-to-head, division winner, conference record), and runs the play-in.</p>
      <p>The model is trained on every regular-season game since 2015-16 and tested walk-forward: each test season is predicted using only data from before it. It does not account for injuries yet.</p>
    </section>

    <footer class="muted">
      <p>Data from NBA.com via nba_api. Team logos are trademarks of the NBA and its teams. Not affiliated with the NBA or the Boston Celtics.</p>
      <p><a href="https://github.com/locnhoang/BostonCelticsNBAPrediction">Source code</a></p>
    </footer>
  </main>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Replace `site/styles.css`**

```css
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --text: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #6f6d68;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.1);
  --accent: #007a33;
  --accent-soft: rgba(0, 122, 51, 0.1);
  --neutral-bar: #a8a69e;
  --s1: 4px; --s2: 8px; --s3: 12px; --s4: 16px; --s5: 24px; --s6: 32px;
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
    --neutral-bar: #6a6862;
  }
}

* { box-sizing: border-box; }
[hidden] { display: none !important; }

body {
  margin: 0;
  background: var(--page);
  color: var(--text);
  font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}

main { max-width: 1000px; margin: 0 auto; padding: var(--s6) var(--s4) 64px; }

h1 { font-size: 28px; line-height: 1.2; margin: 0; }
h2 { font-size: 20px; line-height: 1.3; margin: 0; }
h3 { font-size: 16px; margin: var(--s5) 0 0; }
p { margin: 0 0 var(--s3); }
.subtitle { color: var(--muted); font-size: 14px; margin: var(--s1) 0 var(--s4); }
.muted { color: var(--muted); font-size: 14px; }
.lead { font-size: 16px; color: var(--text-secondary); }

.page-header { display: flex; align-items: center; gap: var(--s4); margin-bottom: var(--s5); }
.hdr-meta { font-size: 16px; color: var(--text-secondary); margin: var(--s1) 0 0; }
#updated { margin: var(--s1) 0 0; font-size: 12px; }

section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: var(--s5);
  margin-top: var(--s4);
}

.two-col { display: grid; gap: var(--s4); grid-template-columns: 1fr; }
.two-col > section { margin-top: var(--s4); min-width: 0; }
@media (min-width: 760px) {
  .two-col { grid-template-columns: 1fr 1fr; }
}

/* team marks */
.team { display: inline-flex; align-items: center; gap: var(--s2); }
.logo { display: inline-block; object-fit: contain; flex: none; }
.badge {
  display: inline-flex; align-items: center; justify-content: center; flex: none;
  border-radius: 50%; color: #ffffff; font-weight: 700; letter-spacing: 0.02em;
}

/* next game */
.matchup { display: flex; align-items: center; justify-content: center; gap: var(--s5); flex-wrap: wrap; }
.side { display: flex; flex-direction: column; align-items: center; gap: var(--s1); min-width: 120px; }
.side-name { font-weight: 650; }
.side-rec { color: var(--muted); font-size: 14px; font-variant-numeric: tabular-nums; }
.at { font-size: 20px; color: var(--muted); }
.next-meta { text-align: center; color: var(--text-secondary); font-size: 14px; margin: var(--s3) 0; }
.next-prob { text-align: center; margin-bottom: var(--s3); }
.next-prob .big { display: block; font-size: 40px; font-weight: 700; line-height: 1.1; }
.next-prob .sub { color: var(--text-secondary); font-size: 14px; }
.next-opp { text-align: center; }

/* tiles */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: var(--s3); }
.tile { border: 1px solid var(--border); border-radius: 12px; padding: var(--s3) var(--s4); }
.tile-label { color: var(--text-secondary); font-size: 14px; }
.tile-value { font-size: 28px; font-weight: 650; line-height: 1.2; }
.tile-note { color: var(--muted); font-size: 12px; min-height: 1.2em; }
.tile-change { color: var(--text-secondary); font-size: 12px; margin-top: var(--s1); }

/* team context */
.facts, .ratings { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: var(--s3); }
.ratings { margin-top: var(--s3); }
.fact, .rating { border: 1px solid var(--border); border-radius: 12px; padding: var(--s3) var(--s4); }
.fact-value { font-size: 20px; font-weight: 650; font-variant-numeric: tabular-nums; }
.rating-head { display: flex; justify-content: space-between; align-items: baseline; }
.rating-rank { font-size: 12px; color: var(--text-secondary); }
.rank-strip { display: grid; grid-template-columns: repeat(30, 1fr); gap: 2px; margin: var(--s2) 0 var(--s1); }
.rank-strip span { height: 8px; border-radius: 2px; background: var(--grid); }
.rank-strip span.on { background: var(--accent); }

/* tabs */
.tabs { display: flex; gap: var(--s2); margin-top: var(--s3); }
.tabs button {
  font: inherit; font-size: 14px; padding: var(--s1) var(--s3); border-radius: 999px; cursor: pointer;
  border: 1px solid var(--border); background: transparent; color: var(--text-secondary);
}
.tabs button[aria-selected="true"] { background: var(--accent-soft); color: var(--text); border-color: var(--accent); }

/* tables */
.chart { width: 100%; }
.chart svg { max-width: 100%; height: auto; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: var(--s2); border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--text-secondary); font-weight: 600; }
tr.highlight td { font-weight: 650; background: var(--accent-soft); }
tr.expandable { cursor: pointer; }
tr.expandable:hover td, tr.expandable:focus td { background: var(--accent-soft); }
tr.expandable:focus { outline: 2px solid var(--accent); outline-offset: -2px; }
tr.detail td { color: var(--text-secondary); font-size: 12px; white-space: normal; }
@media (max-width: 559px) {
  .secondary { display: none; }
  section { padding: var(--s4); }
}

.prob { display: inline-flex; align-items: center; gap: var(--s2); }
.prob-track { width: 80px; height: 6px; background: var(--grid); border-radius: 3px; overflow: hidden; }
.prob-fill { display: block; height: 100%; background: var(--accent); border-radius: 3px; }

details { margin-top: var(--s3); }
summary { cursor: pointer; color: var(--text-secondary); font-size: 14px; }

footer { margin-top: var(--s5); font-size: 12px; }
footer a { color: var(--text-secondary); }
```

- [ ] **Step 3: Replace `site/app.js`**

```js
"use strict";

// Nickname and primary color per team; the color backs the badge shown if a logo fails to load.
const TEAMS = {
  ATL: ["Hawks", "#E03A3E"], BKN: ["Nets", "#000000"], BOS: ["Celtics", "#007A33"], CHA: ["Hornets", "#1D1160"],
  CHI: ["Bulls", "#CE1141"], CLE: ["Cavaliers", "#860038"], DAL: ["Mavericks", "#00538C"], DEN: ["Nuggets", "#0E2240"],
  DET: ["Pistons", "#C8102E"], GSW: ["Warriors", "#1D428A"], HOU: ["Rockets", "#CE1141"], IND: ["Pacers", "#002D62"],
  LAC: ["Clippers", "#C8102E"], LAL: ["Lakers", "#552583"], MEM: ["Grizzlies", "#5D76A9"], MIA: ["Heat", "#98002E"],
  MIL: ["Bucks", "#00471B"], MIN: ["Timberwolves", "#0C2340"], NOP: ["Pelicans", "#0C2340"], NYK: ["Knicks", "#006BB6"],
  OKC: ["Thunder", "#007AC1"], ORL: ["Magic", "#0077C0"], PHI: ["76ers", "#006BB6"], PHX: ["Suns", "#1D1160"],
  POR: ["Trail Blazers", "#E03A3E"], SAC: ["Kings", "#5A2D81"], SAS: ["Spurs", "#000000"], TOR: ["Raptors", "#CE1141"],
  UTA: ["Jazz", "#002B5C"], WAS: ["Wizards", "#002B5C"],
};
const teamName = (abbr) => TEAMS[abbr]?.[0] ?? abbr;

const pct = (p) => (p == null ? "—" : `${Math.round(p * 100)}%`);
const pct1 = (p) => `${(p * 100).toFixed(1)}%`;
const signed = (m) => (m > 0 ? `+${m.toFixed(1)}` : m.toFixed(1));
const ord = (n) => {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
};
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

function teamMark(abbr, size = 20) {
  const badge = () =>
    el(
      "span",
      {
        class: "badge",
        style: `background:${TEAMS[abbr]?.[1] ?? "#555555"};width:${size}px;height:${size}px;font-size:${Math.max(8, Math.round(size * 0.3))}px`,
        "aria-hidden": "true",
      },
      abbr,
    );
  const img = el("img", { class: "logo", src: `logos/${abbr}.png`, alt: "", width: size, height: size });
  img.addEventListener("error", () => img.replaceWith(badge()), { once: true });
  return img;
}

const teamLabel = (abbr, size = 20) => el("span", { class: "team" }, teamMark(abbr, size), el("span", {}, abbr));

function table(headers, rows, { highlighted = [], secondary = [] } = {}) {
  const cls = (j) => (secondary.includes(j) ? { class: "secondary" } : {});
  return el(
    "table",
    {},
    el("thead", {}, el("tr", {}, ...headers.map((h, j) => el("th", cls(j), h)))),
    el(
      "tbody",
      {},
      ...rows.map((cells, i) =>
        el("tr", highlighted[i] ? { class: "highlight" } : {}, ...cells.map((c, j) => el("td", cls(j), c))),
      ),
    ),
  );
}

function theme() {
  return {
    accent: token("--accent"),
    neutral: token("--neutral-bar"),
    surface: token("--surface"),
    ink: token("--text-secondary"),
    rule: token("--axis"),
  };
}

function plotStyle(t) {
  return { background: t.surface, color: t.ink, fontSize: "12px", fontFamily: "inherit" };
}

const nextGame = (pred) => pred.games.find((g) => !g.completed) ?? null;

function renderHeader(pred) {
  const team = pred.team_summary.team;
  const c = pred.team_context[team];
  document.getElementById("hdr-logo").replaceChildren(teamMark(team, 28));
  document.getElementById("hdr-title").textContent = `Boston ${teamName(team)} ${pred.season}`;
  document.getElementById("hdr-meta").textContent = `${c.wins}-${c.losses} · ${ord(c.conf_rank)} in the East`;
  document.getElementById("updated").textContent =
    `Updated ${new Date(pred.generated_at).toLocaleString()} · model ${pred.model_version}`;
}

function renderNextGame(pred, t) {
  const card = document.getElementById("next-card");
  const g = nextGame(pred);
  const team = pred.team_summary.team;
  if (!g) {
    card.replaceChildren(el("p", { class: "muted" }, "Regular season complete."));
    renderDrivers(null, t);
    return;
  }
  const ctx = pred.team_context;
  const side = (abbr) =>
    el(
      "div",
      { class: "side" },
      teamMark(abbr, 56),
      el("div", { class: "side-name" }, teamName(abbr)),
      el("div", { class: "side-rec" }, `${ctx[abbr].wins}-${ctx[abbr].losses}`),
    );
  const [away, home] = g.home ? [g.opponent, team] : [team, g.opponent];
  const when = new Date(`${g.date}T12:00:00Z`).toLocaleDateString(undefined, {
    weekday: "short", month: "short", day: "numeric", timeZone: "UTC",
  });
  const opp = ctx[g.opponent];
  card.replaceChildren(
    el("div", { class: "matchup" }, side(away), el("div", { class: "at", "aria-hidden": "true" }, "@"), side(home)),
    el("div", { class: "next-meta" }, `${when} · ${g.home ? "Home" : "Away"}${g.b2b ? " · Back-to-back" : ""}`),
    el(
      "div",
      { class: "next-prob" },
      el("span", { class: "big" }, pct(g.win_prob)),
      el("span", { class: "sub" }, `${teamName(team)} win chance · projected margin ${signed(g.pred_margin)}`),
    ),
    el(
      "p",
      { class: "muted next-opp" },
      `${teamName(g.opponent)}: ${opp.wins}-${opp.losses}, net rating ${ord(opp.net_rank)} of 30, last 10 ${opp.last10}`,
    ),
  );
  renderDrivers(g, t);
}

function renderDrivers(g, t) {
  const wrap = document.getElementById("drivers-wrap");
  if (!g || !g.drivers) {
    wrap.hidden = true;
    return;
  }
  wrap.hidden = false;
  const box = document.getElementById("chart-drivers");
  const data = g.drivers;
  const ext = Math.max(4, ...data.map((d) => Math.abs(d.value))) * 1.3;
  box.replaceChildren(
    Plot.plot({
      width: box.clientWidth,
      height: 40 * data.length + 40,
      marginLeft: 120,
      marginRight: 16,
      style: plotStyle(t),
      x: { domain: [-ext, ext], label: "Points toward Boston", grid: true },
      y: { domain: data.map((d) => d.label), label: null },
      marks: [
        Plot.barX(data, { x: "value", y: "label", fill: (d) => (d.value >= 0 ? t.accent : t.neutral), insetTop: 8, insetBottom: 8 }),
        Plot.ruleX([0], { stroke: t.rule }),
        Plot.text(data.filter((d) => d.value >= 0), { x: "value", y: "label", text: (d) => signed(d.value), textAnchor: "start", dx: 6, fill: t.ink }),
        Plot.text(data.filter((d) => d.value < 0), { x: "value", y: "label", text: (d) => signed(d.value), textAnchor: "end", dx: -6, fill: t.ink }),
        Plot.tip(data, Plot.pointerY({ x: "value", y: "label", title: (d) => `${d.label}: ${signed(d.value)} points` })),
      ],
    }),
  );
  document.getElementById("table-drivers").replaceChildren(
    table(["Factor", "Points"], [...data.map((d) => [d.label, signed(d.value)]), ["Projected margin", signed(g.pred_margin)]]),
  );
}

function change(cur, prev, kind) {
  if (prev == null) return "";
  const d = kind === "wins" ? Math.round(cur - prev) : Math.round((cur - prev) * 100);
  if (d === 0) return "No change since last update";
  const unit = kind === "wins" ? (Math.abs(d) === 1 ? "win" : "wins") : "pts";
  return `${d > 0 ? "▲" : "▼"} ${Math.abs(d)} ${unit} since last update`;
}

function renderOutlook(pred, history) {
  const s = pred.team_summary;
  const prev = history.length >= 2 ? history[history.length - 2] : null;
  const median = Math.round(s.proj_wins_median);
  const tiles = [
    ["Projected record", `${median}-${82 - median}`, `80% range ${Math.round(s.proj_wins_p10)}–${Math.round(s.proj_wins_p90)} wins`, change(s.proj_wins_median, prev?.projected_wins_median, "wins")],
    ["Make playoffs", pct(s.p_playoffs), "includes the play-in", change(s.p_playoffs, prev?.playoff_prob, "pts")],
    ["#1 seed in East", pct(s.p_seed1), "", change(s.p_seed1, prev?.seed1_prob, "pts")],
  ];
  document.getElementById("outlook").replaceChildren(
    ...tiles.map(([label, value, note, delta]) =>
      el(
        "div",
        { class: "tile" },
        el("div", { class: "tile-label" }, label),
        el("div", { class: "tile-value" }, value),
        el("div", { class: "tile-note" }, note),
        delta ? el("div", { class: "tile-change" }, delta) : "",
      ),
    ),
  );
}

function rankStrip(rank, label) {
  return el(
    "div",
    { class: "rank-strip", role: "img", "aria-label": `${label}: ${ord(rank)} of 30 teams` },
    ...Array.from({ length: 30 }, (_, i) => el("span", i + 1 === rank ? { class: "on" } : {})),
  );
}

function renderContext(pred) {
  const team = pred.team_summary.team;
  const c = pred.team_context[team];
  const gb = c.games_back === 0 ? "Leads the conference" : `${c.games_back} ${c.games_back === 1 ? "game" : "games"} back`;
  const facts = [
    ["East standing", ord(c.conf_rank), gb],
    ["Last 10", c.last10, c.streak ? `Current streak: ${c.streak}` : ""],
  ];
  const ratings = [
    ["Offense", c.off_rating.toFixed(1), c.off_rank, "Points scored per 100 possessions"],
    ["Defense", c.def_rating.toFixed(1), c.def_rank, "Points allowed per 100 possessions"],
    ["Net rating", signed(c.net_rating), c.net_rank, "Offense minus defense"],
  ];
  document.getElementById("context").replaceChildren(
    el(
      "div",
      { class: "facts" },
      ...facts.map(([label, value, note]) =>
        el("div", { class: "fact" }, el("div", { class: "tile-label" }, label), el("div", { class: "fact-value" }, value), el("div", { class: "tile-note" }, note)),
      ),
    ),
    el(
      "div",
      { class: "ratings" },
      ...ratings.map(([label, value, rank, note]) =>
        el(
          "div",
          { class: "rating" },
          el("div", { class: "rating-head" }, el("span", { class: "tile-label" }, label), el("span", { class: "rating-rank" }, `${ord(rank)} of 30`)),
          el("div", { class: "fact-value" }, value),
          rankStrip(rank, label),
          el("div", { class: "tile-note" }, `${note} · best on the left`),
        ),
      ),
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

const whereCell = (g) => el("span", { class: "team" }, el("span", { class: "muted" }, g.home ? "vs" : "@"), teamMark(g.opponent, 20), el("span", {}, g.opponent));
const calledIt = (g) => (g.pregame_win_prob >= 0.5) === g.result.startsWith("W");

function renderSchedule(pred) {
  const rest = (g) => (g.b2b ? "Back-to-back" : `${Math.round(g.rest_days)}d rest`);
  const upcoming = pred.games.filter((g) => !g.completed);
  const done = pred.games.filter((g) => g.completed).reverse();

  const up = document.getElementById("upcoming");
  if (!upcoming.length) {
    up.replaceChildren(el("p", { class: "muted" }, "Regular season complete."));
  } else {
    const head = el(
      "thead",
      {},
      el("tr", {}, el("th", {}, "Date"), el("th", {}, "Opponent"), el("th", { class: "secondary" }, "Rest"), el("th", {}, "Win chance"), el("th", { class: "secondary" }, "Proj. margin")),
    );
    const body = el("tbody");
    for (const g of upcoming) {
      const attrs = g.drivers ? { class: "expandable", tabindex: "0", "aria-expanded": "false" } : {};
      const row = el(
        "tr",
        attrs,
        el("td", {}, g.date),
        el("td", {}, whereCell(g)),
        el("td", { class: "secondary" }, rest(g)),
        el("td", {}, probBar(g.win_prob)),
        el("td", { class: "secondary" }, signed(g.pred_margin)),
      );
      body.append(row);
      if (g.drivers) {
        const text = `${g.drivers.map((d) => `${d.label} ${signed(d.value)}`).join(" · ")} = ${signed(g.pred_margin)} projected margin`;
        const detail = el("tr", { class: "detail" }, el("td", { colspan: "5" }, text));
        detail.hidden = true;
        const toggle = () => {
          detail.hidden = !detail.hidden;
          row.setAttribute("aria-expanded", String(!detail.hidden));
        };
        row.addEventListener("click", toggle);
        row.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            toggle();
          }
        });
        body.append(detail);
      }
    }
    up.replaceChildren(el("table", {}, head, body));
  }

  const mark = (g) => (g.pregame_win_prob == null ? "" : calledIt(g) ? "✓" : "✗");
  document.getElementById("completed").replaceChildren(
    done.length
      ? table(
          ["Date", "Opponent", "Result", "Pre-game win chance", "Model called it"],
          done.map((g) => [g.date, whereCell(g), g.result, pct(g.pregame_win_prob), mark(g)]),
          { secondary: [3] },
        )
      : el("p", { class: "muted" }, "No games played yet."),
  );
}

function setupTabs() {
  const tabs = [...document.querySelectorAll("[role=tab]")];
  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      for (const other of tabs) {
        const on = other === tab;
        other.setAttribute("aria-selected", String(on));
        document.getElementById(other.getAttribute("aria-controls")).hidden = !on;
      }
    });
  }
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
      ["Date", "Record", "Median wins", "80% range", "Playoffs", "#1 seed"],
      data.map((d) => [d.date, d.record, String(d.projected_wins_median), `${d.p10}–${d.p90}`, pct(d.playoff_prob), pct(d.seed1_prob)]),
    ),
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
      height: 240,
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
      ["Team", "Record", "Proj. wins", "Playoffs", "#1 seed"],
      rows.map((r) => [teamLabel(r.team), `${r.wins}-${r.losses}`, r.proj_wins.toFixed(1), pct(r.p_playoffs), pct(r.p_seed1)]),
      { highlighted: rows.map((r) => r.team === pred.team_summary.team) },
    ),
  );
}

const MODEL_NAMES = { homecourt: "Home court only", elo: "Elo only", ridge: "Ridge regression", gbm: "Gradient boosting" };

function renderReportCard(pred, backtest, t) {
  const scored = pred.games.filter((g) => g.completed && g.pregame_win_prob != null);
  const right = scored.filter(calledIt).length;
  const live = pred.metrics.live;
  document.getElementById("live").textContent = scored.length
    ? `This season: called ${right} of ${scored.length} Celtics games${live ? `, log loss ${live.log_loss.toFixed(3)}` : ""}.`
    : "This season's record of picks appears once predicted games have been played.";
  if (!backtest) {
    document.getElementById("backtest-note").textContent = "Backtest results not available yet.";
    return;
  }
  const seasons = backtest.test_seasons;
  const label = (s) => `${s}-${String(s + 1).slice(2)}`;
  document.getElementById("backtest-note").textContent =
    `Walk-forward backtest over the ${label(seasons[0])} through ${label(seasons.at(-1))} seasons. Lower log loss and Brier are better.`;
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
      { highlighted: entries.map(([k]) => k === backtest.main_model), secondary: [2, 4] },
    ),
  );
  // Rendered at a fixed size: the box sits inside a closed <details>, where clientWidth is 0.
  const cal = backtest.calibration;
  document.getElementById("chart-calibration").replaceChildren(
    Plot.plot({
      width: 380,
      height: 380,
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

function renderHowExample(pred) {
  const box = document.getElementById("how-example");
  const g = nextGame(pred);
  if (!g || !g.drivers) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const parts = g.drivers.map((d) => `${d.label.toLowerCase()} ${signed(d.value)}`).join(", ");
  box.textContent =
    `Take the next game (${g.home ? "vs" : "@"} ${teamName(g.opponent)}). The model adds up ${parts}, ` +
    `for a projected margin of ${signed(g.pred_margin)} points. Given how much NBA results vary from game to game, ` +
    `that margin works out to a ${pct(g.win_prob)} chance that the ${teamName(pred.team_summary.team)} win.`;
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
  renderHeader(pred);
  renderNextGame(pred, t);
  renderOutlook(pred, history);
  renderContext(pred);
  renderSchedule(pred);
  renderHistory(history, t);
  renderWins(pred, t);
  renderSeeds(pred, t);
  renderEast(pred);
  renderReportCard(pred, backtest, t);
  renderHowExample(pred);
}

function safeRender() {
  try {
    renderAll();
  } catch (err) {
    console.error(err);
    document.getElementById("updated").textContent = `Some sections failed to render (${err.message}).`;
  }
}

async function main() {
  let pred, history, backtest;
  try {
    [pred, history, backtest] = await Promise.all([
      loadJson("data/predictions.json"),
      loadJson("data/history.json").catch(() => []),
      loadJson("data/backtest.json").catch(() => null),
    ]);
  } catch (err) {
    document.getElementById("updated").textContent = `Could not load predictions (${err.message}).`;
    return;
  }
  state = { pred, history, backtest };
  setupTabs();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", safeRender);
  let timer;
  let lastWidth = window.innerWidth;
  window.addEventListener("resize", () => {
    // Mobile browsers fire resize when the address bar shows or hides; re-render only on width change.
    if (window.innerWidth === lastWidth) return;
    lastWidth = window.innerWidth;
    clearTimeout(timer);
    timer = setTimeout(safeRender, 150);
  });
  safeRender();
}

main();
```

- [ ] **Step 4: Static checks**

Run:
```bash
node --check site/app.js && echo js-ok
grep -c "p_50_wins\|p_top6\|top6_prob\|Top 6\|Top-6" site/app.js site/index.html
```
Expected: `js-ok`, then `0` for both files.

- [ ] **Step 5: Commit**

```bash
git add site/index.html site/styles.css site/app.js
git commit -m "feat(site): next-game-first layout, team context, drivers chart, logos"
```

---

### Task 6: Regenerate real data, verify the page contract, docs

**Files:**
- Modify: `site/data/predictions.json`, `site/data/history.json`, `site/data/pregame_log.json` (regenerated)
- Modify: `README.md`

- [ ] **Step 1: Regenerate with real cached data**

Run: `uv run celtics update --offline`
Expected: a line like `BOS 10-2 | proj 61 wins (80%: 53-67) | playoffs 100%`, with no validation error.

- [ ] **Step 2: Field-contract check against the real JSON**

Write the script below to the session scratchpad at `/private/tmp/claude-501/-Users-lochoang-Desktop-BostonCeltics/94cd812d-a88e-4cfc-a083-3eee631688fd/scratchpad/contract.mjs`, not to the repo, and run `node <that path>` from the repo root.

```js
import { readFileSync } from "node:fs";
const pred = JSON.parse(readFileSync("site/data/predictions.json", "utf8"));
const history = JSON.parse(readFileSync("site/data/history.json", "utf8"));
const fail = (m) => { console.error("FAIL", m); process.exitCode = 1; };
for (const k of ["team_summary", "team_context", "games", "seed_probabilities", "conference_table", "metrics", "season", "generated_at", "model_version"]) if (!(k in pred)) fail(`predictions.${k}`);
if (Object.keys(pred.team_context).length !== 30) fail("team_context size");
for (const [t, c] of Object.entries(pred.team_context)) for (const k of ["wins", "losses", "conf_rank", "games_back", "last10", "streak", "off_rating", "def_rating", "net_rating", "off_rank", "def_rank", "net_rank"]) if (!(k in c)) fail(`team_context.${t}.${k}`);
const next = pred.games.find((g) => !g.completed);
if (next && (!next.drivers || next.drivers.length !== 4)) fail("next game drivers");
if (next && !(next.opponent in pred.team_context)) fail("opponent missing from team_context");
for (const r of pred.conference_table) for (const k of ["team", "wins", "losses", "proj_wins", "p_playoffs", "p_seed1"]) if (!(k in r)) fail(`conference_table.${k}`);
for (const h of history) for (const k of ["date", "record", "projected_wins_median", "p10", "p90", "playoff_prob", "seed1_prob"]) if (!(k in h)) fail(`history.${k}`);
const s = pred.team_summary;
for (const k of ["team", "proj_wins_median", "proj_wins_p10", "proj_wins_p90", "p_playoffs", "p_seed1", "win_distribution"]) if (!(k in s)) fail(`team_summary.${k}`);
console.log(process.exitCode ? "contract FAILED" : "contract ok", next ? `next: ${next.opponent} drivers=${JSON.stringify(next.drivers)}` : "");
```

Expected: `contract ok next: <OPP> drivers=[...]`. Check that the four driver values sum to that game's `pred_margin` within 0.25.

- [ ] **Step 3: HTTP smoke test**

Run:
```bash
(python3 -m http.server 8765 --directory site >/dev/null 2>&1 &) ; sleep 1
for p in / /app.js /styles.css /data/predictions.json /logos/BOS.png /logos/DAL.png; do curl -s -o /dev/null -w "%{http_code} $p\n" "http://localhost:8765$p"; done
pkill -f "http.server 8765"
```
Expected: `200` for every path.

- [ ] **Step 4: Update `README.md`**

In the "How it works" list, change item 5 ("Site") to:

```markdown
5. **Site:** static HTML/JS on GitHub Pages reading JSON written by the pipeline. It leads with the next game, including a breakdown of what drives the prediction (season strength, recent form, home court, rest), then shows the season outlook, team context (standings, last 10, offensive/defensive rank), schedule, and the model's track record. Team logos live in `site/logos/` (fetched once with `scripts/fetch_logos.py`).
```

Then add this line to the Limitations list:

```markdown
- Team logos are trademarks of the NBA and its teams, shown with credit; this project is not affiliated with either.
```

- [ ] **Step 5: Full suite, then commit**

Run: `uv run ruff check . && uv run pytest -q`
Expected: all pass.

```bash
git add site/data README.md
git commit -m "data: regenerate predictions with team context and drivers; README for refreshed site"
```

- [ ] **Step 6: Hand off for the visual check (user)**

Tell the user to run `cd site && python3 -m http.server 8000` and open http://localhost:8000, then check:
- light and dark mode
- a phone-width window (about 400px)
- the next-game card and drivers chart
- the Upcoming/Results tabs, and that clicking an upcoming row shows its breakdown
- logos in the tables and the header
