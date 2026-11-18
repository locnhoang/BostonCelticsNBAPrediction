# League-Wide Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Celtics-only site into a league-wide one, with a Home tab showing the daily slate, a tab for each followed team holding the full team page, an About page, and a light/dark switch. Then rename the repo to NBAPredictor.

**Architecture:** The model and simulation already cover all 30 teams, so only the export and the site change.
- **Pipeline:** a new export module builds a 3-day slate, one file per team, a league file and per-team history. All of it comes from one run, is validated, and is only then written.
- **Pregame log:** it becomes league-wide and stores the home team's win probability. Old Boston-perspective entries are migrated.
- **Site:** it becomes a hash-routed single page with views for Home, team, game and About. The existing renderers take a team as an input.

**Tech Stack:** Python 3.12, uv, pandas, pydantic 2, pytest, ruff. Vanilla JS with Observable Plot 0.6.16 and d3 7.9.0. No build step.

**Spec:** `docs/superpowers/specs/2026-11-18-league-wide-design.md` (mockups in `docs/superpowers/specs/assets/2026-11-18-league-wide/`).

## Global Constraints

**Hosting:**
- The site stays static on GitHub Pages, with no build step and no new front-end libraries.
- Followed teams and the theme are stored in `localStorage` under the keys `followed` (a JSON array of abbreviations) and `theme` (`"light"` | `"dark"`).
- Every storage access is wrapped in try/catch, and the site must work without storage.

**Pick records:**
- Pick rule: the home team if `pregame_home_win_prob >= 0.5`, else the away team. `correct` means the pick won.
- `TRACK_PICKS_FROM = "2026-11-18"` stays and applies to BOS. The new `LEAGUE_TRACK_FROM` applies to the league banner and to the other 29 teams. Its value is the date the final data regeneration runs in Task 8.
- No history is back-filled. Records and wins-over-time history start when they were really logged.

**Data files:**
- Under `site/data/`: `slate.json`, `league.json`, `teams/{ABBR}.json`, `history/{ABBR}.json`, `pregame_log.json`, `dtd_log.json` (unchanged) and `backtest.json` (unchanged).
- `predictions.json` and `history.json` are removed in the same commit that ships the new site. `history.json` becomes `history/BOS.json`.
- `teams/*.json`, `slate.json` and `league.json` are written as compact JSON, with no indent and separators `(",", ":")`. Target: under 1 MB of changed JSON per daily run.
- Drivers are stored only for incomplete slate games and for each team's next game.
- Injury lists in the output contain only Out and Day-To-Day players, Out first, then by expected minutes descending.

**Publishing and the runner:**
- Validate every output, then write. Any validation failure raises `DataValidationError` (exit 3) and writes nothing.
- The Mac runner keeps working: `scripts/run_update.sh` checks `site/data/league.json` instead of `history.json` for "already updated today".
- Not renamed: the Python package and CLI `celtics`, the launchd label and plist, `~/celtics-runner`, and the `TEAM` constant.

**Copy:**
- The nav brand is "NBA PREDICTOR". The page `<title>` is "NBA Predictor".
- Wording rules: no em-dashes in new copy.

## Review Focus

1. **A team's first run after launch:** with no history file and 0 tracked picks, the team tab must render "0-0" and a one-point (or empty) history chart, without errors. Tests: Task 3 (empty history and record) and Task 7 (headless render of `#/team/NYK` on launch-day data).
2. **A slate day with no games** (All-Star break, or the day after the season ends): Home shows "No games" for that day. When there are no games in any of the three days, Home still renders. Tests: Task 2 (empty day) and Task 6 (render check).
3. **Exactly 0.5 home probability:** the pick is the home team, everywhere. Test: Task 2.
4. **The injury feed is unavailable:** slate and team files still publish. `injury_report.available` is false, injury lists are empty, and the healthy probability equals the adjusted one. Test: Task 4.
5. **Old data on disk at first launch:** a `pregame_log.json` with Boston-perspective entries, a `history.json`, and a `predictions.json` but no `league.json`. Migration converts the log, moves history to `history/BOS.json`, deletes both old files, and runs the "not backwards" check against the legacy file. Tests: Task 1 (log migration) and Task 4 (file migration and legacy check).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/celtics/export/league.py` (new) | Pregame-log migration and update, pick/correct, records, slate days, per-team file assembly |
| `src/celtics/export/build.py` (modify) | `game_rows` reads the home-perspective log; `injury_teams` gains `not_next`, injured-only output; `write_json_atomic` gains `compact` |
| `src/celtics/export/schema.py` (modify) | `PregameEntry` (home), `SlateGame`, `Slate`, `TeamFile`, `League`; `Predictions` removed |
| `src/celtics/config.py` (modify) | `LEAGUE_TRACK_FROM` |
| `src/celtics/pipeline.py` (modify) | `run_update` builds and writes all league outputs; migration of old files; league-wide backwards check; roster fetch removed |
| `scripts/run_update.sh` (modify) | "already updated today" watches `league.json` |
| `site/index.html` (rewrite) | Nav bar, view containers (home, team, game, about), footer |
| `site/app.js` (modify) | Router, storage helpers, theme switch, follow; team renderers parameterized |
| `site/home.js` (new, `type="module"` not required; plain script loaded before `app.js`) | Home view: day strip, banner, cards; game detail view |
| `site/styles.css` (modify) | Nav bar, cards, theme tokens with `[data-theme]` |
| `README.md` (modify) | League-wide description, data files, rename |

---

### Task 1: League-wide pregame log (home perspective) with migration

**Files:**
- Create: `src/celtics/export/league.py`
- Modify: `src/celtics/export/schema.py` (`PregameEntry`), `src/celtics/export/build.py` (`game_rows`, `update_pregame_log` removed)
- Test: `tests/test_league.py` (new), `tests/test_export.py` (update the pregame tests to the new shape)

**Interfaces:**
- Produces:
  - `migrate_pregame_log(log: dict, games: pd.DataFrame, team: str = "BOS") -> dict`. An entry with a `win_prob` key is old-format and holds `team`'s probability. It becomes `{"home_win_prob", "pred_margin", "logged_at"}` from the home side, using the game's `home_team` in `games`. An old entry whose game_id is not in `games` is dropped with a warning. New-format entries pass through.
  - `update_league_pregame_log(log: dict, feats: pd.DataFrame, model: MarginModel, season: int, now_iso: str, today: str) -> dict`. For every incomplete current-season game dated `>= today`, it stores the current home prob and margin. Same freeze rule as today: past-dated entries are never touched or back-filled.
  - `PregameEntry`: `home_win_prob: Prob | None`, `pred_margin: float | None`, `logged_at: str`.
  - `game_rows(features, model, season, team, pregame_log)` keeps its signature. For completed games, `pregame_win_prob` = `home_win_prob` if `team` was home, else `1 - home_win_prob` (rounded to 4 decimals).

- [ ] **Step 1: Write the failing tests** in `tests/test_league.py`:

```python
def test_migrate_converts_boston_perspective_to_home():
    games = pd.DataFrame({"game_id": ["g1", "g2"], "home_team": ["BOS", "DEN"], "away_team": ["NYK", "BOS"]})
    old = {
        "g1": {"win_prob": 0.7, "pred_margin": 5.0, "logged_at": "t"},
        "g2": {"win_prob": 0.6, "pred_margin": 3.0, "logged_at": "t"},
        "gx": {"win_prob": 0.5, "pred_margin": 0.0, "logged_at": "t"},  # unknown game: dropped
    }
    out = migrate_pregame_log(old, games)
    assert out["g1"] == {"home_win_prob": 0.7, "pred_margin": 5.0, "logged_at": "t"}
    assert out["g2"] == {"home_win_prob": 0.4, "pred_margin": -3.0, "logged_at": "t"}
    assert "gx" not in out
    assert migrate_pregame_log(out, games) == out  # idempotent on new format


def test_league_log_covers_every_upcoming_game_and_freezes_past(season_feats_model):
    feats, model = season_feats_model  # synthetic league from tests.synth, ridge model
    log = update_league_pregame_log({}, feats, model, 2023, "2023-11-19T14:00:00", "2023-11-19")
    upcoming = feats[(feats["season"] == 2023) & ~feats["completed"] & (feats["date"] >= "2023-11-19")]
    assert set(log) == set(upcoming["game_id"])
    frozen = {"old": {"home_win_prob": 0.3, "pred_margin": -4.0, "logged_at": "x"}}
    assert update_league_pregame_log(frozen, feats, model, 2023, "2023-11-19T14:00:00", "2023-11-19")["old"] == frozen["old"]
```

Update `test_game_rows_perspective_and_pregame_log` in `tests/test_export.py`:
- It now builds the log with `update_league_pregame_log`.
- It sets the first BOS game's `home_win_prob` to 0.42.
- It asserts the shown `pregame_win_prob` is 0.42 if BOS is home, else 0.58.

Replace the three `update_pregame_log` tests with the same cases against `update_league_pregame_log`:
- the freeze rule
- the given-today-not-UTC case
- the schema rejecting `home_win_prob = 1.5`

- [ ] **Step 2: Run** `uv run pytest tests/test_league.py tests/test_export.py -q`. Expected: FAIL (ImportError on `celtics.export.league`).

- [ ] **Step 3: Implement** `migrate_pregame_log`, `update_league_pregame_log` and the schema change, and switch `game_rows` to the home-perspective log. Delete `update_pregame_log` and `live_metrics` from `build.py`, and their imports in `pipeline.py`. `pipeline.py` is rewired in Task 4. Until then, keep it importable by calling the new functions in place of the old ones.

- [ ] **Step 4: Run** `uv run pytest tests/test_league.py tests/test_export.py -q`. Expected: PASS.

- [ ] **Step 5: Commit** `feat(export): league-wide pregame log from the home side, with migration`.

---

### Task 2: Picks, records and the 3-day slate

**Files:**
- Modify: `src/celtics/export/league.py`, `src/celtics/export/schema.py`, `src/celtics/config.py`, `src/celtics/export/build.py` (`injury_teams`)
- Test: `tests/test_league.py`

**Interfaces:**
- Consumes: `PregameEntry` and `update_league_pregame_log` from Task 1; `explain(model, frame)` from `celtics.models.explain`; `healthy_win_prob(features, model, game_id, team, baselines)` and `injury_teams(...)` from `build.py`.
- Produces:
  - `pick(home: str, away: str, p_home: float) -> str`: returns `home` if `p_home >= 0.5`.
  - `pick_record(games: list[dict], since: str, team: str | None = None) -> dict`. It returns `{"since", "correct", "wrong"}` over slate-shaped game dicts that are completed, have `pregame_home_win_prob`, and are dated `>= since`. With `team`, it counts only that team's games.
  - `slate_days(feats, model, pregame_log, team_context, today: date, baselines, injuries: Callable[[str, str, date, frozenset[str]], dict[str, list[dict]]] | None) -> list[dict]`. It returns three `{"date", "games"}` dicts for `today - 1`, `today` and `today + 1`.
    - Each game is a `SlateGame` dict, the shape in the spec's `slate.json` section.
    - `injuries(home, away, game_date, not_next)` returns `{home: [...], away: [...]}`, or the argument is `None` when the feed is unavailable, which gives empty lists.
    - `not_next` is the set of the two teams for which this game is not their next incomplete game.
  - `injury_teams(..., p_dtd, not_next: frozenset[str] = frozenset())`. Teams in `not_next` use `p_out(..., next_game=False)`. Output rows are only Out and Day-To-Day players, Out first, then `expected_minutes` descending. The `rosters` argument stays, but the pipeline passes `dict.fromkeys(teams)`.
  - `SlateGame` and `Slate` schemas. `Slate` = `{generated_at, model_version, days: list[{date, games}], record: {since, yesterday: {correct, wrong}, total: {correct, wrong}}}`.
  - `LEAGUE_TRACK_FROM: str` in `config.py`. Set it to `"2026-11-18"` for now; Task 8 sets the launch date.

- [ ] **Step 1: Write the failing tests:**

```python
def test_pick_ties_go_home():
    assert pick("BOS", "DEN", 0.5) == "BOS"
    assert pick("BOS", "DEN", 0.4999) == "DEN"


def _g(gid, day, home, away, p, hp, ap):
    return {"game_id": gid, "date": day, "home": home, "away": away, "completed": True,
            "pregame_home_win_prob": p, "home_pts": hp, "away_pts": ap}


def test_pick_record_since_and_per_team():
    games = [
        _g("a", "2026-11-17", "BOS", "DEN", 0.6, 100, 90),   # before since: ignored
        _g("b", "2026-11-18", "BOS", "DEN", 0.6, 100, 90),   # correct
        _g("c", "2026-11-18", "NYK", "MIA", 0.7, 90, 100),   # wrong
        _g("d", "2026-11-19", "LAL", "BOS", 0.5, 80, 90),    # pick LAL (home), BOS won: wrong
        {**_g("e", "2026-11-19", "UTA", "SAC", None, 1, 0)},  # no logged pick: ignored
    ]
    assert pick_record(games, "2026-11-18") == {"since": "2026-11-18", "correct": 1, "wrong": 2}
    assert pick_record(games, "2026-11-18", team="BOS") == {"since": "2026-11-18", "correct": 1, "wrong": 1}


def test_slate_has_three_days_and_empty_day(season_feats_model):
    feats, model = season_feats_model
    cur = feats[feats["season"] == 2023]
    dates = sorted({d.date() for d in cur["date"]})
    today = next(d for d in dates if d + timedelta(days=1) not in dates)  # tomorrow has no games
    days = slate_days(feats, model, {}, CONTEXT_30, today, {}, None)
    assert [d["date"] for d in days] == [(today + timedelta(days=k)).isoformat() for k in (-1, 0, 1)]
    assert days[2]["games"] == []
    Slate.model_validate({"generated_at": "x", "model_version": "m", "days": days,
                          "record": {"since": "2026-11-18", "yesterday": {"correct": 0, "wrong": 0},
                                     "total": {"correct": 0, "wrong": 0}}})


def test_slate_game_fields(season_feats_model):
    ...  # assertions listed below


def test_injury_teams_only_injured_and_not_next():
    ...  # assertions listed below
```

`CONTEXT_30` is `team_context(games, 2023)` on the synthetic league.

`test_slate_game_fields` asserts, on the `today` slate from the test above:
- **An incomplete game:**
  - `home_win_prob` is not None.
  - The `drivers` labels equal `["Season strength", "Recent form", "Home court", "Rest & schedule"]` (ridge pinned).
  - `pick == pick(home, away, home_win_prob)`.
  - `injuries == {"home": [], "away": []}`.
  - `home_win_prob_healthy == home_win_prob`.
- **A completed game from the `today - 1` day,** after logging `{"home_win_prob": 0.7, ...}` for it:
  - `home_win_prob is None` and `drivers is None`.
  - `pregame_home_win_prob == 0.7`.
  - `correct == (home won)`.

`test_injury_teams_only_injured_and_not_next` uses hand-built frames for team `"AAA"`: one Out player, one Day-To-Day player and one available player in `current`. It asserts:
- Two rows come back, with Out first.
- With `not_next=frozenset({"AAA"})`, the Day-To-Day row has `p_out == 0.0`.
- Without it, the Day-To-Day row has `p_out == p_dtd`.

- [ ] **Step 2: Run** `uv run pytest tests/test_league.py -q`. Expected: FAIL (names not defined).
- [ ] **Step 3: Implement** the functions and schemas. Each team's record comes from `team_context`, for example `"11-3"`.
- [ ] **Step 4: Run** `uv run pytest tests/test_league.py tests/test_export.py tests/test_injuries.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(export): pick records and the 3-day slate`.

---

### Task 3: Per-team files, league file and per-team history

**Files:**
- Modify: `src/celtics/export/league.py`, `src/celtics/export/schema.py`, `src/celtics/export/build.py` (`write_json_atomic`)
- Test: `tests/test_league.py`

**Interfaces:**
- Consumes: `game_rows`, `team_summary`, `seed_probabilities`, `conference_table`, `history_entry`, `upsert_history` and `healthy_win_prob` from `build.py`; `pick_record` from Task 2.
- Produces:
  - `record_since(team: str) -> str`: `TRACK_PICKS_FROM` for `"BOS"`, else `LEAGUE_TRACK_FROM`.
  - `team_file(team, feats, model, result, inputs, season, pregame_log, injury_report: dict, generated_at, model_version) -> dict`. It has the same fields as today's `predictions.json`, minus `team_context` and `metrics.live`.
    - `metrics.record` is computed with `record_since(team)`, using the team-side rows. The rule is the same as `pick` from the home side.
    - `drivers` is kept only on the first incomplete row and set to `None` on every other row.
    - `win_prob_healthy` is set on that row.
  - `league_file(team_context, generated_at, model_version, completed_games: int) -> dict`.
  - `write_json_atomic(path, obj, compact: bool = False)`. Compact means `separators=(",", ":")` and no indent.
  - Schemas `TeamFile` and `League`. `TeamFile` keeps today's `Predictions` validators, except for the `team_context` length check, which moves to `League` (30 teams).
- Produces for the site:
  - `teams/{ABBR}.json` keys: `generated_at, model_version, season, team_summary, games, seed_probabilities, conference_table, metrics{record}, injury_report`.
  - `league.json` keys: `generated_at, model_version, season, completed_games, team_context`.

- [ ] **Step 1: Write the failing tests:**
  - `test_team_file_matches_old_predictions_shape_for_bos`: build `team_file("BOS", ...)` on the synthetic league. `TeamFile.model_validate` passes. The `games` length is 82. Exactly one row has non-null `drivers`, and it's the first incomplete one. `metrics.record.since == "2026-11-18"`.
  - `test_team_file_other_team_uses_league_start`: `team_file("NYK", ...)["metrics"]["record"]["since"] == LEAGUE_TRACK_FROM`.
  - `test_team_file_empty_record_and_history`: with an empty pregame log, the record is `{"correct": 0, "wrong": 0}`. `upsert_history([], entry)` has length 1.
  - `test_league_file_requires_30_teams`: `League.model_validate` fails with 29 teams.
  - `test_compact_json`: `write_json_atomic(p, {"a": [1, 2]}, compact=True)` writes exactly `{"a":[1,2]}`.
- [ ] **Step 2: Run** `uv run pytest tests/test_league.py tests/test_export.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat(export): per-team files, league file, compact JSON`.

---

### Task 4: Pipeline writes the league outputs

**Files:**
- Modify: `src/celtics/pipeline.py`, `scripts/run_update.sh`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: `run_update(...) -> dict` returns the slate dict. Files written:
  - `slate.json`, `league.json` and `teams/{ABBR}.json` (compact)
  - `history/{ABBR}.json` (indent 1, all 30)
  - `pregame_log.json` and `dtd_log.json` (indent 1)

**Behavior:**
- **Load and migrate:**
  - Load `pregame_log.json` and run `migrate_pregame_log(log, games)` before updating it.
  - If `site/data/history.json` exists and `history/BOS.json` doesn't, seed BOS history from it.
  - After a successful write, delete `predictions.json` and `history.json` if present.
- **Backwards check:** compare the league-wide count of completed current-season games with `league.json["completed_games"]`. If `league.json` is missing but `predictions.json` exists, use the existing BOS check against it. If `season` differs, skip the check.
- **Injuries:**
  - Remove `_fetch_rosters` and the `fetch_team_roster` import. Pass `dict.fromkeys(teams)` as `rosters`.
  - Slate injuries come from a closure over `injury_teams`. The same exception fallback as today applies, giving empty lists for that game.
  - Each team file's `injury_report`: `{available, as_of, p_dtd, unmatched: [names for that team and its opponent], teams: {team: [...], opponent: [...]}}`, for the team's next game. When there is no next game, `teams` is `{}`.
- **Publishing:** validate all outputs (Slate, League, 30 × TeamFile, 30 × HistoryEntry lists, PregameLog, DtdLog) before writing any of them.
- **Runner script:** in `scripts/run_update.sh`, replace `-- site/data/history.json` with `-- site/data/league.json` in the "already updated today" check.

- [ ] **Step 1: Write and update the failing tests:**
  - `test_run_update_writes_valid_outputs` (rewrite):
    - 30 team files and 30 history files exist.
    - `Slate.model_validate` and `League.model_validate` pass.
    - `teams/BOS.json` games has length 82. Its first incomplete row has 4 drivers (ridge pinned).
    - `predictions.json` and `history.json` don't exist.
  - `test_first_launch_migrates_old_files`:
    - Seed the repo with an old-format `pregame_log.json` (one BOS entry, BOS away, `win_prob` 0.6), a `history.json` with one entry, and a `predictions.json` from a prior run.
    - After `run_update`, the log entry has `home_win_prob == 0.4`.
    - `history/BOS.json` starts with the seeded entry.
    - Both old files are gone.
  - `test_backwards_check_league_wide`: a second run with one fewer completed game raises `DataValidationError`, and the files are unchanged.
  - `test_feed_unavailable_publishes_without_adjustment` (update the existing test):
    - Every slate game has empty injury lists.
    - `home_win_prob_healthy == home_win_prob`.
    - `teams/BOS.json` `injury_report.available is False`.
  - `test_invalid_output_writes_nothing`: monkeypatch `team_file` to return an invalid probability for one team. The result is exit 3 or `DataValidationError`, and no file in `site/data` changes.
  - Update the other pipeline tests that read `predictions.json` or `history.json` so they read `teams/BOS.json` or `history/BOS.json`.
- [ ] **Step 2: Run** `uv run pytest tests/test_pipeline.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the pipeline and script changes.
- [ ] **Step 4: Run** `uv run ruff check . && uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(pipeline): publish slate, team, league and history files for all 30 teams`.

---

### Task 5: Site shell: nav bar, router, theme switch, following, About

**Files:**
- Rewrite: `site/index.html`
- Modify: `site/app.js`, `site/styles.css`

**Interfaces:**
- Consumes: `league.json`, `backtest.json`.
- Produces, as globals in `app.js` and used by Tasks 6-7:
  - `store.get(key, fallback)` and `store.set(key, value)`: JSON values in `localStorage`, with try/catch.
  - `followed()` returns a string array. `setFollowed(list)`.
  - `route()` returns `{view: "home" | "team" | "game" | "about", arg: string | null}` parsed from `location.hash`. Unknown routes give `{view: "home"}`. `#/team/XYZ` with an unknown abbreviation gives home.
  - `registerView(name, render: (arg, t) => void)`. Views render into `#view-{name}`. The router shows exactly one.
  - `loadJson(path)` is cached per path for the page's lifetime.
  - `state.league`: the loaded `league.json`.

**Behavior:**
- **Nav bar:** "NBA PREDICTOR", then Home, then one tab per followed team (logo 22px and the short name, from `TEAMS`), then "+ Add team", then "About", then the ☀/☾ button. The current route's tab is marked `aria-current="page"`.
- **Add team:** opens a dialog (`<dialog>`) listing all 30 teams by conference with logos. Choosing one follows the team and navigates to `#/team/{ABBR}`.
- **Theme:**
  - `data-theme` on `<html>` is set from the stored theme. Without a stored theme it isn't set, so the CSS media query applies.
  - The button toggles between light and dark against the effective current theme, stores the choice, and re-renders the current view.
  - Light and dark tokens are defined under `:root`, `@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {...} }` and `:root[data-theme="dark"]`.
  - The nav bar is black in both themes.
- **About view:** the existing backtest table, availability note, calibration chart and "How it works" sections, moved unchanged into `#view-about`. "How it works" is rendered with BOS's team file as its example, loaded on demand.
- **Footer:** same as today, minus "or the Boston Celtics".

- [ ] **Step 1:** Implement the shell, with Home and team views as placeholders that print their arg.
- [ ] **Step 2: Verify** with `node --check site/app.js`. Serve `site/` with `python3 -m http.server`. Render `#/`, `#/about` and `#/team/BOS` in headless Chrome (`--dump-dom`) and confirm that:
  - The nav shows Home and About.
  - The About view contains "Walk-forward backtest".
  - The page contains no "could not be drawn".
  - With `localStorage` seeded to `followed=["BOS"]` (via a `?seed` query param that only the test harness uses, or a headless script), the BOS tab appears.
- [ ] **Step 3: Commit** `feat(site): nav bar, routes, theme switch, following, About page`.

---

### Task 6: Home view and game detail

**Files:**
- Create: `site/home.js` (loaded by `index.html` after `app.js` with `defer`, or before `main()` runs; pick one and keep the order deterministic)
- Modify: `site/styles.css`, `site/index.html` (the view containers)

**Interfaces:**
- Consumes: `slate.json`; `registerView`, `loadJson` and `teamMark` from `app.js`; `renderDrivers(g, t)` and `rosterColumn(abbr, players)`, generalized in Task 7 or here, whichever lands first. Each takes the perspective explicitly. See Task 7's interface.
- Produces: views `home` and `game`.

**Behavior** (matches `assets/2026-11-18-league-wide/home-dark.png`):
- **Day strip:**
  - Yesterday, today and tomorrow, with today selected.
  - Selecting a day shows that day's cards. No route change is needed.
  - A day with no games shows "No games".
- **Banner:** "Model pick record, all games: {y.c}-{y.w} yesterday · {t.c}-{t.w} since {Mon D} ({pct}%)". When the total is 0-0, it shows "first picks graded after tonight's games" in place of the %.
- **Upcoming card:**
  - Away row, then home row, each with logo, name, record and win %. The pick's name and % are bold.
  - A probability bar (away share).
  - "Model pick: {Name}".
  - The whole card is a link to `#/game/{id}`. Team names link to `#/team/{ABBR}`.
- **Final card:**
  - Scores, with the winner bold.
  - "Final · Model picked {Name} ({p}%) ✓/✗". With no logged pick, it shows just "Final".
- **Game view `#/game/{id}`:**
  - Look up the id in the three slate days. If it isn't found, show "This game is not in the current three-day window" and a link home.
  - Incomplete game: the matchup, the home team's win chance and margin, the line "healthy rosters X% → Y% after the injury report" when they differ, and the drivers chart from the home side with the title "Points each factor adds to {Home}'s projected margin". Below it, both injury columns, and the injury source line as on today's Celtics card.
  - Final game: the score and the logged pick with ✓/✗.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify:** `node --check site/home.js site/app.js`. Then run headless renders on current real data (after Task 4, run `uv run celtics update --offline` if supported, else `uv run celtics update`) and confirm:
  - `#/` shows one card per game today and the banner.
  - `#/game/{first today id}` shows the drivers chart and two injury columns.
  - A fake id shows the not-found text.
  - Take screenshots in light and dark (`--force-dark-mode`, and with the stored theme) and look at them.
- [ ] **Step 3: Commit** `feat(site): Home slate and game detail`.

---

### Task 7: Team view (the Celtics page for any team)

**Files:**
- Modify: `site/app.js`, `site/index.html` (team view markup: the current Celtics sections, with "East" wording made dynamic)

**Interfaces:**
- Consumes: `teams/{ABBR}.json`, `history/{ABBR}.json` and `state.league.team_context`.
- Produces:
  - `renderTeam(abbr, t)`, registered as view `team`. It loads both files and builds `pred = {...teamFile, team_context: state.league.team_context}`, then runs the existing section renderers inside `guarded`.
  - `renderDrivers(g, t, sideName)`: the subtitle reads "Points each factor adds to {sideName}'s projected margin".
  - `renderEast` becomes `renderConference(pred)`. Headings use "Eastern" or "Western" from `CONFERENCE_OF` (a new JS map of all 30 teams), and "East seed" becomes "{Conf} seed".
  - The header shows the team logo (64px), "{Full name} 2026-27", "{record} · {rank} in the {Conf}", and a ☆ Follow / ★ Following button that toggles `followed` and re-renders the nav.

**Behavior:**
- **Celtics parity:** every existing section renders for any team. The text that names Boston or Celtics uses the team's name. BOS output must match today's page, apart from the planned moves (the backtest and How it works go to About).
- **Empty history:** a history with fewer than 2 points shows "Tracking started {date}; the chart fills in daily" in place of the line chart.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify:** `node --check site/app.js`. Run headless renders of `#/team/BOS`, `#/team/NYK` and `#/team/LAL` (a West team) and confirm:
  - The header names the team.
  - The conference heading is correct, with Western for LAL.
  - The report card shows `since` Nov 18 for BOS and the launch date for others.
  - There is no "could not be drawn".
  - Screenshot BOS in dark and compare section by section with the live Celtics page. Nothing is missing except the sections moved to About.
- [ ] **Step 3: Commit** `feat(site): team view for any team`.

---

### Task 8: Launch data, docs and checks

**Files:**
- Modify: `src/celtics/config.py` (`LEAGUE_TRACK_FROM`), `README.md`, `site/data/*` (regenerated)

- [ ] **Step 1:** Set `LEAGUE_TRACK_FROM` to today's Eastern date, the date this step runs.
- [ ] **Step 2:** Run `uv run celtics update`. Expected: exit 0. Then `ls site/data/teams | wc -l` → `30`, and `ls site/data/history | wc -l` → `30`.
- [ ] **Step 3: Size check:** run `du -sh site/data` and `git diff --stat -- site/data`. Record both in the commit message. If one day's diff exceeds 1 MB, stop and report to the controller rather than committing.
- [ ] **Step 4: README:** rewrite the intro and "How it works" item 5 for the league-wide site. List the data files. Update the pick-record item: league from launch, BOS from 2026-11-18. Leave the old repo URL in place; Task 9 changes it.
- [ ] **Step 5:** Run `uv run ruff check . && uv run pytest -q && node --check site/app.js site/home.js`. Expected: all pass.
- [ ] **Step 6:** Run headless renders of `#/`, `#/team/BOS`, `#/team/NYK`, one `#/game/` id and `#/about`, each in light and dark. Expected: no "could not be drawn" text.
- [ ] **Step 7: Commit** `data: launch league-wide site` (with the size numbers in the body), then push the branch.

---

### Task 9: Rename to NBAPredictor (after merge; needs the user)

Do this only after the branch is merged to `main` and live under the old URL.

- [ ] **Step 1: The user renames the repo.** Ask them to go to GitHub, Settings, General, and set Repository name to `NBAPredictor`. Wait for confirmation.
- [ ] **Step 2: Update both clones.** In both `/Users/lochoang/Desktop/BostonCeltics` and `~/celtics-runner`, run `git remote set-url origin git@github.com:locnhoang/NBAPredictor.git`. Then `git fetch` in each. Expected: no error.
- [ ] **Step 3: Update the links.** Point the README links and the footer source link in `site/index.html` at `https://github.com/locnhoang/NBAPredictor`. Commit `docs: point links at NBAPredictor` on `main` and push.
- [ ] **Step 4: Check the new address.** Run `curl -s -o /dev/null -w "%{http_code}" https://locnhoang.github.io/NBAPredictor/data/league.json`. Expected: `200`, within a few minutes of the push.
- [ ] **Step 5: Check the runner.** Run `bash ~/celtics-runner/scripts/run_update.sh`. Expected: "already updated today", or a successful push to the renamed repo.
