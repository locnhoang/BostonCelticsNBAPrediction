# Injury & Availability Adjustments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adjust win probabilities for missing players. For the next Celtics game this uses both teams' injury reports and shows both rosters on the site. For the season simulation it uses Out players until their return dates. The change goes live only if the backtest shows a log-loss gain.

**Architecture:**
- **Player logs and point-in-time value:** player game logs are cached per season. A new `celtics.features.availability` module computes each player's point-in-time value, rotation status and absences. From those it produces two features, `home_avail` and `away_avail`.
- **Model feature lists:** models now declare their own feature list (`model.features`), so a new `ridge_avail` model can use the extra columns while `ridge` and the others are unchanged.
- **Live data:** the injury report comes from ESPN's JSON feed and is matched to player IDs. That drives the live availability for upcoming games, the `injury_report` block in `predictions.json`, and a committed Day-To-Day log that learns `p_dtd`.

**Tech Stack:** Python 3.12, uv, pandas 3.1, numpy, scikit-learn, pydantic 2, requests, nba_api (`LeagueGameLog` player mode and `CommonTeamRoster`). Front end: plain HTML/CSS/JS with Observable Plot.

**Spec:** `docs/superpowers/specs/2026-11-13-injury-adjustments-design.md`

## Global Constraints

**Data sources:**
- Player logs: `LeagueGameLog(season=<label>, season_type_all_star="Regular Season", player_or_team_abbreviation="P")`, `"002"` game IDs only.
- Injury feed: `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries`.

**Thresholds** (put them in `celtics/config.py` under these names):

| Name | Value | Meaning |
|---|---|---|
| `ROTATION_WINDOW` | 10 | games looked back to define the rotation |
| `ROTATION_MIN_GAMES` | 5 | minimum games played in that window |
| `ROTATION_MIN_MINUTES` | 15.0 | minimum average minutes in those games |
| `VALUE_DECAY` | 0.95 | per-game decay for player value |
| `VALUE_PRIOR_MINUTES` | 100.0 | prior weight W0, in minutes |
| `EXPECTED_MIN_WINDOW` | 10 | games used for expected minutes |
| `AVAIL_BASELINE_GAMES` | 10 | games used for the availability baseline |
| `DTD_PRIOR` | 0.5 | starting Day-To-Day probability |
| `DTD_MIN_CASES` | 100 | resolved cases needed before learning `p_dtd` |
| `AVAIL_ADOPT_MARGIN` | 0.001 | log-loss gain required to adopt `ridge_avail` |

**Formulas:**
- Game Score: `PTS + 0.4·FGM − 0.7·FGA − 0.4·(FTA − FTM) + 0.7·OREB + 0.3·DREB + STL + 0.7·AST + 0.7·BLK − 0.4·PF − TOV`.
- Value per minute: `(W0·league_avg + Σ λᵏ·GmSc_k) / (W0 + Σ λᵏ·MIN_k)`, using earlier games only.
- Replacement level and `league_avg` come from the previous season. `FIRST_SEASON` uses its own season because it is warm-up only.
- Game value: `max(0, value_per_min − replacement) × expected_minutes`.
- `avail = missing_value(game) − mean(missing_value over the team's previous AVAIL_BASELINE_GAMES games)`, point-in-time.

**Status rules:**
- ESPN `Out` and `Suspension` map to **Out**. Any other status is **Day-To-Day**.
- Live `p_out`: Out = 1, Day-To-Day = `p_dtd` (next game only), anyone else = 0.
- In the season simulation, an Out player counts as missing for games dated before his `return_date`.

**Failure handling:**
- An ESPN failure never blocks a run. The run publishes `injury_report.available = false` and no adjustment.
- An invalid player log is a `DataValidationError`, which exits with code 3.

**Labels and outputs:**
- The driver label is exactly `Injuries & availability`. It appears only when the live model has availability features.
- `predictions.json` adds `injury_report` and `win_prob_healthy` on the next game row. `site/data/dtd_log.json` is committed.

**Testing:** CI never calls ESPN or stats.nba.com. Every test uses fixtures or synthetic data.

## Review Focus

1. **Injured player not in this season's player logs** (two-way signing, or a player out since the preseason). He must appear in `unmatched` with `p_out` recorded and no crash. Test in Task 4.
2. **Player listed under his new team after a trade**, while his logs are under the old team. If the in-team match fails, fall back to a league-wide unique normalized-name match and use that player's value. Test in Task 4.
3. **Out entry whose `returnDate` is missing or before today.** Treat it as Out for the team's next game only, never as "back already". Test in Task 5.
4. **Teams missing from the ESPN feed** (the 2026-11-13 feed listed 28 of 30). Those teams have no injuries, and the code must not raise a KeyError. Test in Task 5.
5. **Opening day or a team with fewer than `AVAIL_BASELINE_GAMES` completed games.** The baseline is the mean over the available games, or 0 if none. It must not be NaN. Test in Task 2.

## File Map

```
src/celtics/config.py                 # + constants above
src/celtics/data/parse.py             # + PLAYER_COLUMNS, playerlog_to_frame
src/celtics/data/fetch.py             # + fetch_season_playerlog, fetch_espn_injuries, fetch_team_roster
src/celtics/data/cache.py             # + player cache: player_season_path, load_all_players, refresh_players
src/celtics/data/validate.py          # + validate_players
src/celtics/features/availability.py  # NEW: game_score, player_states, team_missing, historical_avail, live_avail
src/celtics/features/build.py         # + AVAIL_FEATURES; build_features(games, players=None)
src/celtics/models/margin.py          # features per model; + RidgeAvailModel ("ridge_avail")
src/celtics/models/explain.py         # baseline/groups from model.features; + 5th driver
src/celtics/injuries.py               # NEW: ESPN parse, team map, name matching, dtd log, p_dtd
src/celtics/evaluate/backtest.py      # use model.features; accept players
src/celtics/simulate/inputs.py        # use model.features; matchup rows carry avail = 0
src/celtics/export/build.py, schema.py, pipeline.py, cli.py   # wiring
site/index.html, styles.css, app.js   # injury panel, healthy→adjusted, report row
tests/fixtures/espn_injuries.json     # small committed feed sample
tests/test_players.py, test_availability.py, test_injuries.py (+ extend existing tests)
```

---

### Task 1: Player game logs (parse, fetch, cache, validate)

**Files:**
- Modify: `src/celtics/config.py`, `src/celtics/data/parse.py`, `src/celtics/data/fetch.py`, `src/celtics/data/cache.py`, `src/celtics/data/validate.py`
- Test: `tests/test_players.py`, `tests/factories.py`

**Interfaces:**
- Produces:
  - `PLAYER_COLUMNS = ["game_id","season","date","team","player_id","player_name","min","pts","fgm","fga","ftm","fta","oreb","dreb","ast","stl","blk","tov","pf"]`
  - `playerlog_to_frame(raw: pd.DataFrame, season: int) -> pd.DataFrame` (canonical, sorted by date and game_id)
  - `fetch_season_playerlog(season: int) -> pd.DataFrame` (raw nba_api frame, via `with_retries`)
  - `player_season_path(raw_dir, season) -> Path` (`players_{season}.parquet`)
  - `load_all_players(raw_dir) -> pd.DataFrame`
  - `refresh_players(raw_dir, today, fetch=fetch_season_playerlog, pause=1.5) -> pd.DataFrame`. Past seasons are fetched once. They are guarded by ≥ 900 distinct game_ids and 30 teams, otherwise it raises `FetchBlockedError`. The current season is always refetched, with atomic writes.
  - `validate_players(players: pd.DataFrame, games: pd.DataFrame) -> None`. It raises `DataValidationError` if:
    - a `(game_id, team)` pair is not in games, or
    - there is a duplicate `(game_id, player_id)`, or
    - `min` is below 0.
  - Factory `player_rows(game_id, date, team, player_id, name, minutes, **stats) -> dict` (raw nba_api column names, uppercase).

- [ ] **Step 1: Write the failing tests** in `tests/test_players.py`:
  - `test_playerlog_to_frame_maps_columns_and_filters`:
    - The input is raw rows with `GAME_ID` values "0022600001" and "0062600001" (the second is a Cup final).
    - The output has `PLAYER_COLUMNS`, exactly 1 game ID, and `team`/`player_id`/`min` mapped from `TEAM_ABBREVIATION`/`PLAYER_ID`/`MIN`.
    - `season` is 2026.
  - `test_refresh_players_fetches_past_once_and_current_always`:
    - Mirror `tests/test_cache.py`: monkeypatch `FIRST_SEASON` to 2024 and use a fake fetcher. The fake returns enough rows to pass the guard: build 900 distinct games across 30 teams with 1 player row per team-game.
    - First call fetches `[2024, 2025, 2026]`, second call fetches `[2026]`.
  - `test_refresh_players_rejects_incomplete_past_season`: a fake returning 10 games for 2025 raises `FetchBlockedError` with "incomplete", and no parquet file is written for 2025.
  - `test_validate_players`: passes on a consistent pair; raises `DataValidationError` matching "duplicate" for a duplicated row, "not in games" for an unknown game/team, and "minutes" for `min = -1`.

- [ ] **Step 2: Run** `uv run pytest tests/test_players.py -v`. Expected: FAIL (ImportError).

- [ ] **Step 3: Implement.**
  - Add the config constants from Global Constraints.
  - Implement the interfaces above, following the existing team-log equivalents in the same files: `gamelog_to_games`, `refresh`, `_write_parquet_atomic`, `validate_games`.
  - Cast `GAME_ID` to str and `GAME_DATE` to a normalized Timestamp.

- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.

- [ ] **Step 5: Commit** `feat(data): player game logs: parse, fetch, cache, validate`

---

### Task 2: Player value, absences and historical availability

**Files:**
- Create: `src/celtics/features/availability.py`
- Modify: `src/celtics/features/build.py`
- Test: `tests/test_availability.py`

**Interfaces:**
- Consumes: the canonical games frame and the player frame (Task 1).
- Produces:
  - `game_score(players: pd.DataFrame) -> pd.Series`, using the formula in Global Constraints.
  - `season_baselines(players: pd.DataFrame) -> dict[int, tuple[float, float]]`, mapping season → `(league_avg, replacement)` per Global Constraints.
    - `league_avg` is league Game Score per minute.
    - `replacement` is Game Score per minute among non-rotation player-games.
    - Seasons after the first use the previous season's values; `FIRST_SEASON` uses its own.
  - `player_states(games, players) -> pd.DataFrame` with one row per (game_id, team, player_id) for every player in the team's recent rotation candidates at that game. Columns:
    - `game_id, team, player_id, player_name`
    - `value_per_min`, `expected_minutes`, `game_value`, `rotation` (bool), `played` (bool)
    - Every value is computed only from games strictly before this game.
  - `team_missing(states: pd.DataFrame) -> pd.DataFrame` with columns `game_id, team, missing_value`: the sum of `game_value` over rotation players with `played == False`.
  - `historical_avail(games, players) -> pd.DataFrame` with columns `game_id, home_avail, away_avail`.
    - Applies to completed games. Incomplete games get 0.0.
    - The baseline is the mean `missing_value` over the team's previous ≤ `AVAIL_BASELINE_GAMES` games in the same season, or 0.0 if there are none.
  - `AVAIL_FEATURES = ["home_avail", "away_avail"]` in `features/build.py`.
    - `build_features(games, players: pd.DataFrame | None = None)` adds `AVAIL_FEATURES`. They come from `historical_avail` when `players` is given and are 0.0 otherwise.
    - `FEATURES` is unchanged.
  - `matchup_frame` also emits the `AVAIL_FEATURES` columns, all 0.0.

- [ ] **Step 1: Write the failing tests** in `tests/test_availability.py`. Hand-build data with `make_games` and `player_rows`: one team, BOS, over 12 games, with player "A" playing 30 min in every game except game 12.
  - `test_game_score_formula`: one row with PTS 20, FGM 8, FGA 15, FTM 4, FTA 5, OREB 2, DREB 6, AST 5, STL 1, BLK 1, TOV 3, PF 2 gives `20+3.2−10.5−0.4+1.4+1.8+1+3.5+0.7−0.8−3 = 16.9` (approx).
  - `test_rotation_requires_min_games_and_minutes`:
    - A player with 4 games at 30 min in the last 10 is not rotation.
    - A player with 6 games at 14 min is not rotation.
    - A player with 6 games at 20 min is rotation.
  - `test_absent_rotation_player_counts_as_missing`: at game 12, A is `rotation=True, played=False`, and `team_missing` for BOS at game 12 equals A's `game_value`, which is greater than 0.
  - `test_values_are_point_in_time`: changing A's PTS in game 12 doesn't change any `player_states` row for games ≤ 12.
  - `test_low_minutes_shrink_to_league_avg`: a player with one 2-minute game with Game Score 10 has `value_per_min` within 0.1 of `league_avg`.
  - `test_avail_sitting_negative_returning_positive`:
    - At game 12 (A out), BOS `avail` is below 0.
    - Add a game 13 where A returns after being out in games 11–12. Game 13's `avail` is above 0.
  - `test_baseline_with_no_history_is_zero` (Review Focus 5): BOS's first game of a season gives `home_avail` equal to 0.0 when nobody was absent, never NaN.
  - `test_build_features_without_players_has_zero_avail`: `build_features(make_league(...))` has `AVAIL_FEATURES` columns, all 0.0, and `FEATURES` columns unchanged.

- [ ] **Step 2: Run** `uv run pytest tests/test_availability.py -v`. Expected: FAIL (ImportError).

- [ ] **Step 3: Implement.**
  - **Player value:** process player rows in date order per player. Keep the decayed sums for the value-per-minute formula, and compute each state from the sums *before* adding the current game.
  - **Rotation:** for each team-game, take the team's previous `ROTATION_WINDOW` games. A player is in the rotation if he played ≥ `ROTATION_MIN_GAMES` of them with mean minutes ≥ `ROTATION_MIN_MINUTES`.
  - Iterating in Python over about 300k player rows is acceptable. Keep it under about 60s for the full history; measure it in Task 8.

- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.

- [ ] **Step 5: Commit** `feat(features): point-in-time player value, absences and availability feature`

---

### Task 3: Per-model feature lists, `ridge_avail`, availability driver

**Files:**
- Modify: `src/celtics/models/margin.py`, `src/celtics/models/explain.py`, `src/celtics/evaluate/backtest.py`, `src/celtics/simulate/inputs.py`, `src/celtics/export/build.py`, `src/celtics/pipeline.py`
- Test: `tests/test_models.py`, `tests/test_explain.py`, `tests/test_backtest.py`

**Interfaces:**
- Consumes: `AVAIL_FEATURES` and `build_features(games, players)` from Task 2.
- Produces:
  - **`MarginModel`:**
    - New class attribute `features: list[str] = FEATURES`.
    - `fit`, `predict_margin` and `win_prob` select `X[self.features]`.
    - Every caller now passes the full features frame, not `X[FEATURES]`.
  - **`RidgeAvailModel`:** `name = "ridge_avail"`, `is_linear = True`, `features = FEATURES + AVAIL_FEATURES`. Same pipeline as `RidgeModel`.
  - **`MODELS`:** gains `"ridge_avail"`.
  - **`explain`:**
    - The baseline includes `home_avail` / `away_avail` = 0.0.
    - New group `"Injuries & availability": AVAIL_FEATURES`.
    - `DRIVER_ORDER` becomes `["Season strength", "Recent form", "Home court", "Rest & schedule", "Injuries & availability"]`.
    - Returned columns are `DRIVER_ORDER` filtered to groups whose columns are all in `model.features`, so `ridge` still returns 4 columns.
  - **`walk_forward(features, model_name, test_seasons)`** and **`season_coverage(games, model_name, ..., players=None)`** pass full frames. `season_coverage` forwards `players` to `build_features`.
  - **`run_backtest`** loads players (`load_all_players` when offline, else `refresh_players`), validates them, and builds features with players. Every model in `MODELS`, including `ridge_avail`, is scored.

- [ ] **Step 1: Write the failing tests:**
  - `tests/test_models.py`:
    - `test_ridge_avail_uses_avail_columns`: on synthetic X with avail columns where `y` depends on `home_avail`, `ridge_avail` predictions change when `home_avail` changes and `ridge` predictions don't.
    - `"ridge_avail" in MODELS`.
  - `tests/test_explain.py`:
    - `test_ridge_avail_has_five_drivers_summing_to_prediction`: 5 columns in `DRIVER_ORDER` order, and rows sum to the prediction within 1e-9.
    - `test_ridge_still_has_four_drivers`.
  - `tests/test_backtest.py`:
    - `test_walk_forward_ridge_avail_runs_on_synthetic_league`: with `make_league` features plus zero avail columns, it returns probabilities in (0, 1) for the test season.

- [ ] **Step 2: Run** the three files. Expected: FAIL.

- [ ] **Step 3: Implement.**
  - Replace every `X[FEATURES]` and `frame[FEATURES]` passed to a model with the full frame. `grep -rn "\[FEATURES\]" src/` must return only `margin.py` internals and `explain.py`.
  - In `game_rows`, the drivers list is built from the columns `explain` returns.

- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass. Existing outputs are unchanged while `MODEL_NAME == "ridge"`.

- [ ] **Step 5: Commit** `feat(models): per-model feature lists, ridge_avail, injuries driver`

---

### Task 4: ESPN injury feed, team map, player matching

**Files:**
- Create: `src/celtics/injuries.py`, `tests/fixtures/espn_injuries.json`, `tests/test_injuries.py`
- Modify: `src/celtics/data/fetch.py`

**Interfaces:**
- Produces:
  - `ESPN_INJURIES_URL`
  - `fetch_espn_injuries() -> dict` (via `with_retries`, timeout 30)
  - `ESPN_TEAM: dict[str, str]`: all 30 ESPN team display names → abbreviations, e.g. `"LA Clippers": "LAC"`, `"Los Angeles Lakers": "LAL"`.
  - `normalize_name(name: str) -> str`: NFKD, strip accents, lowercase, remove punctuation and the suffixes `jr sr ii iii iv`, collapse spaces.
  - `parse_injuries(raw: dict) -> pd.DataFrame` with columns `team, name, status, comment, return_date, reported_at`.
    - `status` is one of `"Out"` / `"Day-To-Day"`, per the Global Constraints mapping.
    - `return_date` is `datetime.date | None`.
  - `match_players(injuries: pd.DataFrame, players: pd.DataFrame, season: int) -> tuple[pd.DataFrame, list[str]]`.
    - Adds `player_id` to injuries, and returns `(matched, unmatched_names)`.
    - Order of attempts:
      1. Exact normalized name within the team.
      2. Last name plus first initial within the team, if unique.
      3. League-wide unique normalized name, for traded players.
    - Anything else goes into `unmatched` as `"Name (TEAM)"`.

- [ ] **Step 1: Save the fixture.** Build `tests/fixtures/espn_injuries.json` by trimming the real feed, keeping its JSON shape (`{"injuries":[{"displayName":…,"injuries":[{"status":…,"shortComment":…,"date":…,"athlete":{"displayName":…},"details":{"returnDate":…}}]}]}`), to these 6 entries:
  - BOS "Jaylen Brown" Day-To-Day
  - BOS "Jayson Tatum" Out with returnDate "2026-12-01"
  - DAL "Dereck Lively II" Out
  - LAC "Kawhi Leonard" Day-To-Day
  - one "Suspension" entry
  - one entry for a name that won't match: "Unknown Player"
  The script that does the trimming lives in the scratchpad, not the repo.

- [ ] **Step 2: Write the failing tests** in `tests/test_injuries.py`:
  - `test_espn_team_map_covers_all_teams`: `set(ESPN_TEAM.values()) == set(TEAMS)`.
  - `test_normalize_name`:
    - `normalize_name("Luka Dončić") == "luka doncic"`
    - `normalize_name("Dereck Lively II") == "dereck lively"`
    - `normalize_name("Gary Payton Jr.") == "gary payton"`
  - `test_parse_injuries_maps_status_and_dates`: Suspension → Out, Day-To-Day stays, Tatum's `return_date == date(2026, 12, 1)`, LAC → "LAC".
  - `test_match_players_in_team_and_unmatched` (Review Focus 1): with a player frame containing BOS Brown and Tatum and DAL "Dereck Lively", all three match, and `"Unknown Player (…)"` is in `unmatched`.
  - `test_match_players_league_wide_fallback_for_traded_player` (Review Focus 2): "Kawhi Leonard" listed under LAC but whose logs are under another team matches league-wide.

- [ ] **Step 3: Run** `uv run pytest tests/test_injuries.py -v`. Expected: FAIL.

- [ ] **Step 4: Implement** the interfaces above.

- [ ] **Step 5: Run** `uv run pytest -q`. Expected: all pass.

- [ ] **Step 6: Commit** `feat(injuries): ESPN injury feed parsing, team map, player matching`

---

### Task 5: Live availability, Day-To-Day learning, rosters

**Files:**
- Modify: `src/celtics/features/availability.py`, `src/celtics/injuries.py`, `src/celtics/data/fetch.py`
- Test: `tests/test_availability.py`, `tests/test_injuries.py`

**Interfaces:**
- Consumes:
  - `player_states`, `team_missing` (Task 2)
  - `match_players` output (Task 4)
- Produces:
  - `current_player_values(games, players, season) -> pd.DataFrame` with columns `team, player_id, player_name, value_per_min, expected_minutes, game_value, rotation`.
    - Each player's state is as of after his latest completed game, under his latest team.
  - `live_avail(games, players, injuries_matched, season, today: date, p_dtd: float) -> pd.DataFrame` with columns `game_id, home_avail, away_avail`. It covers current-season incomplete games.
    - **Per team, next incomplete game:** missing = Σ `game_value × p_out`. Out gives 1, Day-To-Day gives `p_dtd`.
    - **Later games:** missing = Σ `game_value` over Out players whose `return_date` is after the game date.
    - **Out with a missing or past `return_date`:** counts for the next game only.
    - **Baseline:** the mean `missing_value` over the team's last ≤ `AVAIL_BASELINE_GAMES` completed games (0.0 if none). `avail` = missing − baseline.
    - **Teams with no injury rows:** missing = 0.
  - `update_dtd_log(log: list[dict], injuries_matched, games, players, today) -> list[dict]`.
    - Appends today's Day-To-Day players as `{date, team, player_id, name, played: None}`, with no duplicate (date, player_id).
    - Resolves `played` for earlier cases once the team's first completed game after `date` exists: True if the player has a row in that game.
  - `learn_p_dtd(log) -> float`:
    - Returns `DTD_PRIOR` until there are ≥ `DTD_MIN_CASES` resolved cases.
    - After that, returns the share that did not play.
  - `fetch_team_roster(team: str, season: int) -> pd.DataFrame`: `CommonTeamRoster` with columns `player_id, player_name`, via `with_retries`. It needs a static abbreviation → NBA team_id map. Use `nba_api.stats.static.teams`.

- [ ] **Step 1: Write the failing tests:**
  - `test_live_avail_next_game_uses_out_and_dtd`:
    - BOS's next game, with Tatum Out (game value G1) and Brown Day-To-Day (G2), at `p_dtd = 0.5`.
    - home or away missing = G1 + 0.5·G2. `avail` = that minus the baseline.
  - `test_live_avail_return_date_limits_later_games`: Tatum Out until 2026-12-01 counts in BOS games before that date and not in games on or after it. Brown, Day-To-Day, counts only in the next game.
  - `test_out_without_return_date_counts_next_game_only` (Review Focus 3).
  - `test_team_missing_from_feed_has_no_injuries` (Review Focus 4): a team not present in the injuries frame gets missing = 0 with no KeyError.
  - `test_update_dtd_log_appends_and_resolves`:
    - Day 1 logs Brown.
    - After a completed BOS game in which Brown has no row, his case resolves to `played == False`.
    - Running again on the same day doesn't duplicate.
  - `test_learn_p_dtd_threshold`:
    - 99 resolved cases gives `DTD_PRIOR`.
    - 100 cases with 30 non-plays gives 0.30.

- [ ] **Step 2: Run** the tests. Expected: FAIL.

- [ ] **Step 3: Implement** the interfaces above.

- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.

- [ ] **Step 5: Commit** `feat: live availability from injury report, Day-To-Day learning, rosters`

---

### Task 6: Pipeline, export and schema wiring

**Files:**
- Modify: `src/celtics/pipeline.py`, `src/celtics/export/build.py`, `src/celtics/export/schema.py`, `src/celtics/cli.py`
- Test: `tests/test_pipeline.py`, `tests/test_export.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces:
  - **Data loading in `run_update`:**
    - Loads or refreshes players and runs `validate_players`.
    - Builds features with players, then overwrites `AVAIL_FEATURES` for current-season incomplete games with `live_avail` output.
    - If the model's features don't include the avail columns, there is no effect on predictions.
  - **ESPN handling:** fetch the feed; on any exception, log a warning and continue with no injuries and `injury_report.available = false`.
    - On success, save the raw feed to `data/injuries/{today}.json` (cached, not committed).
  - **`celtics fetch`:** also runs `refresh_players`.
  - **`p_dtd`:** loaded from `site/data/dtd_log.json` via `learn_p_dtd`. The log is updated and written.
  - **`predictions.json`:**
    - The next game row gains `win_prob_healthy`: the prediction with that row's avail features set to 0.
    - New top-level `injury_report` block: `{available: bool, as_of: str | None, p_dtd: float, unmatched: list[str], teams: {ABBR: [player…]}}`.
    - `teams` covers the next game's two teams. Each player is `{player_id, name, status, comment, return_date, p_out, expected_minutes, game_value, rotation}`, where `status` is one of `"Available"`, `"Out"` or `"Day-To-Day"`.
    - Players come from the roster. If the roster fetch fails, use `current_player_values` for that team plus the injured players. Rotation players come first, sorted by `expected_minutes` descending.
  - **Schema:** `InjuryPlayer`, `InjuryReport`, `GameRow.win_prob_healthy: Prob | None`, `Predictions.injury_report: InjuryReport`. Also a `DtdLog` adapter, validated before writing.
  - **`run_backtest`:** adds `"availability_note": "Backtest uses actual absences from box scores; live predictions use the injury report, so live accuracy will be somewhat lower."` to `backtest.json`, and to the `BacktestReport` schema.

- [ ] **Step 1: Write the failing tests** in `tests/test_pipeline.py`, extending the `repo` fixture:
  - Write synthetic player logs for every synthetic game: 8 players per team, about 30 min each, ids `team_index*100+k`.
  - Monkeypatch `celtics.pipeline.fetch_espn_injuries` to return a fixture-shaped dict naming one rotation player of BOS's next opponent as Out.
  - Monkeypatch `fetch_team_roster` to return each team's 8 synthetic players.
  - Tests:
    - `test_run_update_publishes_injury_report`: `injury_report.available` is true, `teams` has exactly the two next-game teams, and the Out player has `p_out == 1`.
    - `test_run_update_survives_espn_failure`: patch the fetch to raise. The run still writes `predictions.json`, `available` is false, and next game `win_prob == win_prob_healthy`.
    - `test_invalid_player_log_exits_3`: a duplicate player row makes `main([... "update", "--offline"])` return 3.
    - `test_dtd_log_written_and_validated`: `site/data/dtd_log.json` exists and validates.

- [ ] **Step 2: Run** the tests. Expected: FAIL.

- [ ] **Step 3: Implement** the interfaces above. Keep every existing guard: `validate_games`, `_check_not_backwards`, the pregame log rules, and atomic writes.

- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.

- [ ] **Step 5: Commit** `feat(pipeline): injury report, live availability and dtd log in daily update`

---

### Task 7: Website injury panel

**Files:**
- Modify: `site/index.html`, `site/styles.css`, `site/app.js`

**Interfaces:**
- Consumes: `injury_report`, `games[].win_prob_healthy`, the fifth driver, and `backtest.availability_note` (Task 6).

- [ ] **Step 1: Implement in the Next game section:**
  - **Injury report subsection** under the matchup:
    - Two columns, one per team, each headed by the logo and nickname, stacking below 560px.
    - **Player rows:** the name and a status chip.
      - **Out:** comment, plus "back ~{return_date}" when present.
      - **Day-To-Day:** "~{round(p_out·100)}% out".
      - **Available:** a muted chip.
    - Rotation players are shown first. The rest sit behind a "Show full roster" button.
  - **Win-chance line:** "{healthy}% healthy roster → {win_prob}% after the injury report", shown only when `available` is true and the two values differ.
  - **Source line:** "Injury report as of {local time} (ESPN)". If `available` is false: "Injury report unavailable — prediction uses healthy rosters."
  - **Unmatched names:** if `unmatched` is non-empty, a muted line reading "Not matched to player stats: …".
  - **Driver bar:** the fifth bar uses the existing green/gray rule. No change is needed if `renderDrivers` iterates `drivers`; confirm it does.
  - **Report card:** show `availability_note` under the backtest table when present.
  - **Status chips:** use text plus color, never color alone. Out uses `--chip-out` and Day-To-Day uses `--chip-dtd`, both defined for light and dark mode and distinct from the accent green. Label the chips "Out", "Day-To-Day" and "Available".

- [ ] **Step 2: Verify.**
  - `node --check site/app.js` passes.
  - Generate data with the Task 6 pipeline offline, using `--offline` with the live ESPN feed. If ESPN is unreachable, the panel must show the "unavailable" state.
  - Serve with `python3 -m http.server`, check that `/` returns 200, then `git checkout -- site/data`.

- [ ] **Step 3: Commit** `feat(site): injury report panel with both rosters and adjusted win chance`

---

### Task 8: Real data, backtest, adoption decision, docs

**Files:**
- Modify: `src/celtics/config.py` (`MODEL_NAME`, only if adopted), `README.md`, `docs/superpowers/specs/2026-11-13-injury-adjustments-design.md` (results), `site/data/*.json`

- [ ] **Step 1: Fetch the player logs.**
  - Run `uv run celtics fetch`, or the equivalent that runs `refresh_players`. This takes 13 stats.nba.com calls, about 2–5 minutes.
  - Expected: player parquet files for 2014–2026, each past season with at least 900 game IDs.

- [ ] **Step 2: Run the backtest.**
  - Run `uv run celtics backtest --offline`. Record the timing and the full output.
  - Expected: a `ridge_avail` row is present.

- [ ] **Step 3: Apply the adoption rule.**
  - If `ridge_avail.log_loss ≤ ridge.log_loss − AVAIL_ADOPT_MARGIN`, set `MODEL_NAME = "ridge_avail"` and rerun the backtest.
  - Otherwise keep `"ridge"`.
  - Record both numbers and the decision in the spec under `## Results`.

- [ ] **Step 4: Run the update.**
  - Run `uv run celtics update --force-train`. This is the live ESPN fetch.
  - Expected: the output line prints and `injury_report.available` is true. Spot-check that BOS and the next opponent rosters list sensible names and statuses.

- [ ] **Step 5: Update the README.**
  - In How it works, add an item on injuries:
    - absences derived from box scores for the backtest
    - the ESPN injury feed for live predictions
    - Day-To-Day probability learned over time
    - adopted or not, with the numbers
  - Add a limitation: the ESPN feed has no Questionable/Probable levels, and the backtest uses actual absences.

- [ ] **Step 6: Run** `uv run ruff check . && uv run pytest -q`. Expected: all pass.

- [ ] **Step 7: Commit** `data: player logs backtest, injury-adjusted predictions, docs`
