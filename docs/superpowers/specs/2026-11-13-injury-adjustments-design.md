# Injury & Availability Adjustments — Design

**Date:** 2026-11-13
**Status:** Approved design, pending implementation plan
**Builds on:**
- `2026-11-12-celtics-prediction-design.md` (model, simulation, backtest)
- `2026-11-12-frontend-refresh-design.md` (site, drivers)

## Goal

Adjust win probabilities for which players are missing.

- **Next Celtics game:** use both teams' current injury report, and show both rosters with each player's status.
- **Season simulation:** remove players listed Out from every remaining game until their expected return date, for all 30 teams.
- **Backtest:** validate the adjustment on past seasons. It goes live only if it improves log loss.

## Scope

**In scope:**
- Player game logs
- Box-score player value
- Historical absences derived from box scores
- ESPN injury feed
- Roster display
- A new availability feature in the ridge model
- A fifth driver
- Return-date handling in the season simulation
- Day-To-Day probability learned from logged snapshots

**Out of scope:**
- Play-by-play / adjusted plus-minus models
- Official NBA injury-report PDFs: `ak-static.cms.nba.com` returns 403 from this machine, so Questionable / Probable / Doubtful levels are unavailable
- Trades and roster transactions beyond what current rosters show

## Data

### Player game logs (new)

- **Source:** `nba_api` `LeagueGameLog(season=…, season_type_all_star="Regular Season", player_or_team_abbreviation="P")`. That is one call per season, 2014-15 onward, served by stats.nba.com, which is reachable.
- **Canonical player-game frame:** `game_id, season, date, team, player_id, player_name, min, pts, fgm, fga, ftm, fta, oreb, dreb, ast, stl, blk, tov, pf`. Only `"002"` game IDs are kept.
- **Cache:** `data/raw/players_{season}.parquet`.
  - Past seasons are fetched once, behind the same incomplete-season guard as the team data (≥ 900 games' worth of rows across 30 teams).
  - The current season is refetched daily.
- **Validation:**
  - Every `(game_id, team)` in the player logs exists in the games frame.
  - No duplicate `(game_id, player_id)`.
  - Minutes ≥ 0.
  - A failure raises `DataValidationError` (exit 3).

### Absences in past games (derived)

- A player is a **rotation player** for a team-game if, over the team's previous 10 games, he played in ≥ 5 and averaged ≥ 15 minutes in the games he played.
- A rotation player with no row for that game is **absent**. This covers injury, rest and suspension.
- Players traded mid-season count for their new team from their first game there.

### Live injury report (ESPN)

- **Source:** `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries` (JSON, reachable).
- **Fields used per entry:** athlete display name, team display name, `status` (`Out` / `Day-To-Day` / `Suspension`), `shortComment`, `details.returnDate`, `date`.
- **Status mapping:**
  - `Out` and `Suspension` → **Out**
  - `Day-To-Day` → **Day-To-Day**
  - anything else → **Day-To-Day**
- **Team mapping:** a static table from ESPN team display names to abbreviations, e.g. "LA Clippers" → LAC.
- **Player matching**, done within each team against current-season player-log names:
  1. Normalize names: Unicode NFKD, strip accents, lowercase, remove punctuation and the suffixes jr/sr/ii/iii/iv.
  2. Match on the exact normalized name first.
  3. Otherwise match on last name + first initial, if that is unique.
  4. Unmatched entries are kept in an `unmatched` list and shown on the site. They are never silently dropped.
- **Snapshots:**
  - Each run saves the raw feed to `data/injuries/{date}.json`, which is cached and not committed.
  - Each run appends Day-To-Day cases to `site/data/dtd_log.json`, which is committed. Each case is `{date, team, player_id, name, played: null}`.
  - `played` is filled in once that team's next game has a result: true if the player has a box-score row.
- **Failure:** the run still publishes, with no injury adjustment. `predictions.json` sets `injury_report.available = false`, and the site shows "Injury report unavailable".

### Rosters (display)

`nba_api` `CommonTeamRoster(team_id, season)` for the two teams in the next game. Every roster player with no injury entry is shown as "Available". If the roster fetch fails, the panel lists only players who appear in the injury feed and the player logs.

## Model

### Player value (point-in-time)

- **Game Score per game:**
  `PTS + 0.4·FGM − 0.7·FGA − 0.4·(FTA − FTM) + 0.7·OREB + 0.3·DREB + STL + 0.7·AST + 0.7·BLK − 0.4·PF − TOV`
- **Value per minute**, as of a date and using only earlier games:
  - `(W0·league_avg + Σ λᵏ·GmSc_k) / (W0 + Σ λᵏ·MIN_k)`
  - λ = 0.95 per game played, and W0 = 100 minutes of prior weight.
  - `league_avg` is the previous season's league-wide Game Score per minute.
  - Rookies and players with no history start at `league_avg`.
- **Replacement level:** the previous season's average Game Score per minute among non-rotation players.
- **Expected minutes:** the average minutes over the player's last 10 games played. If there are none, use his previous-season average; if there is none, 0.
- **Game value:** `max(0, value_per_min − replacement) × expected_minutes`.

### Missing value and the availability feature

- **Missing value** for a team in a game: `Σ game_value × p_out` over the team's rotation players.
  - **Backtest:** `p_out` is 1 if the player was absent, else 0.
  - **Live, next game:** Out → 1, Day-To-Day → `p_dtd`, everyone else → 0.
  - **Live, later games in the season simulation:** Out players count until their return date, and Day-To-Day players don't count.
- **Availability feature:** `avail = missing_value(game) − mean(missing_value over the team's previous 10 games)`. Both terms are point-in-time.
- **New model features:** `home_avail` and `away_avail`, added to `FEATURES` under the registry name `ridge_avail`. The current ridge stays registered as `ridge`.

### Day-To-Day probability

- `p_dtd` starts at 0.5.
- Once `dtd_log.json` has ≥ 100 resolved cases, `p_dtd` becomes the observed fraction that did **not** play.
- The value is recomputed on every run and stored in `predictions.json`.

### Adoption rule

- The backtest scores `ridge` and `ridge_avail` with the same walk-forward setup.
- `MODEL_NAME` becomes `ridge_avail` only if its log loss is lower by ≥ 0.001.
- Either way, both rows appear in `backtest.json` and on the site.
- The report card notes that the backtest uses actual absences, so live accuracy will be somewhat lower.

### Drivers

When the live model has availability features, the breakdown gains a fifth driver, **"Injuries & availability"**: the combined effect of `home_avail` and `away_avail` against a baseline of 0. The parts still sum exactly to the projected margin.

### Season simulation

For each remaining game of every team, `avail` uses that team's Out players whose return date is after the game date. The recent-games baseline is frozen at today's value. Day-To-Day players affect only each team's next game.

## Outputs (`predictions.json` additions)

`injury_report`:
- `available` (bool) and `as_of` (ISO timestamp of the feed)
- `p_dtd`
- `unmatched` (a list of names)
- `teams`, covering the two teams in the next game:
  - For each team, a list of players with `{player_id, name, status, comment, return_date, p_out, expected_minutes, game_value, rotation}`.
  - `status` is one of Available, Out or Day-To-Day.
  - Rotation players come first, sorted by expected minutes.

On the next game row:
- `win_prob_healthy`: the win probability with the availability features set to 0.
- The existing `win_prob` includes the injury adjustment.

All new fields are validated by the schema.

## Website

**Next-game card:**
- An **Injury report** panel with both rosters side by side and their logos.
  - Each player shows his status: Out with comment and return date, Day-To-Day with "~N% out", or Available.
  - Rotation players are shown first, and the rest sit behind "Show full roster".
- A "**X% healthy roster → Y% after the injury report**" line.
- "Injury report as of {time} (ESPN)", or "Injury report unavailable" when the feed failed.
- The unmatched list, if it isn't empty.

**Other sections:**
- The drivers chart shows the fifth bar when it is present.
- The report card shows the `ridge_avail` backtest row and the caveat.

## Testing

- **Unit tests on hand-built data:**
  - rotation and absence detection
  - point-in-time player value, including shrinkage
  - replacement level
  - availability for a player sitting (negative) and a player returning (positive)
  - name normalization and matching, including the unmatched list
  - p_out weighting
  - p_dtd learning threshold
  - return-date handling in the simulation inputs
- **Parser test** on a small committed ESPN fixture, `tests/fixtures/espn_injuries.json`. CI never calls ESPN.
- **Pipeline tests:**
  - When the ESPN fetch fails, the run still publishes with `injury_report.available = false`.
  - When the player logs are invalid, the run exits with code 3.
- **Backtest:** both models are reported, and the result is recorded in this spec. If `ridge_avail` doesn't clear the adoption rule, the live model stays `ridge` and the site shows the injury report as information only.
