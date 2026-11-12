# Front-End Refresh — Design

**Date:** 2026-11-12
**Status:** Approved design, pending implementation plan
**Builds on:** `2026-11-12-celtics-prediction-design.md`

## Goal

Clean up and reorganize the website, lead with the next game, add team context and model transparency, and show team logos.

## Scope

**In scope:**
- New export fields: team context for every team, per-game prediction breakdown.
- Reorganized page, with the next game first.
- Team logos stored in the repo.
- Visual polish.

**Out of scope:**
- Injuries.
- Any change to the model, the features, the simulation or the backtest.

## Data additions (`predictions.json`)

All new fields are validated by the pydantic schema before writing, like the existing fields.

### `team_context`: dict keyed by team abbreviation, all 30 teams

Computed from the current season's completed games only.

| Field | Definition |
|---|---|
| `wins`, `losses` | Current record |
| `conf_rank` | Rank within its conference, 1–15. Ordered by win % then wins; remaining ties broken by abbreviation (no tiebreakers). |
| `games_back` | `((leader_w − w) + (l − leader_l)) / 2` versus the conference's rank-1 team; 0.0 for the leader |
| `last10` | Record over the team's last 10 completed games, e.g. `"7-3"` (fewer than 10 if fewer played) |
| `streak` | `"W3"` / `"L2"` from consecutive results ending with the latest game; `""` if no games |
| `off_rating` | 100 × points scored / possessions, summed over the season |
| `def_rating` | 100 × points allowed / possessions, summed over the season |
| `net_rating` | `off_rating − def_rating` |
| `off_rank`, `def_rank`, `net_rank` | League rank 1–30, where 1 is best (lowest `def_rating` ranks 1) |

Teams with no completed games get 0–0, ratings 0.0, last10 `"0-0"` and streak `""`. They rank after every team that has played (in the conference and in each league rank), ordered by abbreviation. A 0–0 team has a win % of 0 for conference ordering.

### Prediction breakdown on each upcoming game row: `drivers`

A list of `{label, value}` from the tracked team's perspective, summing exactly (within rounding) to `pred_margin`:

| Label | Contribution |
|---|---|
| `Season strength` | Effect of `elo_diff` versus 0 |
| `Recent form` | Effect of `net_diff` versus 0 |
| `Home court` | Predicted margin for two equal teams on normal rest (2 days, no back-to-back, no 3-in-4). Negated when the team is away. |
| `Rest & schedule` | Effect of the actual rest / back-to-back / 3-in-4 values versus that normal-rest baseline |

- Computed by evaluating the model on baseline rows with feature groups swapped in. This is exact for linear models.
- `MarginModel` gains `is_linear: bool`: True for ridge, elo and homecourt; False for gbm.
- When the live model is not linear, `drivers` is `null` and the site hides the breakdown.
- Completed games get `drivers: null`.

### Unchanged

`p_50_wins` and `p_top6` / `top6_prob` stay in the JSON and schema but are no longer displayed.

## Page layout (top to bottom)

1. **Header.** Celtics logo, "Boston Celtics 2026-27", current record, East rank, last-updated time.
2. **Next game.** The first upcoming game:
   - both logos with records, date, home/away
   - Boston win % as a large number, and the predicted margin
   - a breakdown chart: one horizontal bar per driver, diverging from zero (right helps Boston); the table view lists the values
   - one opponent context line, e.g. "DAL 6-7 · net rating 21st"
   - if no upcoming games: "Regular season complete."
3. **Season outlook.** Three tiles:
   - projected record with its 80% range
   - make playoffs
   - #1 seed in East

   Each tile shows the change since the previous `history.json` entry (▲/▼ with the amount; hidden when there is no previous entry).
4. **Team context.**
   - East rank and games back, last 10, streak.
   - Offense, defense and net rating, each with its league rank and a 30-position rank strip showing where Boston sits.
5. **Schedule, two tabs.**
   - *Upcoming:* opponent logo, home/away, rest, win-chance bar, predicted margin. Clicking a row expands its driver breakdown as text values.
   - *Results:* W/L and score, pre-game win %, and ✓/✗ for whether the favorite won. No mark when no pre-game value was logged.
6. **Projection over time.** Existing chart and table view.
7. **Win distribution and seed probabilities.** Side by side at ≥ 760px wide, stacked below that.
8. **East standings projection.** Logo for every team, Boston's row highlighted. Columns: team, record, projected wins, playoffs %, #1 seed %.
9. **Model report card.**
   - This season's Celtics record of picks: "Called X of N Celtics games" from completed games with a pre-game value, plus live log loss.
   - Backtest table.
   - Calibration chart and table inside a collapsed `<details>`.
10. **How it works.** A plain-English walkthrough of one prediction, using the next game's drivers when available, plus the existing method text.

Removed from display: the 50+ wins tile, the top-6 tile, and the "Top 6" columns (East table and history table).

## Logos

- `scripts/fetch_logos.py` downloads the 30 team logos once into `site/logos/{ABBR}.png`.
  - Target: small PNGs (≤ 128px, about 1 MB total).
  - The source is a public image host reachable from the user's Mac, verified at implementation time. NBA.com's CDN returns 403 from this machine.
  - The script fails loudly if any file is missing or not a valid PNG.
- The logo files are committed to the repo, by the user's decision. This replaces the original spec's "no logos" rule.
- Footer credit: "Team logos are trademarks of the NBA and its teams. Not affiliated with the NBA or the Boston Celtics."
- The page falls back to a team-color badge (abbreviation on the team's primary color) if a logo image fails to load. A static map of 30 primary colors lives in `app.js`.

## Visual polish

- One spacing scale (4/8/12/16/24/32px) and a type scale (12/14/16/20/28/40px), applied consistently.
- Every section header gets a one-line muted subtitle.
- At < 560px, tables hide secondary columns (rest, predicted margin) and keep the key ones.
- Logo sizes: 20px in tables, 56px in the next-game card, 28px in the header.
- Existing rules still apply:
  - Celtics green as the only accent; diverging driver bars use green (helps) vs a neutral gray (hurts)
  - light and dark mode
  - a table view for every chart
  - hover tooltips
  - no horizontal page scroll at 400px

## Testing

- **Python unit tests:**
  - `team_context` on hand-built games: ranks, games back, last 10, streak, ratings, def-rank direction.
  - `drivers` sum to `pred_margin` for ridge; the sign flips when away.
  - `drivers` is null for gbm and for completed games.
  - The schema rejects malformed new fields.
- **Pipeline test:** `run_update` output includes `team_context` for 30 teams and `drivers` on upcoming games.
- **Front end:**
  - `node --check`
  - a field-contract check against real generated JSON
  - the HTTP 200 smoke test
  - the logo files exist for all 30 teams and are valid PNGs
- **Manual (user):** a browser check in light and dark mode at phone width.
