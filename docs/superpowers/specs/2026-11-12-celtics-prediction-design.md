# Boston Celtics 2026-27 Prediction Model — Design

**Date:** 2026-11-12
**Status:** Approved design, pending implementation plan

## Goal

Predict the Boston Celtics' 2026-27 regular season:

1. **Per-game win probability** for every remaining game.
2. **Season outcome distribution:** final win total, seed, and playoff / play-in probabilities.

The project serves three purposes at once: a learning project, a portfolio piece, and an accuracy-focused tool that updates daily during the season. Results are published on a public website.

## Scope

**Phase 1 (this spec):** team-level model, full-league season simulation, walk-forward backtest, daily automated pipeline, static website.

**Phase 2 (out of scope, future spec):** player-availability adjustments (injuries, trades, lineups) and interactive what-if scenarios. Any phase 2 feature must improve backtest log loss to be adopted.

**Explicitly out of scope:** live in-game win probability, playoff series simulation beyond the play-in, betting-market data.

## Architecture

Static site + scheduled pipeline. The pipeline computes everything and writes JSON; the website only reads JSON and never contacts NBA.com.

```
BostonCeltics/
├── src/celtics/
│   ├── data/          # nba_api fetching + local parquet cache
│   ├── features/      # raw games → point-in-time model features
│   ├── models/        # Elo baseline, ridge (main), LightGBM (challenger)
│   ├── simulate/      # Monte Carlo season + standings + tiebreakers + play-in
│   ├── evaluate/      # walk-forward backtest, metrics, calibration
│   ├── export/        # JSON writers + schema validation
│   └── cli.py         # `celtics update`, `celtics backtest`, `celtics train`
├── data/              # cached raw data (gitignored)
├── site/
│   ├── index.html, app.js, styles.css
│   └── data/          # predictions.json, history.json (committed by pipeline)
├── tests/
├── notebooks/methodology.ipynb   # optional portfolio walkthrough
└── .github/workflows/
    ├── daily.yml      # scheduled pipeline run
    └── ci.yml         # tests + lint on push
```

**Data flow:** nba_api → parquet cache → features → model → per-game probabilities → season simulation → JSON → website.

Each stage is a separate module with one responsibility and a DataFrame-based interface, testable in isolation.

## Data

- **Source:** `nba_api` (stats.nba.com).
- **Coverage:** all NBA regular-season games, 2014-15 through the current season, plus the current season's full schedule.
- **Caching:** completed past seasons are fetched once and stored as parquet. Daily runs fetch only new games and the current schedule.
- **Rate limiting:** requests are spaced out; failed requests retry with exponential backoff.
- **NBA Cup:** the championship game is excluded from regular-season records and standings. Group-stage and knockout games before the final count normally.

## Model

**Target:** home-team point margin. Converted to win probability with a normal CDF: `P(home win) = Φ(predicted_margin / σ)`, where σ is the residual standard deviation estimated from training data (expected ≈ 12).

**Features** (all point-in-time: computed only from games before the game being predicted):

1. **Elo difference.** Margin-of-victory adjusted Elo; ratings regress 25% toward the mean between seasons.
2. **Net rating (per 100 possessions) for each team**, exponentially weighted toward recent games. Early season, blended with a prior (previous season's net rating regressed toward the mean); the prior's weight decays as games are played.
3. **Home court** indicator.
4. **Rest:** days since last game, back-to-back flag, 3-games-in-4-nights flag (for each team).

**Models:**

| Role | Model | Notes |
|---|---|---|
| Baseline | Elo only | Benchmark |
| Main | Ridge regression on all features | Default production model |
| Challenger | LightGBM on all features | Replaces ridge only if it clearly wins the backtest on log loss |

**Retraining cadence:** model coefficients are refit weekly. Daily runs update Elo and rolling features with new results, without refitting.

## Season Simulation

- **10,000 simulations** of the full remaining league schedule (all 30 teams).
- **Correlated team strength:** in each simulation, every team draws a strength offset from a distribution reflecting uncertainty in its true rating. The offset applies to all of that team's remaining games, so outcomes within a simulation are correlated. The offset distribution's width is tuned so that backtest win-total intervals are calibrated.
- **Game outcomes:** sampled margin `~ Normal(predicted_margin + offsets, σ)`; win if margin > 0.
- **Standings:** per conference, ordered by win percentage.
- **Tiebreakers (simplified):** head-to-head record → division winner → conference record → random draw. Deeper official tiebreakers are omitted and documented as such.
- **Play-in:** seeds 7–10 play the play-in games, simulated with the same model, to decide seeds 7 and 8.

**Outputs:**

- Celtics win probability and predicted margin for each remaining game.
- Distribution of Celtics final wins: median, 50% and 80% intervals, P(≥50 wins).
- Celtics seed probabilities (1–15), P(top 6), P(play-in), P(make playoffs after the play-in), P(#1 seed in East), P(miss postseason).
- Projected wins and playoff probability for every Eastern Conference team.

## Evaluation

**Walk-forward backtest:** for each test season 2018-19 through 2025-26, train on all prior seasons only, then predict each game using only information available before tip-off.

**Per-game metrics:** log loss (primary), Brier score, accuracy, calibration curve (predicted-probability buckets vs. observed win rate).

**Season-level calibration:** simulate past seasons starting from checkpoints (e.g., after each team's 15th game). Check that actual final win totals fall inside the predicted 80% interval about 80% of the time.

**Comparisons:** main model vs. Elo-only vs. home-court-only coin flip.

**Reference target:** strong public team-level NBA models reach roughly 66–68% accuracy and log loss ≈ 0.60–0.62. Results are reported honestly against this range.

**Feature acceptance rule:** a new feature or model is adopted only if it improves walk-forward log loss.

## Website

Single static page (HTML/JS, no build step) on GitHub Pages. Charts use Observable Plot via CDN. Reads `site/data/predictions.json` and `site/data/history.json`.

**Sections, top to bottom:**

1. **Headline:** current record, projected record (median + 80% range), playoff %, top-6 %, #1 seed %, last-updated timestamp.
2. **Projection over time:** projected wins with an 80% band, one point per daily run.
3. **Remaining schedule:** date, opponent, home/away, rest situation, win probability (inline bar), predicted margin. Completed games show the result next to the pre-game prediction.
4. **Win distribution:** histogram of simulated final win totals.
5. **Seed probabilities:** seeds 1–10 plus "miss postseason".
6. **East standings projection:** projected wins and playoff % for every Eastern team.
7. **Model report card:** backtest metrics vs. baselines, calibration chart, current-season live accuracy.
8. **Methodology:** short explanation, link to repo.

**Design:** mobile-friendly; light and dark mode; Celtics green as the accent color only. No NBA or team logos or other trademarked branding.

**JSON contract:**

- `predictions.json`: current snapshot with keys `generated_at`, `model_version`, `season`, `team_summary`, `games`, `seed_probabilities`, `conference_table`, `metrics` (`metrics.live` = this season's accuracy on logged pre-game predictions).
- `history.json`: array of daily entries `{date, record, projected_wins_median, p10, p90, playoff_prob, top6_prob, seed1_prob}`.
- `backtest.json`: written by `celtics backtest`. Holds per-model scores, the calibration table, interval coverage per team-uncertainty value, and the chosen value.
- `pregame_log.json`: the latest prediction for each Celtics game, frozen once the game is played, so completed games show a real pre-game prediction.
- All files are validated against a schema before writing. Invalid output stops the run, and the previous files remain.

## Operations

**Daily workflow** (GitHub Actions, about 9:00 ET):

1. Restore data cache.
2. Run `celtics update` (fetch new games, update Elo and features, refit weekly, simulate, export).
3. Validate JSON.
4. Commit `site/data/` if changed. GitHub Pages redeploys.

**Fallback:** NBA.com often blocks cloud IPs. If the fetch is blocked, the workflow fails with a clear "fetch blocked" error and publishes nothing; the site keeps showing the last good run with its timestamp. A `launchd` config is provided so the same `celtics update` command runs daily from the user's Mac and pushes results.

## Error Handling

- **Fetching:** retries with exponential backoff; fails loudly after the retry limit.
- **Data checks before modeling:** no duplicate game IDs; every completed game has scores; exactly 30 teams; 80–82 scheduled games per team in the current season. Until mid-December the NBA lists only 80; the last 2 depend on NBA Cup results. Missing games are simulated against a league-average opponent at a neutral site. Any failure aborts the run before publishing.
- **Output checks:** probabilities in [0, 1]; each team's seed probabilities sum to 1; league-wide simulated wins equal losses.
- **Postponed games:** remain in the remaining schedule using the NBA-listed date.

## Testing

- **Unit tests** (pytest) on hand-built fixtures: Elo update math, rolling features (including a no-future-leakage test), tiebreaker ordering, play-in bracket logic, margin-to-probability conversion.
- **Simulation sanity checks:** equal teams come out near 50%; every simulated season has 82 games per team; league-wide wins equal losses.
- **Pipeline smoke test:** full `update` run on a small saved data sample with no network access.
- **CI:** tests + ruff lint on every push; CI never calls NBA.com.

## Tooling

Python 3.12, `uv`, pandas, pyarrow, scikit-learn, LightGBM, `nba_api`, pytest, ruff. Frontend: plain HTML/CSS/JS + Observable Plot (pinned CDN version).
