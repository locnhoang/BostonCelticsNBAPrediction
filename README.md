# Boston Celtics 2026-27 Prediction Model

Daily-updated projections for the Boston Celtics: win probability for every remaining game, the distribution of final wins, seed probabilities, and playoff / play-in odds.

**Live site:** https://locnhoang.github.io/BostonCelticsNBAPrediction/

## How it works

1. **Data:** every regular-season game since 2014-15 from stats.nba.com (`nba_api`), for both game logs and the schedule, cached locally as parquet. The NBA CDN schedule was dropped because it returns 403.
2. **Features** (all point-in-time, so there's no leakage): margin-of-victory Elo, recency-weighted net rating per 100 possessions with a preseason prior, home court, rest days, back-to-backs, three-in-four.
3. **Model:** ridge regression predicts the point margin; win probability = Φ(margin / σ). It's compared against Elo-only, home-court-only, and gradient boosting (scikit-learn) in a walk-forward backtest (2018-19 → 2025-26).
4. **Simulation:** 10,000 runs of the full league's remaining schedule, with correlated team-strength uncertainty (tuned so 80% intervals cover about 80% of outcomes), simplified tiebreakers, and the play-in.
5. **Site:** static HTML/JS on GitHub Pages reading JSON written by the pipeline. It leads with the next game, including a breakdown of what drives the prediction (season strength, recent form, home court, rest, injuries & availability), then shows the season outlook, team context (standings, last 10, offensive/defensive rank), schedule, and the model's track record. Team logos live in `site/logos/` (user-supplied artwork, resized to 128px; `scripts/fetch_logos.py` is the original ESPN fetch and would overwrite them).
6. **Injuries:** the backtest derives each player's absences from box scores (who played in each game); live predictions use the ESPN injury feed instead. A Day-To-Day player's chance of missing the game is learned over time from logged outcomes (starts at 0.5). Each team's availability is the value of missing rotation players relative to its recent baseline, fed to the ridge model as `ridge_avail`. In the walk-forward backtest `ridge_avail` scored log loss 0.6140 vs 0.6193 for ridge, a 0.0053 gain that clears the 0.001 margin, so `ridge_avail` is live. The site shows only the Celtics' and the next opponent's injured players, Out first and then Day-To-Day.
7. **Pick record:** each run logs the model's win probability for upcoming Celtics games, and the last value before tip-off is kept. From 2026-11-18 on (`TRACK_PICKS_FROM` in `config.py`), each finished game is graded: the pick is correct if the side the model favored won. The site shows the running record (for example 5-2) on the next-game card and in the report card.

## Backtest results

| Model | Log loss | Brier | Accuracy |
|---|---|---|---|
| Home court only | 0.6868 | 0.2468 | 55.7% |
| Elo only | 0.6279 | 0.2192 | 64.6% |
| Ridge | 0.6193 | 0.2152 | 65.5% |
| Ridge + availability (live) | 0.6140 | 0.2130 | 65.9% |
| Gradient boosting (scikit-learn) | 0.6237 | 0.2172 | 65.2% |

Ridge + availability beat plain ridge by 0.0053 log loss (margin required: 0.001), so it is live. Gradient boosting did not beat ridge.

## Usage

```bash
uv sync
uv run celtics fetch        # refresh data cache
uv run celtics backtest     # walk-forward backtest -> site/data/backtest.json
uv run celtics update       # daily run -> site/data/*.json
uv run pytest               # tests (offline)
```

**Local setup note:** if the repo lives in an iCloud-synced folder such as Desktop or Documents, iCloud hides `.venv` contents and Python skips the editable-install `.pth` file. Fix: `mv .venv .venv.nosync && ln -s .venv.nosync .venv`.

## Automation

- The daily update runs from a Mac via launchd (`scripts/install_launchd.sh`, 10:30 local) and pushes to `main`; pushes to `site/` deploy the site. `.github/workflows/daily.yml` is manual-only, because stats.nba.com blocks GitHub-hosted runners.
- GitHub Pages must be set to Source "GitHub Actions" (repo Settings → Pages), or the deploy step will fail.
- The local fallback only runs when this clone is on `main`; if you develop on branches, use a separate clone for launchd.
- The local fallback skips if CI already updated today, and supports `--force`.
- On macOS, the launchd job may need Full Disk Access for `/bin/bash`, or the repo moved out of Desktop, to read the folder.

## Limitations

- Injury adjustments rely on the ESPN feed, which has no Questionable/Probable levels (only Out and Day-To-Day), and the backtest uses actual absences from box scores, so live accuracy is somewhat lower than the backtest suggests. Trades show up only through results.
- Tiebreakers are simplified (head-to-head → division winner → conference record → random).
- The two NBA Cup–determined games per team are simulated against an average opponent until the NBA schedules them.
- Neutral-site games (international games, NBA Cup semifinals in Las Vegas) are still modeled with home-court advantage for the nominal home team.
- Between Aug 1 and the next season's schedule release, the daily job exits with "schedule for season not published yet". That is expected, not an IP block.
- Team logos are trademarks of the NBA and its teams, shown with credit; this project is not affiliated with either.

Not affiliated with the NBA or the Boston Celtics.
