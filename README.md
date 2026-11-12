# Boston Celtics 2026-27 Prediction Model

Daily-updated projections for the Boston Celtics: win probability for every remaining game, the distribution of final wins, seed probabilities, and playoff / play-in odds.

**Live site:** https://locnhoang.github.io/BostonCelticsNBAPrediction/

## How it works

1. **Data:** every regular-season game since 2014-15 from stats.nba.com (`nba_api`), for both game logs and the schedule, cached locally as parquet. The NBA CDN schedule was dropped because it returns 403.
2. **Features** (all point-in-time, so there's no leakage): margin-of-victory Elo, recency-weighted net rating per 100 possessions with a preseason prior, home court, rest days, back-to-backs, three-in-four.
3. **Model:** ridge regression predicts the point margin; win probability = Φ(margin / σ). It's compared against Elo-only, home-court-only, and gradient boosting (scikit-learn) in a walk-forward backtest (2018-19 → 2025-26).
4. **Simulation:** 10,000 runs of the full league's remaining schedule, with correlated team-strength uncertainty (tuned so 80% intervals cover about 80% of outcomes), simplified tiebreakers, and the play-in.
5. **Site:** static HTML/JS on GitHub Pages reading JSON written by the pipeline.

## Backtest results

| Model | Log loss | Brier | Accuracy |
|---|---|---|---|
| Home court only | 0.6868 | 0.2468 | 55.7% |
| Elo only | 0.6279 | 0.2192 | 64.6% |
| Ridge (live) | 0.6192 | 0.2152 | 65.5% |
| Gradient boosting (scikit-learn) | 0.6238 | 0.2172 | 65.3% |

Gradient boosting did not beat ridge by the 0.002 log-loss margin required to switch, so ridge is live.

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

- `.github/workflows/daily.yml` runs `celtics update` at 14:00 UTC and deploys the site.
- NBA.com sometimes blocks cloud IPs. If the daily job fails with "fetch blocked", install the local fallback with `scripts/install_launchd.sh` (runs daily at 10:30 from your Mac).
- The local fallback only runs on `main`, skips if CI already updated today, and supports `--force`.
- On macOS, the launchd job may need Full Disk Access for `/bin/bash`, or the repo moved out of Desktop, to read the folder.

## Limitations

- Team-level only: injuries and trades show up only through results (player-level adjustments are planned).
- Tiebreakers are simplified (head-to-head → division winner → conference record → random).
- The two NBA Cup–determined games per team are simulated against an average opponent until the NBA schedules them.

Not affiliated with the NBA or the Boston Celtics.
