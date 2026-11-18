# League-Wide Site — Design

**Date:** 2026-11-18
**Status:** Approved 2026-11-18, pending implementation plan
**Builds on:**
- `2026-11-12-celtics-prediction-design.md` (model, simulation, backtest)
- `2026-11-12-frontend-refresh-design.md` (site layout, drivers)
- `2026-11-13-injury-adjustments-design.md` (availability model, injury panel)

## Intent

The site moves from Celtics-only to the whole league. It should predict every NBA game each day and let any visitor follow the teams they care about. Each followed team gets the full page the Celtics have today.

It serves three audiences, and the user named all three:
- fans of other teams, who should land on something useful whatever team they follow
- the user's portfolio, where it shows a whole-league model, not a single-team page
- the user's own use: checking the day's games and keeping tabs on the Celtics

## Goals

1. A **Home** tab with the day's full slate: every game, with both teams' win chances and the model's pick.
2. **Team tabs:** any team can be followed, and each followed team gets a tab. The tab holds everything on the current Celtics page.
3. A **model pick record** across all games, plus a per-team record on each team page.
4. A **light/dark switch**.
5. The repo and site are **renamed** to `NBAPredictor`.

## Non-goals

- Accounts or server-side storage of followed teams.
- Changes to the model, its features, the simulation or the backtest. The model already covers all 30 teams.
- Playoff or bracket predictions.
- Live in-game updates. Data still refreshes once a day.
- Betting picks (moneyline, spread, totals, player props). The user wants a "best picks of the day" feature; it is a separate project designed after this one, and it will live in its own **Best picks** tab in the top bar (user's decision). Routing and the nav should make adding that tab easy; it is not built here.

## Anti-goals

These would make it a failure even if it technically works.

- **The Celtics view loses anything** it has today: report card, next game with drivers and injuries, outlook, context, schedule, wins over time, win distribution, seed odds, conference table.
- **History filled in with hindsight.** Records and wins-over-time charts start on the day they were really logged.
- **A slow Home tab.** Home must not download all 30 team files.
- **A broken daily update.** The Mac launchd runner must keep publishing without manual steps.

## Constraints

- **Static hosting:** the site stays static HTML/JS on GitHub Pages.
- **No server:** followed teams and the theme choice live in the visitor's browser (`localStorage`). A new device starts with no teams followed. Every read and write is wrapped in try/catch, and the site works without storage.
- **Mac runner:** data is produced by the Mac runner (`~/celtics-runner`, launchd at 10:30 local). It runs `scripts/run_update.sh`, which runs `celtics update` and pushes `site/data`. GitHub-hosted runners can't reach stats.nba.com.
- **Repo growth:** every day's data is committed, so daily data churn must stay modest. Aim for well under 1 MB of changed JSON per day. Prefer compact JSON for large files, and don't store per-game drivers for every future game of every team.
- **Libraries:** existing front-end libraries only (Observable Plot 0.6.16, d3 7.9.0). No build step.

## Approaches

| Approach | What it is | Trade-off |
|---|---|---|
| **A. One page with tabs (chosen)** | Extend the current site. The pipeline writes a slate file plus one file per team. The existing renderers take a team as input. Hash routes give each tab a shareable link. | Reuses everything and adds no tooling. Search engines see one page. |
| B. Static page per team | The pipeline writes 31 HTML pages. | Cleaner URLs and per-team SEO, but layouts repeat, and the followed-team tabs still need client JS. Same work as A, more files. |
| C. Framework rewrite (React/Vite) | Rebuild the front end. | A résumé line, but it rewrites a working site and adds a build step to every deploy. |

The user chose **A**.

## Site design

### Mockups

The look the user reacted to is saved in `assets/2026-11-18-league-wide/`:
- `home-dark.html` and `home-dark.png`
- `home-light.html`, the earlier light version

Open the HTML files in a browser; logos are referenced as `/files/{ABBR}.png` and map to `site/logos/`.

The user asked for dark mode on seeing the light version, then for a light/dark switch. The top bar stays black in both themes, like ESPN's (builder's call, made by the controller).

### Navigation (user's decisions)

An ESPN-style dark top bar:

```
NBA PREDICTOR | Home | (logo) Celtics ★ | (logo) Knicks ★ | + Add team        ☀/☾
```

- **Home** is the first tab and the landing page.
- Each followed team has a tab with its logo and name.
- **+ Add team** opens a picker of all 30 teams (logos, grouped by conference). Picking a team follows it and opens its tab.
- On narrow screens the tab row scrolls horizontally, and the page itself never scrolls sideways.

### Routes (hash)

- `#/` → Home
- `#/team/BOS` → team page (works for any team, followed or not)
- `#/game/{game_id}` → game detail
- `#/about` → model details (see below)
- An unknown route shows Home.

### Following teams

- A team page shows **☆ Follow** / **★ Following**. Following adds the team's tab, and unfollowing removes it.
- Tab order is the order teams were followed.
- Storage key: `followed` (a list of abbreviations). The first visit has none.
- Any team name or logo on a game card links to that team's page.

### Theme (user's decision: a switch)

- A sun/moon toggle in the nav bar.
- Until the visitor uses it, the site follows the device setting (`prefers-color-scheme`), as it does today.
- Once used, the choice is saved (`theme` = `light` | `dark`) and applied through `data-theme` on `<html>`.
- Charts redraw with the new theme's colors.
- The dark palette follows the mockup: near-black page, dark cards, white for the pick and the winning score, green ✓ and red ✗.

### Home tab (user's decision: today's slate first)

```
Tue Nov 17 | [Wed Nov 18] | Thu Nov 19
Model pick record, all games: 3-2 yesterday · 41-22 since Dec 1 (65%)
TODAY · 12 GAMES
[card] [card] [card]
YESTERDAY · FINAL
[card ✓] [card ✗]
```

**Day strip:**
- Yesterday, today and tomorrow, with today selected.
- "Today" is the Eastern date of the run (the same notion the pipeline uses for freezing pre-game picks).

**Banner:** the league-wide pick record, for yesterday and since tracking began.

**Upcoming game card:**
- away team, then home team, each with logo, name and record
- each team's win chance; the pick is bold
- a probability bar
- "Model pick: X"
- Tapping the card opens game detail.

**Final game card:**
- final score, with the winner bold
- "Model picked X (p%)" with ✓ or ✗

**Card order:** by game_id within a day (the NBA's own order). Builder's call: tip-off time may replace it if the schedule provides one.

**Empty days:** a day with no games shows "No games".

### Game detail (`#/game/{id}`)

The current Celtics next-game card, generalized to any game:
- the matchup
- the win chance after the injury report, and the healthy-roster line
- the "why" drivers chart, from the home team's perspective with each bar labelled
- both teams' injury panels, listing Out first and then Day-To-Day, with no available players

This covers games on the Home day strip that haven't been played. A final game shows its score and the logged pre-game pick.

### Team tab (`#/team/{ABBR}`)

The current Celtics page, top to bottom, with the team as a parameter:
1. Header with logo, record and conference rank. Plus ☆/★ follow.
2. **Model report card:** the pick record on this team's games.
   - Boston's record keeps its 2026-11-18 start.
   - Other teams start on launch day.
3. **Next game:** the same content as game detail for the team's next game, but drivers and win chance are shown **from this team's side**, as the Celtics page does today. Game detail (`#/game/`) shows the home team's side. Store drivers home-side in `slate.json` and team-side in `teams/{ABBR}.json`; one renderer can take a sign. (Controller's call: keeps the Celtics view unchanged.)
4. Season outlook
5. Team context
6. Schedule: upcoming games and results, with ✓/✗ on tracked games
7. Projected wins over time. For teams other than Boston this starts on launch day, and that is said on the chart while it has under 2 points.
8. Final win total distribution
9. Seed odds, for the team's own conference
10. Conference projection table, for the team's own conference, with the team highlighted

### About (`#/about`)

The current "Backtest: past seasons" section (models table, calibration) and "How it works" move here. That keeps team pages focused. A link to it sits at the end of the nav bar and in the footer. (User approved.)

## Data

All files are validated by pydantic schemas before writing, like today.

### `site/data/slate.json` (Home)

- `generated_at`
- `model_version`
- `days`: three entries (yesterday, today, tomorrow), each `{date, games: [...]}`

Each game in a day:
- `game_id`, `date`, `home`, `away`
- `home_record`, `away_record`
- `completed`
- `home_win_prob` (current, injury-adjusted; null once completed)
- `home_win_prob_healthy`
- `pred_margin` (home perspective)
- `pick`: abbreviation, or null
- `home_pts`, `away_pts` (completed only)
- `pregame_home_win_prob`: the logged pick (null if none)
- `correct`: bool or null
- `drivers` (home perspective), for incomplete games only
- `injuries`: `{home: [...], away: [...]}`, the injured-player rows as today (Out first, then Day-To-Day), for incomplete games only

It also carries `record`: `{since, yesterday: {correct, wrong}, total: {correct, wrong}}` for all games.

### `site/data/teams/{ABBR}.json` (team tabs)

This is today's `predictions.json`, keyed to that team:
- `team_summary`, `games` (from the team's perspective), `seed_probabilities`, `conference_table`
- `metrics.record`
- `injury_report` for the team's next game

`team_context` for all 30 teams is identical across files, so it moves to `site/data/league.json` with `generated_at` and is loaded once.

Drivers are stored for the team's **next game only**, to limit size and churn. The schedule shows win chances, not drivers.

### `site/data/history/{ABBR}.json`

Per-team projected-wins history, in today's `history.json` entry format. Boston's existing `history.json` moves to `history/BOS.json`.

### `site/data/pregame_log.json`

- Every game in the league, keyed by `game_id`.
- Entries store the **home** win probability: `{home_win_prob, pred_margin (home), logged_at}`.
- The same freeze rule as today: refreshed until the game's Eastern date passes, never back-filled.
- **Migration:** existing entries hold Boston's win probability. Convert each one to the home perspective using the game's home team, once, in the code that reads the old format, so nothing is lost.

### Pick definition

- A game's pick is the home team if `pregame_home_win_prob >= 0.5`, else the away team.
- `correct` means the pick won.
- Team records count `correct` on the team's games.
- This replaces "exactly 0.5 favors Boston" with "exactly 0.5 favors home". It is the same in practice: probabilities are stored to 4 decimals.

### Tracking start dates

- `TRACK_PICKS_FROM` stays `2026-11-18` for Boston.
- A new `LEAGUE_TRACK_FROM` is set to the launch date. It applies to the league banner and to every team other than Boston.

### Retired files

- `predictions.json` and `history.json` are replaced by the files above, in the same commit that ships the new site.
- `dtd_log.json` and `backtest.json` stay as they are. The Day-To-Day log already covers every team.

## Pipeline changes

1. **Run the export for every team.** The `TEAM` constant stops driving the export. It may remain only as the default for anything that is Boston-specific, such as `TRACK_PICKS_FROM`.
2. **Injury reports for every team in the slate**, not just the Celtics game. The ESPN feed is already league-wide.
   - The team roster fetch existed to show Available players, and those are no longer shown.
   - Builder's call: drop the roster fetch if matching only needs the current-season player log. Otherwise fetch rosters for teams playing in the slate only.
3. **"Not backwards" check league-wide.** Count completed games across the league, against the last published slate/league file.
4. **Healthy-roster probability for every slate game.**
5. **Atomic publish.** If any file fails validation, nothing is written. Same exit codes as today.
6. **`scripts/run_update.sh`:** the "already updated today" check looks at `site/data/history.json`. It must point at a file that changes every run, such as `site/data/league.json`.

## Rename (user's decision)

- At launch, rename the GitHub repo `BostonCelticsNBAPrediction` → `NBAPredictor`. The site moves to `https://locnhoang.github.io/NBAPredictor/`. The old Pages address stops working, and that was accepted.
- Update the `origin` URL in both clones: the Desktop repo and `~/celtics-runner`.
- Update the page title, meta description, the footer source link and the README.
- Remove the "not affiliated with the Boston Celtics" wording; the general NBA disclaimer stays.
- The rename is the last step, after the new site is merged and live under the old name.
- **Who does what (controller's call):** `gh` is not installed, so the user renames the repo in GitHub Settings (one field). The builder then updates both `origin` URLs, checks a push from `~/celtics-runner`, and confirms the new Pages URL serves the site. Both clones are outside the plan's code changes, so this is a short checked phase at the end of the plan.
- **Not renamed (controller's call):** the Python package and CLI `celtics`, the launchd label `com.celtics.update` and its plist, the `~/celtics-runner` folder, and the `TEAM` constant. Renaming them would touch the live runner for no visible gain.

## Testing

**Unit tests:**
- slate building for a three-day window, including days with no games
- pick and `correct`, including the 0.5 case
- league and team records, with each team's start date
- pregame-log migration from Boston-perspective entries
- per-team file contents match today's `predictions.json` for BOS (same fields and values, apart from the planned changes)

**Pipeline tests (synthetic league):**
- all 30 team files, `slate.json`, `league.json` and the history files are written and valid
- an injury feed failure still publishes, with no adjustment
- invalid data exits 3 and writes nothing

**Site checks:**
- `node --check`
- a headless-Chrome render of `#/`, `#/team/BOS`, `#/team/NYK`, a `#/game/` route and `#/about`, in light and dark, with no section showing a render error

**Size check:** report the total `site/data` size and one day's diff size from a real run.

## Left to the builder

- File and function names inside `src/` and `site/`.
- Whether `app.js` splits into modules (plain `<script type="module">` is fine, with no build step).
- Exact spacing and colors, within the mockup's look.
- Tip-off times, if `ScheduleLeagueV2` provides them cheaply.
