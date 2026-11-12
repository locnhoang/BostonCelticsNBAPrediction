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
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
  let timer;
  let lastWidth = window.innerWidth;
  window.addEventListener("resize", () => {
    // Mobile browsers fire resize when the address bar shows or hides; re-render only on width change.
    if (window.innerWidth === lastWidth) return;
    lastWidth = window.innerWidth;
    clearTimeout(timer);
    timer = setTimeout(renderAll, 150);
  });
  try {
    renderAll();
  } catch (err) {
    console.error(err);
    document.getElementById("updated").textContent = `Some sections failed to render (${err.message}).`;
  }
}

main();
