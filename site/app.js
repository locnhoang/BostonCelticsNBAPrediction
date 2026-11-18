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
  document.getElementById("hdr-logo").replaceChildren(teamMark(team, 64));
  document.getElementById("hdr-title").textContent = `Boston ${teamName(team)} ${pred.season}`;
  const started = c.wins + c.losses > 0;
  document.getElementById("hdr-meta").textContent = started
    ? `${c.wins}-${c.losses} · ${ord(c.conf_rank)} in the East`
    : `${c.wins}-${c.losses}`;
  updatedBase = `Updated ${new Date(pred.generated_at).toLocaleString()} · model ${pred.model_version}`;
  document.getElementById("updated").textContent = updatedBase;
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
      opp.wins + opp.losses === 0
        ? `${teamName(g.opponent)}: ${opp.wins}-${opp.losses}`
        : `${teamName(g.opponent)}: ${opp.wins}-${opp.losses}, net rating ${ord(opp.net_rank)} of 30, last 10 ${opp.last10}`,
    ),
  );
  renderDrivers(g, t);
  guarded("injury report", "injury-panel", () => renderInjuries(pred, g));
}

const STATUS_CLASS = { Out: "out", "Day-To-Day": "dtd" };

function playerRow(p) {
  let tag = "";
  if (p.status === "Out") tag = p.return_date ? `back ~${p.return_date}` : "";
  else tag = `~${Math.round((p.p_out ?? 0) * 100)}% out`;
  return el(
    "li",
    { class: "player" },
    el(
      "div",
      { class: "prow" },
      el("span", { class: `chip chip-${STATUS_CLASS[p.status]}` }, p.status),
      el("span", { class: "pname" }, p.name),
      tag ? el("span", { class: "ptag muted" }, tag) : "",
    ),
    p.status === "Out" && p.comment ? el("div", { class: "pcomment muted" }, p.comment) : "",
  );
}

function rosterColumn(abbr, players) {
  const byMinutes = (x, y) => (y.expected_minutes ?? 0) - (x.expected_minutes ?? 0);
  const listed = [
    ...players.filter((p) => p.status === "Out").sort(byMinutes),
    ...players.filter((p) => p.status === "Day-To-Day").sort(byMinutes),
  ];
  const col = el("div", { class: "inj-col" }, el("h4", { class: "team" }, teamMark(abbr, 24), el("span", {}, teamName(abbr))));
  if (!listed.length) {
    col.append(el("p", { class: "muted" }, "No injuries reported."));
    return col;
  }
  const list = el("ul", { class: "players", hidden: "" }, ...listed.map(playerRow));
  list.id = `inj-list-${abbr}`;
  const label = `Injury report (${listed.length})`;
  const btn = el("button", { class: "inj-toggle", type: "button", "aria-expanded": "false", "aria-controls": list.id }, `▸ ${label}`);
  btn.addEventListener("click", () => {
    list.hidden = !list.hidden;
    btn.setAttribute("aria-expanded", String(!list.hidden));
    btn.textContent = `${list.hidden ? "▸" : "▾"} ${label}`;
  });
  col.append(btn, list);
  return col;
}

function renderInjuries(pred, g) {
  const box = document.getElementById("injury-panel");
  const rep = pred.injury_report;
  if (!rep) {
    box.replaceChildren();
    return;
  }
  const parts = [el("h3", {}, "Injury report")];
  if (rep.available && g.win_prob_healthy != null && Math.round(g.win_prob * 100) !== Math.round(g.win_prob_healthy * 100)) {
    parts.push(el("p", { class: "inj-line" }, `${pct(g.win_prob_healthy)} healthy roster → ${pct(g.win_prob)} after the injury report`));
  }
  const teams = Object.entries(rep.teams ?? {});
  if (teams.length) parts.push(el("div", { class: "inj-cols" }, ...teams.map(([abbr, players]) => rosterColumn(abbr, players))));
  let source = "Injury report unavailable — prediction uses each team's usual availability.";
  if (rep.available) {
    const asOf = rep.as_of ? new Date(rep.as_of) : null;
    source = asOf && !Number.isNaN(asOf.getTime())
      ? `Injury report as of ${asOf.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })} (ESPN)`
      : "Injury report time unknown (ESPN)";
  }
  parts.push(el("p", { class: "muted" }, source));
  box.replaceChildren(...parts);
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
  const minValue = Math.min(...data.map((d) => d.value));
  const maxValue = Math.max(...data.map((d) => d.value));
  box.replaceChildren(
    Plot.plot({
      width: box.clientWidth,
      height: 40 * data.length + 40,
      marginLeft: 120,
      marginRight: 16,
      style: plotStyle(t),
      x: { domain: [Math.min(-1, minValue * 1.3), Math.max(1, maxValue * 1.3)], label: "Points toward Boston", grid: true, ticks: 5, tickFormat: "d" },
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
  const unit = kind === "wins" ? (Math.abs(d) === 1 ? "win" : "wins") : Math.abs(d) === 1 ? "pt" : "pts";
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
  const started = c.wins + c.losses > 0;
  const games = (n) => `${n} ${n === 1 ? "game" : "games"}`;
  const gb = !started
    ? ""
    : c.conf_rank === 1
      ? "Leads the conference"
      : c.games_back > 0
        ? `${games(c.games_back)} back`
        : c.games_back === 0
          ? "Tied for 1st"
          : `${games(-c.games_back)} ahead of 1st`;
  const facts = [
    ["East standing", started ? ord(c.conf_rank) : "—", started ? gb : "Season not started"],
    ["Last 10", c.last10, c.streak ? `Current streak: ${c.streak}` : ""],
  ];
  const ratings = [
    ["Offense", started ? c.off_rating.toFixed(1) : "—", c.off_rank, "Points scored per 100 possessions"],
    ["Defense", started ? c.def_rating.toFixed(1) : "—", c.def_rank, "Points allowed per 100 possessions"],
    ["Net rating", started ? signed(c.net_rating) : "—", c.net_rank, "Offense minus defense"],
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
          el("div", { class: "rating-head" }, el("span", { class: "tile-label" }, label), el("span", { class: "rating-rank" }, started ? `${ord(rank)} of 30` : "—")),
          el("div", { class: "fact-value" }, value),
          started ? rankStrip(rank, label) : "",
          el("div", { class: "tile-note" }, started ? `${note} · best on the left` : note),
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
// Exactly 0.5 counts as favoring Boston.
const calledIt = (g) => (g.pregame_win_prob >= 0.5) === g.result.startsWith("W");

function renderUpcoming(pred) {
  const rest = (g) => (g.b2b ? "Back-to-back" : `${Math.round(g.rest_days)}d rest`);
  const upcoming = pred.games.filter((g) => !g.completed);

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
    const rows = [];
    for (const g of upcoming) {
      const dateCell = el("td", {}, g.date);
      const row = el(
        "tr",
        g.drivers ? { class: "expandable" } : {},
        dateCell,
        el("td", {}, whereCell(g)),
        el("td", { class: "secondary" }, rest(g)),
        el("td", {}, probBar(g.win_prob)),
        el("td", { class: "secondary" }, signed(g.pred_margin)),
      );
      const group = [row];
      if (g.drivers) {
        const text = `${g.drivers.map((d) => `${d.label} ${signed(d.value)}`).join(" · ")} = ${signed(g.pred_margin)} projected margin`;
        const detail = el("tr", { class: "detail", id: `drv-${g.game_id}` }, el("td", { colspan: "5" }, text));
        detail.hidden = true;
        const chevron = el("span", { class: "chevron", "aria-hidden": "true" }, "▸");
        const button = el("button", { class: "expand", type: "button", "aria-expanded": "false", "aria-controls": `drv-${g.game_id}` }, chevron, g.date);
        dateCell.replaceChildren(button);
        const toggle = () => {
          detail.hidden = !detail.hidden;
          button.setAttribute("aria-expanded", String(!detail.hidden));
          chevron.textContent = detail.hidden ? "▸" : "▾";
        };
        button.addEventListener("click", (e) => {
          e.stopPropagation();
          toggle();
        });
        row.addEventListener("click", toggle);
        group.push(detail);
      }
      rows.push(group);
    }
    const LIMIT = 10;
    let showAll = false;
    const apply = () => {
      body.replaceChildren(...rows.slice(0, showAll ? rows.length : LIMIT).flat());
    };
    apply();
    const nodes = [el("table", {}, head, body)];
    if (rows.length > LIMIT) {
      const more = el("button", { class: "show-all", type: "button" }, `Show all ${rows.length} games`);
      more.addEventListener("click", () => {
        showAll = !showAll;
        more.textContent = showAll ? "Show fewer" : `Show all ${rows.length} games`;
        apply();
      });
      nodes.push(more);
    }
    up.replaceChildren(...nodes);
  }
}

function renderResults(pred) {
  const done = pred.games.filter((g) => g.completed).reverse();
  const mark = (g) => {
    if (g.pregame_win_prob == null) return "";
    const ok = calledIt(g);
    const label = ok ? "Model favored the winner" : "Model favored the loser";
    return el("span", { title: label, "aria-label": label, role: "img" }, ok ? "✓" : "✗");
  };
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
  const activate = (tab, focus) => {
    for (const other of tabs) {
      const on = other === tab;
      other.setAttribute("aria-selected", String(on));
      other.setAttribute("tabindex", on ? "0" : "-1");
      document.getElementById(other.getAttribute("aria-controls")).hidden = !on;
    }
    if (focus) tab.focus();
  };
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => activate(tab, false));
    tab.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      const step = e.key === "ArrowRight" ? 1 : -1;
      activate(tabs[(i + step + tabs.length) % tabs.length], true);
    });
  });
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
      x: { type: "utc", label: null, ticks: d3.utcDay.every(Math.max(1, Math.ceil(data.length / 8))), tickFormat: "%b %-d" },
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
  const note = document.getElementById("availability-note");
  note.textContent = backtest?.availability_note ?? "";
  note.hidden = !backtest?.availability_note;
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
    `that margin works out to a win chance of ${pct(g.win_prob)} for the ${teamName(pred.team_summary.team)}.`;
}

async function loadJson(path) {
  const resp = await fetch(path, { cache: "no-store" });
  if (!resp.ok) throw new Error(`${path}: HTTP ${resp.status}`);
  return resp.json();
}

let state = null;

let failed = false;
let updatedBase = "";

// Run one section renderer on its own so a failure only blanks that section.
function guarded(name, containerId, fn) {
  try {
    fn();
  } catch (err) {
    console.error(`Section "${name}" failed to draw`, err);
    failed = true;
    document.getElementById(containerId)?.replaceChildren(el("p", { class: "muted" }, "This section could not be drawn."));
  }
}

function safeRender() {
  if (!state) return;
  failed = false;
  const t = theme();
  const { pred, history, backtest } = state;
  guarded("header", "hdr-meta", () => renderHeader(pred));
  guarded("next game", "next-card", () => renderNextGame(pred, t));
  guarded("outlook", "outlook", () => renderOutlook(pred, history));
  guarded("context", "context", () => renderContext(pred));
  guarded("upcoming games", "upcoming", () => renderUpcoming(pred));
  guarded("results", "completed", () => renderResults(pred));
  guarded("history", "chart-history", () => renderHistory(history, t));
  guarded("wins", "chart-wins", () => renderWins(pred, t));
  guarded("seeds", "chart-seeds", () => renderSeeds(pred, t));
  guarded("east", "east", () => renderEast(pred));
  guarded("report card", "models", () => renderReportCard(pred, backtest, t));
  guarded("how it works", "how-example", () => renderHowExample(pred));
  if (failed) {
    const u = document.getElementById("updated");
    u.textContent = `${updatedBase} · some sections failed to draw`;
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
