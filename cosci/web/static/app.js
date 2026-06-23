"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
const txt = (s) => document.createTextNode(s);
const spinner = () => el("span", "spin");

const api = {
  async get(path) { const r = await fetch(path); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); return r.json(); },
  async send(method, path, body) {
    const r = await fetch(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || r.statusText);
    return data;
  },
};

let loadedConfig = null;
let activeRunId = null;
let eloChart = null;
let poller = null;
let timer = null;
let runStartedAt = null;
let logRunId = null;
let logCursor = 0;

/* ── views ─────────────────────────────────────────────── */
function showView(name) {
  $("#view-welcome").hidden = name !== "welcome";
  $("#view-results").hidden = name !== "results";
  $("#view-config").hidden = name !== "config";
  $("#nav-config").classList.toggle("active", name === "config");
}

/* ── grounding badge ───────────────────────────────────── */
function renderGrounding(cfg, faithful) {
  const backend = cfg?.grounding?.backend ?? "arxiv";
  const badge = $("#grounding-badge");
  const note = faithful ? "web search" : (backend === "none" ? "parametric only" : "local convenience");
  badge.replaceChildren(el("span", "dot " + (faithful ? "faithful" : "local")), txt(`grounding: ${backend} · ${note}`));
}

/* ── runs archive ──────────────────────────────────────── */
async function loadRuns() {
  const list = $("#runs-list");
  list.replaceChildren();
  let runs = [];
  try { runs = await api.get("/api/runs"); } catch (e) { /* none yet */ }
  if (!runs.length) { list.appendChild(el("li", "empty", "No runs yet.")); return runs; }
  for (const r of runs) {
    const li = el("li");
    const b = el("button");
    b.dataset.id = r.id;
    b.appendChild(el("span", "run-goal", r.goal || "(untitled)"));
    const elo = r.best_elo != null ? `Elo ${Math.round(r.best_elo)}` : "Elo —";
    b.appendChild(el("span", "run-meta", `${r.id.split("_").slice(0, 2).join(" ")} · ${r.n_hypotheses} hyp · ${elo}`));
    b.addEventListener("click", () => selectRun(r.id));
    if (r.id === activeRunId) b.classList.add("active");
    li.appendChild(b);
    list.appendChild(li);
  }
  return runs;
}

/* ── results ───────────────────────────────────────────── */
async function selectRun(id) {
  activeRunId = id;
  document.querySelectorAll(".runs-list button").forEach((b) => b.classList.toggle("active", b.dataset.id === id));
  let run;
  try { run = await api.get(`/api/runs/${id}`); } catch (e) { return; }
  $("#result-id").textContent = id;
  $("#result-goal").textContent = run.goal || "(untitled run)";
  renderGroundingStatus(run.grounding);
  $("#overview").textContent = (run.overview && run.overview.trim()) ? run.overview : "No overview was produced for this run.";
  renderHypotheses(run.hypotheses || []);
  renderTournament(run.tournament || []);
  renderEloChart(run.elo_trajectory || []);
  loadEventsFor(id, true);          // show this run's recorded activity timeline
  showView("results");
}

function safetyTag(s) {
  const v = (s || "unreviewed").toLowerCase();
  return el("span", `tag ${v}`, v);
}

/* Plain statement of how literature-backed the run is. 0/N => arXiv returned
   nothing or rate-limited; the run ran on parametric reasoning, not sources. */
function renderGroundingStatus(g) {
  const node = $("#result-grounding");
  if (!g || g.reviews_total == null) { node.hidden = true; return; }
  node.hidden = false;
  const grounded = g.reviews_grounded || 0, total = g.reviews_total || 0;
  const ok = grounded > 0;
  node.className = "result-grounding " + (ok ? "is-grounded" : "is-ungrounded");
  node.replaceChildren(
    el("span", "dot " + (ok ? "faithful" : "local")),
    txt(`grounding: ${grounded}/${total} reviews grounded`),
    txt(ok ? "" : " · ran on parametric reasoning (no literature sources retrieved)"),
  );
}

function renderHypotheses(hyps) {
  const tb = $("#hyp-table tbody"); tb.replaceChildren();
  $("#hyp-count").textContent = hyps.length ? `n = ${hyps.length}` : "";
  hyps.forEach((h, i) => {
    const tr = el("tr");
    tr.appendChild(el("td", "rank", String(i + 1)));
    tr.appendChild(el("td", "mono", h.id));
    tr.appendChild(el("td", "num", h.elo_rating != null ? Math.round(h.elo_rating) : "—"));
    const safe = el("td"); safe.appendChild(safetyTag(h.safety)); tr.appendChild(safe);
    const org = el("td"); org.appendChild(el("span", "tag origin", h.origin || "generated")); tr.appendChild(org);
    tr.appendChild(el("td", "title-cell", h.title || ""));
    tb.appendChild(tr);
  });
}

function renderTournament(matches) {
  const tb = $("#match-table tbody"); tb.replaceChildren();
  $("#match-count").textContent = matches.length ? `${matches.length} matches` : "";
  matches.forEach((m, i) => {
    const before = m.elo_before?.[m.winner_id];
    const after = m.elo_after?.[m.winner_id];
    const delta = (before != null && after != null) ? (after - before) : null;
    const tr = el("tr");
    tr.appendChild(el("td", "rank", String(i + 1)));
    tr.appendChild(el("td", "mono", (m.mode || "").replace("_", " ")));
    tr.appendChild(el("td", "mono", m.winner_id));
    const d = el("td", "num " + (delta > 0 ? "delta-up" : "delta-down"), delta != null ? (delta > 0 ? "+" : "") + delta.toFixed(1) : "—");
    tr.appendChild(d);
    tr.appendChild(el("td", "mono", m.loser_id));
    tb.appendChild(tr);
  });
}

function renderEloChart(traj) {
  const ctx = $("#elo-chart");
  if (typeof Chart === "undefined") {
    const c = ctx.getContext("2d");
    c.clearRect(0, 0, ctx.width, ctx.height);
    c.font = "12px IBM Plex Mono"; c.fillStyle = "#69798a"; c.textAlign = "center";
    c.fillText("Chart library unavailable.", ctx.width / 2, ctx.height / 2);
    return;
  }
  if (eloChart) eloChart.destroy();
  const ink = "#13212e", line = "#2563a0", grid = "rgba(18,58,94,0.10)", mono = "IBM Plex Mono";
  eloChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: traj.map((p) => p.match),
      datasets: [{
        data: traj.map((p) => p.best_elo),
        borderColor: line, backgroundColor: "rgba(37,99,160,0.08)",
        borderWidth: 1.75, pointRadius: traj.length > 40 ? 0 : 2.5, pointBackgroundColor: line,
        fill: true, tension: 0.18,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { backgroundColor: ink, titleFont: { family: mono }, bodyFont: { family: mono }, displayColors: false, callbacks: { title: (it) => `match ${it[0].label}`, label: (it) => `best Elo ${it.parsed.y.toFixed(1)}` } } },
      scales: {
        x: { title: { display: true, text: "tournament match", font: { family: mono, size: 10 }, color: "#69798a" }, grid: { color: grid }, ticks: { font: { family: mono, size: 10 }, color: "#69798a", maxTicksLimit: 12 } },
        y: { title: { display: true, text: "best Elo", font: { family: mono, size: 10 }, color: "#69798a" }, grid: { color: grid }, ticks: { font: { family: mono, size: 10 }, color: "#69798a" } },
      },
      animation: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? false : { duration: 500 },
    },
  });
  if (!traj.length) {
    const c = ctx.getContext("2d");
    c.font = "12px IBM Plex Mono"; c.fillStyle = "#69798a"; c.textAlign = "center";
    c.fillText("No tournament matches recorded for this run.", ctx.width / 2, ctx.height / 2);
  }
}

/* ── live event log ────────────────────────────────────── */
function clip(s, n) { s = String(s == null ? "" : s); return s.length > n ? s.slice(0, n - 1) + "…" : s; }

// Map each event to [category, one-line summary]. Category drives the dot colour.
function eventSummary(e) {
  const f = e;
  switch (e.event) {
    case "run_started":        return ["run",    `started · ${clip(f.goal, 36)} (${f.mode || ""})`];
    case "run_done":           return ["done",   `complete · ${f.hypotheses} hypotheses, ${f.matches} matches`];
    case "run_aborted":        return ["abort",  `aborted · ${f.reason || "safety review"}`];
    case "run_error":          return ["error",  `failed · ${clip(f.error, 60)}`];
    case "task":               return ["task",   `${f.agent} · ${f.action}${f.target ? " " + f.target : ""}`];
    case "snapshot":           return ["task",   "snapshot saved"];
    case "generation_started": return ["gen",    `generating · ${f.strategy}`];
    case "generation_done":    return ["gen",    `${f.strategy} → ${f.hypotheses} hyp`];
    case "grounding_search":   return ["search", `${f.backend} ← ${clip(f.query, 32)}`];
    case "grounding_result":   return [f.articles > 0 ? "search" : "warn", `${f.articles} article${f.articles === 1 ? "" : "s"} retrieved`];
    case "reflection_started": return ["review", `reviewing ${f.hypothesis_id}`];
    case "reflection_done":    return [f.grounded ? "review" : "warn", `${f.hypothesis_id} · ${f.grounded ? "grounded" : "ungrounded"} · ${f.safety}`];
    case "ranking_match":      return ["match",  `${f.a} vs ${f.b} → ${f.winner} (${f.elo_delta >= 0 ? "+" : ""}${f.elo_delta})`];
    case "evolution_done":     return ["evolve", `${(f.parents || []).join(" + ")} → ${f.child}`];
    default:                   return ["task",   e.event];
  }
}

function appendEvents(events) {
  const log = $("#event-log");
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
  for (const e of events) {
    const [cat, text] = eventSummary(e);
    const li = el("li", "event-row");
    li.appendChild(el("span", "event-tick", e.tick != null ? `t${e.tick}` : "·"));
    li.appendChild(el("span", `event-dot ${cat}`));
    li.appendChild(el("span", "event-text", text));
    log.appendChild(li);
  }
  if (atBottom) log.scrollTop = log.scrollHeight;   // autoscroll only when already at the bottom
}

async function loadEventsFor(runId, reset) {
  if (reset) { logRunId = runId; logCursor = 0; $("#event-log").replaceChildren(); }
  if (logRunId !== runId) return;                    // user navigated away — don't clobber the view
  let body;
  try { body = await api.get(`/api/runs/${runId}/events?since=${logCursor}`); } catch (e) { return; }
  if (body.events && body.events.length) { appendEvents(body.events); logCursor = body.next; }
  $("#log-meta").textContent = logCursor ? `${logCursor} events` : "no events yet";
}

/* ── launch + poll ─────────────────────────────────────── */
function formatElapsed(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

function updateTimer(label = "elapsed") {
  const node = $("#run-timer");
  if (!node || runStartedAt == null) return;
  node.hidden = false;
  node.textContent = `${label} ${formatElapsed(Date.now() - runStartedAt)}`;
}

function startTimer() {
  if (timer) clearInterval(timer);
  runStartedAt = Date.now();
  updateTimer();
  timer = setInterval(() => updateTimer(), 1000);
}

function stopTimer(label = "elapsed") {
  if (timer) clearInterval(timer);
  timer = null;
  updateTimer(label);
}

async function launchRun(e) {
  e.preventDefault();
  const goal = $("#goal").value.trim();
  if (!goal) return;
  const btn = $("#run-btn"); const status = $("#run-status");
  btn.disabled = true;
  startTimer();
  status.hidden = false; status.className = "run-status";
  status.replaceChildren(spinner(), txt("queued…"));
  let runId;
  try {
    ({ run_id: runId } = await api.send("POST", "/api/runs", { goal, mode: $("#mode").value }));
  } catch (err) {
    status.className = "run-status is-error"; status.textContent = `Could not start: ${err.message}`;
    stopTimer("failed after");
    btn.disabled = false; return;
  }
  loadEventsFor(runId, true);            // fresh activity timeline for the new run
  if (poller) clearInterval(poller);
  poller = setInterval(() => pollStatus(runId, btn), 1500);
  pollStatus(runId, btn);
}

async function pollStatus(runId, btn) {
  loadEventsFor(runId, false);           // tail new events each poll (incremental)
  let s;
  try { s = await api.get(`/api/runs/${runId}/status`); } catch (e) { return; }
  const status = $("#run-status");
  if (s.status === "running" || s.status === "queued") {
    status.className = "run-status";
    status.replaceChildren(spinner(), txt(`${s.status}…`));
    if (s.n_hypotheses != null) {
      const live = el("div", "live");
      live.append(el("span", null, `${s.n_hypotheses} hyp`),
                  el("span", null, `${s.n_matches ?? 0} matches`),
                  el("span", null, `best Elo ${s.best_elo != null ? Math.round(s.best_elo) : "—"}`));
      status.appendChild(live);
    }
    return;
  }
  clearInterval(poller); poller = null; btn.disabled = false;
  if (s.status === "done") {
    stopTimer("completed in");
    status.innerHTML = "Run complete.";
    loadRuns().then(() => selectRun(runId));
  } else if (s.status === "aborted") {
    stopTimer("aborted after");
    status.className = "run-status is-error";
    status.textContent = "Aborted: the research goal did not pass the safety review.";
  } else {
    stopTimer("failed after");
    status.className = "run-status is-error";
    status.textContent = `Run failed: ${s.error || "unknown error"}`;
  }
}

/* ── config editor ─────────────────────────────────────── */
const SELECTS = { scheduler: ["continuous", "round_based"], backend: ["arxiv", "none", "tavily"], method: ["embeddings"] };

function fieldInput(path, key, value) {
  const wrap = el("label", "field");
  wrap.appendChild(el("span", "field-label", key));
  let input;
  if (SELECTS[key]) {
    input = el("select");
    SELECTS[key].forEach((o) => { const opt = el("option", null, o); opt.value = o; if (o === value) opt.selected = true; input.appendChild(opt); });
    input.dataset.type = "string";
  } else if (typeof value === "boolean") {
    input = el("select");
    ["true", "false"].forEach((o) => { const opt = el("option", null, o); opt.value = o; if (String(value) === o) opt.selected = true; input.appendChild(opt); });
    input.dataset.type = "bool";
  } else if (typeof value === "number" || value === null) {
    input = el("input"); input.type = "number"; input.step = "any";
    input.value = value == null ? "" : value;
    input.dataset.type = "number";
  } else {
    input = el("input"); input.type = "text"; input.value = value; input.dataset.type = "string";
  }
  input.dataset.path = path;
  wrap.appendChild(input);
  return wrap;
}

function renderConfigForm(cfg) {
  const root = $("#config-groups"); root.replaceChildren();
  const general = el("fieldset", "config-group");
  general.appendChild(el("legend", null, "general"));
  const gf = el("div", "config-fields"); general.appendChild(gf);
  let hasGeneral = false;

  for (const [key, value] of Object.entries(cfg)) {
    if (value !== null && typeof value === "object" && !Array.isArray(value)) {
      // models is free-form -> JSON textarea
      if (key === "models") {
        const fs = el("fieldset", "config-group");
        fs.appendChild(el("legend", null, "models (per-agent overrides)"));
        const ff = el("div", "config-fields");
        const lab = el("label", "field"); lab.style.gridColumn = "1 / -1";
        lab.appendChild(el("span", "field-label", "JSON object, e.g. {\"generation\": \"model/id\"}"));
        const ta = el("textarea"); ta.rows = 2; ta.value = JSON.stringify(value); ta.dataset.path = "models"; ta.dataset.type = "json";
        lab.appendChild(ta); ff.appendChild(lab); fs.appendChild(ff); root.appendChild(fs);
        continue;
      }
      const fs = el("fieldset", "config-group");
      fs.appendChild(el("legend", null, key));
      const ff = el("div", "config-fields");
      for (const [sk, sv] of Object.entries(value)) ff.appendChild(fieldInput(`${key}.${sk}`, sk, sv));
      fs.appendChild(ff); root.appendChild(fs);
    } else {
      gf.appendChild(fieldInput(key, key, value)); hasGeneral = true;
    }
  }
  if (hasGeneral) root.insertBefore(general, root.firstChild);
}

function setPath(obj, path, val) {
  const parts = path.split("."); let o = obj;
  for (let i = 0; i < parts.length - 1; i++) o = o[parts[i]];
  o[parts[parts.length - 1]] = val;
}

async function saveConfig(e) {
  e.preventDefault();
  const cfg = JSON.parse(JSON.stringify(loadedConfig));
  const msg = $("#config-msg"); msg.className = "config-msg";
  for (const input of document.querySelectorAll("#config-groups [data-path]")) {
    const t = input.dataset.type, raw = input.value;
    let val;
    if (t === "number") val = raw === "" ? null : Number(raw);
    else if (t === "bool") val = raw === "true";
    else if (t === "json") { try { val = JSON.parse(raw || "{}"); } catch { msg.className = "config-msg err"; msg.textContent = "models: invalid JSON"; return; } }
    else val = raw;
    setPath(cfg, input.dataset.path, val);
  }
  try {
    await api.send("PUT", "/api/config", cfg);
    loadedConfig = cfg;
    msg.className = "config-msg ok"; msg.textContent = "Saved.";
    renderGrounding(cfg, cfg.grounding.backend === "tavily");
  } catch (err) {
    msg.className = "config-msg err"; msg.textContent = err.message;
  }
}

/* ── init ──────────────────────────────────────────────── */
async function init() {
  $("#run-form").addEventListener("submit", launchRun);
  $("#view-config").addEventListener("submit", saveConfig);
  $("#nav-config").addEventListener("click", () => {
    if (!loadedConfig) return;
    renderConfigForm(loadedConfig); showView("config");
    document.querySelectorAll(".runs-list button").forEach((b) => b.classList.remove("active"));
    activeRunId = null;
  });
  try {
    const { config, faithful_grounding } = await api.get("/api/config");
    loadedConfig = config;
    renderGrounding(config, faithful_grounding);
  } catch (e) { $("#grounding-badge").textContent = "config unavailable"; }
  const runs = await loadRuns();
  if (runs && runs.length) selectRun(runs[0].id);
  else showView("welcome");
}

document.addEventListener("DOMContentLoaded", init);
