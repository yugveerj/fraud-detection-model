"use strict";

/* The Frozen Rule — page controller.
   One shared probability scale [0, SCALE_MAX]; the red rule sits at
   --p = threshold/SCALE_MAX in the hero, the scorer, and the net-value chart.
   Everything hydrates from committed, drift-guarded JSON (metrics.json,
   curves/*.json, presets.json); the only numeric literals are display
   constants (axis domains, CI-pinned) and last-committed fallbacks that
   hydration overwrites — a fetch failure is logged, never silent. */

const CONFIG = window.FRAUD_CONFIG || {};
const API_BASE = (CONFIG.API_BASE || "").replace(/\/$/, "");
const API_LIVE = API_BASE && !API_BASE.includes("__FRAUD_API_BASE__");
const SCORE_TIMEOUT_MS = 3000;
const SCALE_MAX = 0.3; // 97.5% of holdout scores fall below this; the rest pin right

if (CONFIG.GA_MEASUREMENT_ID) {
  const s = document.createElement("script");
  s.async = true;
  s.src = "https://www.googletagmanager.com/gtag/js?id=" + CONFIG.GA_MEASUREMENT_ID;
  document.head.appendChild(s);
  window.dataLayer = window.dataLayer || [];
  window.gtag = function () {
    window.dataLayer.push(arguments);
  };
  window.gtag("js", new Date());
  window.gtag("config", CONFIG.GA_MEASUREMENT_ID);
}

const el = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
const TOKENS = getComputedStyle(document.documentElement);
const COL = {
  ink: TOKENS.getPropertyValue("--ink").trim(),
  signal: TOKENS.getPropertyValue("--signal").trim(),
  grade: TOKENS.getPropertyValue("--grade").trim(),
};

let THRESHOLD = 0.1215;
let PRESETS = [];
let HIST = null;

/* ── boot ──────────────────────────────────────────────────────────────── */
async function boot() {
  loadMonitoring();

  const sources = [
    "metrics.json",
    "presets.json",
    "curves/net_value.json",
    "curves/leakage.json",
    "curves/reliability.json",
    "curves/score_histogram.json",
  ];
  const loaded = await Promise.all(sources.map(getJSON));
  loaded.forEach((v, i) => {
    if (v === null)
      logLine("> " + sources[i] + " unreachable — showing last-committed fallback text");
  });
  const [metrics, presets, nv, leak, rel, hist] = loaded;

  if (metrics) hydrateMetrics(metrics);
  renderAxes();
  if (hist) {
    HIST = hist;
    scheduleField();
  }
  if (presets) {
    PRESETS = presets.presets || [];
    renderRows();
    prescoreBorderline();
  }
  if (nv) chartNetValue(nv);
  if (leak) chartLeakage(leak);
  if (rel) chartReliability(rel);
}

function getJSON(path) {
  return fetch(path, { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null);
}

/* ── ledger strip ──────────────────────────────────────────────────────── */
async function loadMonitoring() {
  const state = await getJSON("monitoring/state.json");
  const target = el("lg-mon");
  if (!state || !Array.isArray(state.history) || !state.history.length) {
    target.textContent = "MONITORING · REPLAYED SIM";
    return;
  }
  const wk = state.history.length;
  const last = state.history[state.history.length - 1];
  const breaches = Array.isArray(last.breaches) ? last.breaches.length : last.breach ? 1 : 0;
  target.innerHTML =
    "MONITORING WK " + wk + " · " +
    (breaches > 0
      ? '<span class="breach">' + breaches + " BREACH" + (breaches === 1 ? "" : "ES") + "</span>"
      : "0 BREACHES") +
    " · REPLAYED SIM";
}

/* ── metrics → hero, rule, stats, conditions ───────────────────────────── */
function hydrateMetrics(m) {
  if (typeof m.threshold === "number") THRESHOLD = m.threshold;
  document.documentElement.style.setProperty("--p", (THRESHOLD / SCALE_MAX).toFixed(6));

  const pct = (THRESHOLD * 100).toFixed(1) + "%";
  el("hero-num").textContent = pct;
  el("hero-num").setAttribute("aria-label", "Review threshold: " + pct);
  el("rule-pct").textContent = pct;
  const lgm = el("lg-model");
  lgm.textContent =
    "MODEL v" + m.registry_version + " · " + String(m.calibration || "").toUpperCase() + " · SERVING";
  lgm.title =
    "isotonic calibration — predicted probabilities are corrected to match observed fraud rates";

  const hop = m.operating_point && m.operating_point.holdout;
  if (hop) {
    el("op-note").innerHTML =
      "<span>alert " + (hop.alert_rate * 100).toFixed(1) + "%</span> · <span>recall " +
      (hop.recall * 100).toFixed(1) + "%</span> · <span>FPR " + (hop.fpr * 100).toFixed(1) +
      "%</span>";
  }

  const GLOSS = [
    [/transactions analyzed/i, "the full public IEEE-CIS dataset, scored end to end"],
    [/inflation/i, 'how much a careless random split overstates this model — <a href="#ch-leak">see the experiment ↓</a>'],
    [/value captured/i, "of fraud dollars caught on unseen future data"],
    [/net value/i, "saved per 100k transactions, after paying for every manual review"],
  ];
  el("stats").innerHTML = (m.headline || [])
    .map((h) => {
      const g = GLOSS.find(([re]) => re.test(h.label || ""));
      return (
        '<li class="stat"><div class="stat-value">' + esc(h.value) +
        '</div><div class="stat-label">' + esc(h.label) + "</div>" +
        (g ? '<div class="stat-gloss">' + g[1] + "</div>" : "") + "</li>"
      );
    })
    .join("");
}

/* ── the shared axis (identical in hero, scorer, E1) ───────────────────── */
function renderAxes() {
  document.querySelectorAll("[data-axis]").forEach((ax) => {
    let html = "";
    [0, 0.1, 0.2].forEach((v) => {
      html +=
        '<span class="tick' + (v === 0 ? " t0" : "") + '" style="left:' +
        ((v / SCALE_MAX) * 100).toFixed(3) + '%">' + Math.round(v * 100) + "%</span>";
    });
    html += '<span class="terminus">≥ 30% →</span>';
    ax.innerHTML = html;
  });
}

/* ── the mark-field (real holdout histogram, drawn once) ───────────────── */
function scheduleField() {
  const run = () => drawField();
  if ("requestIdleCallback" in window) requestIdleCallback(run, { timeout: 800 });
  else setTimeout(run, 120);
  let deb;
  addEventListener("resize", () => {
    clearTimeout(deb);
    deb = setTimeout(drawField, 200);
  });
}

// deterministic jitter so the field renders identically on every load
function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let fieldPass = 0; // invalidates in-flight chunked paints on resize/redraw

function drawField() {
  if (!HIST) return;
  const cv = el("field");
  const wrap = cv.parentElement;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  if (!W || !H) return;
  const DPR = Math.min(2, devicePixelRatio || 1);
  cv.width = W * DPR;
  cv.height = H * DPR;
  const ctx = cv.getContext("2d");
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const rand = mulberry32(42);
  const n = HIST.n;
  const M = 9000; // marks drawn; caption states the per-mark weight
  const ruleX = (THRESHOLD / SCALE_MAX) * W;
  const pad = 6;

  // Precompute mark positions, split by decision side, so painting is two batched
  // fill passes (no per-mark style flips) chunked into short tasks (TBT budget).
  const clear = [], flag = [];
  for (const b of HIST.bins) {
    const marks = Math.round((b.count / n) * M);
    if (!marks) continue;
    const overflow = b.lo >= SCALE_MAX;
    for (let i = 0; i < marks; i++) {
      let x;
      if (overflow) {
        x = W - 4 - rand() * 6; // ≥30% terminus strip at the right edge
      } else {
        const lo = b.lo / SCALE_MAX, hi = Math.min(b.hi, SCALE_MAX) / SCALE_MAX;
        x = (lo + rand() * (hi - lo)) * W;
      }
      const y = pad + rand() * (H - 2 * pad);
      (x >= ruleX ? flag : clear).push(x, y);
    }
  }
  const drawn = (clear.length + flag.length) / 2;
  el("field-cap").textContent =
    "each mark ≈ " + Math.round(n / drawn) + " transactions · 64-bin histogram of the holdout " +
    "score distribution (n = " + n.toLocaleString() + ") · scores ≥ 30% pin right";

  const pass = ++fieldPass;
  const CHUNK = 2400; // coords per slice (1200 marks) — keeps each task well under 50ms
  const jobs = [
    { pts: clear, color: COL.ink, alpha: 0.3, s: 1.6 },
    { pts: flag, color: COL.signal, alpha: 0.55, s: 2.4 },
  ];
  let ji = 0, off = 0;
  const paint = () => {
    if (pass !== fieldPass) return; // superseded by a redraw
    const t0 = performance.now();
    while (ji < jobs.length && performance.now() - t0 < 12) {
      const j = jobs[ji];
      ctx.fillStyle = j.color;
      ctx.globalAlpha = j.alpha;
      const end = Math.min(j.pts.length, off + CHUNK);
      for (let k = off; k < end; k += 2) {
        ctx.fillRect(j.pts[k] - j.s / 2, j.pts[k + 1] - j.s / 2, j.s, j.s);
      }
      off = end;
      if (off >= j.pts.length) {
        ji++;
        off = 0;
      }
    }
    ctx.globalAlpha = 1;
    if (ji < jobs.length) setTimeout(paint, 0);
    else cv.classList.add("on");
  };
  paint();
}

/* ── scorer ────────────────────────────────────────────────────────────── */
function renderRows() {
  el("rows").innerHTML = "";
  PRESETS.forEach((p, i) => {
    const d = p.display || {};
    const row = document.createElement("div");
    row.className = "row";
    const facts = [d.ProductCD && "product " + d.ProductCD, d.card4, d.card6, d.P_emaildomain]
      .filter(Boolean)
      .join(" · ");
    row.innerHTML =
      '<span class="row-amt">$' + fmtNum(d.TransactionAmt) + "</span>" +
      '<span class="row-desc"><b>' + esc(p.label || p.name) + "</b>" + esc(facts) + "</span>" +
      '<button type="button" data-i="' + i + '" aria-label="Score: ' +
      esc(p.label || p.name) + '">SCORE</button>';
    row.querySelector("button").addEventListener("click", () => scorePreset(i));
    el("rows").appendChild(row);
  });
}

function prescoreBorderline() {
  const i = PRESETS.findIndex((p) => p.name === "borderline");
  if (i < 0 || !PRESETS[i].expected) return;
  logLine("> pre-scored at load · " + PRESETS[i].name + " · cached result");
  applyResult(PRESETS[i], PRESETS[i].expected, false);
}

const PENDING = new Set();
let scoreSeq = 0;

async function scorePreset(i) {
  if (PENDING.has(i)) return; // in-flight guard — keeps focus on the button (no disabled)
  PENDING.add(i);
  const preset = PRESETS[i];
  const btn = document.querySelector('.row button[data-i="' + i + '"]');
  if (btn) btn.setAttribute("aria-disabled", "true");
  const seq = ++scoreSeq;
  const t0 = performance.now();
  const { result, live, reason } = await scoreTransaction(preset);
  const ms = Math.round(performance.now() - t0);
  PENDING.delete(i);
  if (btn) btn.removeAttribute("aria-disabled");
  logLine(
    live
      ? "> POST /score · " + preset.name + " · " + ms + "ms · live"
      : API_LIVE
        ? "> POST /score · " + preset.name + " · " + (reason || "unreachable") + " → cached result"
        : "> offline build · " + preset.name + " → cached result"
  );
  if (seq !== scoreSeq) return; // a newer score already owns the readout
  applyResult(preset, result, live);
}

async function scoreTransaction(preset) {
  if (API_LIVE) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), SCORE_TIMEOUT_MS);
    try {
      const res = await fetch(API_BASE + "/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(preset.payload),
        signal: ctrl.signal,
      });
      clearTimeout(timer);
      if (res.ok) return { result: await res.json(), live: true };
      return { result: preset.expected, live: false, reason: "http " + res.status };
    } catch (e) {
      clearTimeout(timer);
      return {
        result: preset.expected,
        live: false,
        reason: ctrl.signal.aborted ? "timeout 3.0s" : "unreachable",
      };
    }
  }
  return { result: preset.expected, live: false };
}

function logLine(text) {
  const log = el("netlog");
  const lines = (log.textContent ? log.textContent.split("\n") : []).concat(text);
  log.textContent = lines.slice(-5).join("\n");
}

function applyResult(preset, r, live) {
  const p = Number(r.fraud_probability);
  const review = r.decision ? r.decision === "review" : p >= THRESHOLD;
  placeMarker(preset.name, p, review);

  el("verdict-wrap").hidden = false;
  const v = el("verdict");
  v.textContent = review ? "Review" : "Approve";
  v.className = "verdict" + (review ? " review" : "");
  const margin = Math.abs(p - THRESHOLD) * 100;
  el("verdict-line").innerHTML =
    "<b>" + (review ? "Review" : "Approve") + "</b> — " +
    esc(preset.label || preset.name) + " — calibrated P(fraud) <b>" + fmtPct(p) + "</b> · <b>" +
    margin.toFixed(1) + " pts " + (p >= THRESHOLD ? "over" : "under") + "</b> the " +
    (THRESHOLD * 100).toFixed(1) + "% threshold · " + (live ? "scored live" : "cached result");

  renderShap(r.top_factors || []);
}

function placeMarker(name, p, review) {
  const plot = el("scorer-plot");
  let m = plot.querySelector('[data-name="' + name + '"]');
  const frac = clamp(p / SCALE_MAX, 0, 1);
  const pinned = p > SCALE_MAX;
  const label = fmtPct(p) + (pinned ? " →" : "");
  if (!m) {
    m = document.createElement("div");
    m.dataset.name = name;
    m.innerHTML =
      '<span class="marker-tag"><span class="marker-p"></span>' +
      '<span class="marker-name"></span></span>';
    plot.appendChild(m);
  }
  m.className = "marker" + (review ? " review" : "") + (pinned ? " pinned" : "");
  m.style.left = (frac * 100).toFixed(3) + "%";
  m.querySelector(".marker-p").textContent = label;
  m.querySelector(".marker-name").textContent = name;
}

function renderShap(factors) {
  const wrap = el("factors");
  const list = el("shap");
  wrap.hidden = false;
  list.innerHTML = "";
  const top = factors.slice(0, 6);
  const maxAbs = Math.max(0.001, ...top.map((f) => Math.abs(Number(f.shap))));
  top.forEach((f, i) => {
    const shap = Number(f.shap);
    const w = (Math.abs(shap) / maxAbs) * 48;
    const dir = shap >= 0 ? "pos" : "neg";
    const li = document.createElement("li");
    li.className = "shap-row";
    li.innerHTML =
      '<span class="shap-feat" title="' + esc(f.feature) + " = " + fmtNum(f.value) + '">' +
      esc(f.feature) + "</span>" +
      '<span class="shap-track"><span class="shap-fill ' + dir + '"></span></span>' +
      '<span class="shap-val">' + (shap >= 0 ? "+" : "−") + Math.abs(shap).toFixed(2) + "</span>";
    list.appendChild(li);
    const fill = li.querySelector(".shap-fill");
    if (REDUCE) fill.style.width = w + "%";
    else setTimeout(() => (fill.style.width = w + "%"), 30 + i * 40);
  });
}

/* ── E1: net value vs threshold (shares the scale + the rule) ──────────── */
const SVGNS = "http://www.w3.org/2000/svg";
function svgTag(tag, attrs) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
const money0 = (v) =>
  (v < 0 ? "−$" : "$") + Math.abs(Math.round(v)).toLocaleString();
const moneyK = (v) => {
  const a = Math.abs(v), sign = v < 0 ? "−$" : "$";
  if (a >= 1000000) {
    const m = a / 1000000;
    return sign + (Number.isInteger(m) ? m : m.toFixed(m < 10 ? 1 : 0)) + "M";
  }
  return sign + Math.round(a / 1000).toLocaleString() + "k";
};

function chartNetValue(d) {
  const plot = el("ch-net-plot");
  const beyond = d.series.findIndex((q) => q.t > SCALE_MAX);
  const pts = beyond < 0 ? d.series : d.series.slice(0, beyond + 1); // svg clips at the edge
  const yMin = Math.min(...d.series.map((q) => q.net));
  const yMax = 320000;
  const lo = Math.floor(yMin / 100000) * 100000;
  const yPct = (v) => ((yMax - v) / (yMax - lo)) * 100;

  const svg = svgTag("svg", { viewBox: "0 0 1000 300", preserveAspectRatio: "none", role: "img" });
  const title = document.createElementNS(SVGNS, "title");
  title.textContent =
    "Net value per 100k transactions at every threshold on validation; frozen at " +
    (d.op.t * 100).toFixed(1) + "% where validation returns " + money0(d.op.net_val) +
    " and holdout returns " + money0(d.op.net_holdout) + ".";
  svg.appendChild(title);

  const ticks = [250000, 0, -500000, -1000000, -1500000].filter((v) => v > lo);
  ticks.forEach((v) => {
    const y = (yPct(v) / 100) * 300;
    svg.appendChild(svgTag("line", {
      class: v === 0 ? "c-zero" : "c-grid",
      x1: 0, y1: y, x2: 1000, y2: y,
    }));
  });
  const path = pts
    .map((q, i) =>
      (i ? "L" : "M") + ((q.t / SCALE_MAX) * 1000).toFixed(2) + " " + ((yPct(q.net) / 100) * 300).toFixed(2)
    )
    .join(" ");
  svg.appendChild(svgTag("path", { class: "c-line", d: path }));
  plot.appendChild(svg);

  // y labels (HTML, in the shared gutter)
  el("ch-net-y").innerHTML = ticks
    .map((v) => '<span class="ylab" style="top:' + yPct(v).toFixed(2) + '%">' + moneyK(v) + "</span>")
    .join("");

  // the two truths at the frozen threshold (HTML overlays on the shared scale)
  const dotVal = document.createElement("span");
  dotVal.className = "nv-dot val";
  dotVal.style.cssText = "left:calc(var(--p) * 100%);top:" + yPct(d.op.net_val).toFixed(2) + "%";
  const dotHold = document.createElement("span");
  dotHold.className = "nv-dot holdout";
  dotHold.style.cssText = "left:calc(var(--p) * 100%);top:" + yPct(d.op.net_holdout).toFixed(2) + "%";
  const note = document.createElement("div");
  note.className = "nv-note";
  note.style.cssText =
    "left:calc(var(--p) * 100% + 14px);top:" + (yPct(d.op.net_holdout) + 3).toFixed(2) + "%";
  note.innerHTML =
    "<b>" + money0(d.op.net_val) + "</b> validation — frozen here<br>" +
    '<span class="holdout"><b>' + money0(d.op.net_holdout) + "</b> holdout, same threshold</span><br>" +
    '<span class="holdout">the out-of-time gap, reported</span>';
  plot.append(dotVal, dotHold, note);

  el("ch-net-sub").textContent =
    "Net dollars per 100k transactions at every threshold, on the validation set — where the " +
    "line was frozen. The plunge to " + moneyK(yMin) + " at 0% is the cost of no threshold at " +
    "all; the held-out future returns " + money0(d.op.net_holdout) + " at the same frozen line.";
  const tail = d.series[d.series.length - 1];
  el("ch-net-foot").textContent =
    "basis: validation" + (d.n ? " (n=" + Number(d.n).toLocaleString() + ")" : "") +
    " · holdout points at the frozen threshold · x clipped at 30% " +
    "(curve continues to " + Math.round(tail.t * 100) + "%: " + moneyK(tail.net) + ") · drift-guarded JSON";
  el("cond-gap").textContent = money0(d.op.net_val) + " → " + money0(d.op.net_holdout);

  table(
    el("tbl-net"),
    ["threshold", "net / 100k (validation)"],
    d.series.filter((_, i) => i % 6 === 0).map((q) => [(q.t * 100).toFixed(1) + "%", money0(q.net)])
      .concat([["frozen " + (d.op.t * 100).toFixed(1) + "%", money0(d.op.net_val) + " val · " + money0(d.op.net_holdout) + " holdout"]])
  );
}

/* ── E2: leakage dumbbells ─────────────────────────────────────────────── */
function chartLeakage(d) {
  const W = 560, H = 250, mL = 96, mR = 30, top = 52, rowH = 56;
  const x = (v) => mL + ((v - 0.4) / 0.4) * (W - mL - mR);
  const infl = d.rows.map((r) => r.inflation).filter((v) => v > 0);
  const lo = Math.round(Math.min(...infl)), hi = Math.round(Math.max(...infl));
  const range = "+" + lo + "–" + hi + "%";
  if (el("leak-range")) el("leak-range").textContent = range;
  const svg = svgTag("svg", { viewBox: "0 0 " + W + " " + H, role: "img" });
  const title = document.createElementNS(SVGNS, "title");
  title.textContent =
    "Validation PR-AUC per model: honest temporal split versus random cross-validation, " +
    "which inflates the tree models by " + lo + " to " + hi + " percent.";
  svg.appendChild(title);

  [0.4, 0.5, 0.6, 0.7, 0.8].forEach((v) => {
    svg.appendChild(svgTag("line", { class: "c-grid", x1: x(v), y1: 34, x2: x(v), y2: H - 26 }));
    const t = svgTag("text", { class: "c-text", x: x(v), y: H - 10, "text-anchor": "middle" });
    t.textContent = v.toFixed(1);
    svg.appendChild(t);
  });

  d.rows.forEach((r, i) => {
    const y = top + i * rowH;
    const name = svgTag("text", { class: "c-text-ink", x: 8, y: y + 4 });
    name.textContent = r.model;
    svg.appendChild(name);
    svg.appendChild(svgTag("line", { class: "c-dumb", x1: x(r.temporal), y1: y, x2: x(r.random_cv), y2: y }));
    svg.appendChild(svgTag("circle", { class: "c-dot-ink", cx: x(r.temporal), cy: y, r: 5.5 }));
    svg.appendChild(svgTag("circle", { class: "c-dot-flag", cx: x(r.random_cv), cy: y, r: 5.5 }));
    const lab = svgTag("text", {
      class: r.inflation > 0 ? "c-text-flag" : "c-text",
      x: x(Math.max(r.temporal, r.random_cv)) + 10,
      y: y - 12,
      "text-anchor": "end",
    });
    lab.textContent = (r.inflation >= 0 ? "+" : "−") + Math.abs(r.inflation).toFixed(1) + "%";
    svg.appendChild(lab);
  });

  const lg1 = svgTag("circle", { class: "c-dot-ink", cx: 13, cy: 16, r: 5 });
  const lt1 = svgTag("text", { class: "c-text", x: 23, y: 20 });
  lt1.textContent = "temporal (honest)";
  const lg2 = svgTag("circle", { class: "c-dot-flag", cx: 181, cy: 16, r: 5 });
  const lt2 = svgTag("text", { class: "c-text", x: 191, y: 20 });
  lt2.textContent = "random-CV (anti-pattern)";
  svg.append(lg1, lt1, lg2, lt2);

  el("ch-leak-plot").appendChild(svg);
  table(
    el("tbl-leak"),
    ["model", "temporal", "random-CV", "inflation"],
    d.rows.map((r) => [
      r.model, r.temporal.toFixed(3), r.random_cv.toFixed(3),
      (r.inflation >= 0 ? "+" : "−") + Math.abs(r.inflation).toFixed(1) + "%",
    ])
  );
}

/* ── E3: reliability ───────────────────────────────────────────────────── */
function chartReliability(d) {
  const W = 560, H = 330, mL = 56, mR = 16, mT = 14, mB = 40;
  const DOM = 0.62;
  const x = (v) => mL + (clamp(v, 0, DOM) / DOM) * (W - mL - mR);
  const y = (v) => H - mB - (clamp(v, 0, DOM) / DOM) * (H - mT - mB);
  const svg = svgTag("svg", { viewBox: "0 0 " + W + " " + H, role: "img" });
  const title = document.createElementNS(SVGNS, "title");
  title.textContent =
    "Reliability: the uncalibrated model predicts higher probabilities than observed; the " +
    "isotonic-calibrated model sits on the diagonal.";
  svg.appendChild(title);

  [0, 0.2, 0.4, 0.6].forEach((v) => {
    svg.appendChild(svgTag("line", { class: "c-grid", x1: x(v), y1: y(0), x2: x(v), y2: mT }));
    svg.appendChild(svgTag("line", { class: "c-grid", x1: mL, y1: y(v), x2: W - mR, y2: y(v) }));
    const tx = svgTag("text", { class: "c-text", x: x(v), y: H - 22, "text-anchor": "middle" });
    tx.textContent = v.toFixed(1);
    const ty = svgTag("text", { class: "c-text", x: mL - 8, y: y(v) + 4, "text-anchor": "end" });
    ty.textContent = v.toFixed(1);
    svg.append(tx, ty);
  });
  svg.appendChild(svgTag("line", { class: "c-ref", x1: x(0), y1: y(0), x2: x(DOM), y2: y(DOM) }));

  const series = (arr, lineCls, dotCls) => {
    const path = arr
      .map((q, i) => (i ? "L" : "M") + x(q.pred).toFixed(1) + " " + y(q.emp).toFixed(1))
      .join(" ");
    svg.appendChild(svgTag("path", { class: lineCls, d: path }));
    arr.forEach((q) =>
      svg.appendChild(svgTag("circle", { class: dotCls, cx: x(q.pred), cy: y(q.emp), r: 4 }))
    );
  };
  series(d.uncalibrated, "c-line-flag", "c-dot-flag");
  series(d.isotonic, "c-line", "c-dot-ink");

  const u = d.uncalibrated[d.uncalibrated.length - 1];
  const ann = svgTag("text", {
    class: "c-text-flag", x: x(u.pred) - 10, y: y(u.emp) - 12, "text-anchor": "end",
  });
  ann.textContent = u.pred.toFixed(2) + " predicted → " + u.emp.toFixed(2) + " observed";
  svg.appendChild(ann);

  const xl = svgTag("text", { class: "c-text", x: (mL + W - mR) / 2, y: H - 4, "text-anchor": "middle" });
  xl.textContent = "predicted probability → observed fraud rate";
  svg.appendChild(xl);

  el("ch-rel-plot").appendChild(svg);
  table(
    el("tbl-rel"),
    ["predicted (uncal)", "observed", "predicted (isotonic)", "observed"],
    d.uncalibrated.map((q, i) => [
      q.pred.toFixed(3), q.emp.toFixed(3),
      d.isotonic[i] ? d.isotonic[i].pred.toFixed(3) : "—",
      d.isotonic[i] ? d.isotonic[i].emp.toFixed(3) : "—",
    ])
  );
}

/* ── helpers ───────────────────────────────────────────────────────────── */
function table(host, headers, rows) {
  const h = "<tr>" + headers.map((x) => "<th>" + esc(x) + "</th>").join("") + "</tr>";
  const b = rows
    .map((r) => "<tr>" + r.map((c) => "<td>" + esc(String(c)) + "</td>").join("") + "</tr>")
    .join("");
  host.innerHTML = "<table>" + h + b + "</table>";
}
function fmtPct(p) {
  const v = Number(p) * 100;
  if (v >= 99.95) return "100%";
  if (v < 0.1) return v.toFixed(3) + "%";
  return v.toFixed(1) + "%";
}
function fmtNum(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  return Number.isInteger(n) ? n.toLocaleString() : n.toFixed(2);
}
function esc(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );
}

boot();
