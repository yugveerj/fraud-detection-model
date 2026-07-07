"use strict";

/* The Examination Record — demo controller.
   Masthead + KPIs + frozen operating point from metrics.json; the interactive decision
   (hybrid scoring → control-limit gauge → stamp → SHAP); and three vanilla-SVG charts
   (net value, leakage, reliability) drawn from web/curves/*.json. No charting library. */

const CONFIG = window.FRAUD_CONFIG || {};
const API_BASE = (CONFIG.API_BASE || "").replace(/\/$/, "");
const API_LIVE = API_BASE && !API_BASE.includes("__FRAUD_API_BASE__");
const SCORE_TIMEOUT_MS = 3000;

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

let PRESETS = [];
let THRESHOLD = 0.1215;
let firstScore = true;

async function boot() {
  await loadMetrics();
  await loadPresets();
  loadCharts();
}

/* ── metrics → masthead, KPIs, frozen record ───────────────────────────── */
async function loadMetrics() {
  let m;
  try {
    m = await (await fetch("metrics.json", { cache: "no-store" })).json();
  } catch (e) {
    el("stats").innerHTML = '<li class="kpi"><div class="kpi-label">Metrics unavailable.</div></li>';
    return;
  }
  if (typeof m.threshold === "number") THRESHOLD = m.threshold;
  positionLimit();

  el("fields").innerHTML = [
    ["Model", `${m.calibration}-calibrated ${m.model}`],
    ["Registry", `v${m.registry_version}`],
    ["Data", `${m.provenance} IEEE-CIS`],
    ["Review cost", `$${Number(m.review_cost).toFixed(0)} / alert`],
  ]
    .map(([k, v]) => `<div><dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd></div>`)
    .join("");

  el("stats").innerHTML = (m.headline || [])
    .map(
      (h) =>
        `<li class="kpi"><div class="kpi-value">${withAccentUnit(h.value)}</div>` +
        `<div class="kpi-label">${escapeHtml(h.label)}</div></li>`
    )
    .join("");

  const hop = m.operating_point && m.operating_point.holdout;
  if (hop) {
    const cells = [
      ["Threshold", (THRESHOLD * 100).toFixed(1) + "%"],
      ["Alert rate", (hop.alert_rate * 100).toFixed(1) + "%"],
      ["Recall", (hop.recall * 100).toFixed(1) + "%"],
      ["False-positive", (hop.fpr * 100).toFixed(1) + "%"],
      ["Value captured", (hop.value_capture_rate * 100).toFixed(1) + "%"],
      ["Net / 100k", "$" + Math.round(hop.net_per_100k / 1000) + "k"],
    ];
    el("record").innerHTML =
      '<div class="record-head">Frozen operating point</div>' +
      '<dl class="record-grid">' +
      cells.map(([k, v]) => `<div><dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd></div>`).join("") +
      "</dl>" +
      '<p class="record-caveat">Frozen on validation · reported on holdout · labels lag weeks in production</p>';
  }
}

function withAccentUnit(value) {
  return escapeHtml(value).replace(/([%$+≈K,]+)/g, '<span class="u">$1</span>');
}

/* ── presets → chips ───────────────────────────────────────────────────── */
async function loadPresets() {
  try {
    const data = await (await fetch("presets.json", { cache: "no-store" })).json();
    PRESETS = data.presets || [];
  } catch (e) {
    el("chips").innerHTML = '<p class="caveat">Could not load example transactions.</p>';
    return;
  }
  el("chips").innerHTML = "";
  PRESETS.forEach((p, i) => {
    const amt = p.display && p.display.TransactionAmt;
    const btn = document.createElement("button");
    btn.className = "chip";
    btn.type = "button";
    btn.setAttribute("aria-pressed", "false");
    btn.innerHTML =
      `${escapeHtml(p.label || p.name)}` +
      (amt != null ? ` <span class="chip-amt">$${fmtNum(amt)}</span>` : "");
    btn.addEventListener("click", () => selectPreset(i));
    el("chips").appendChild(btn);
  });
}

/* ── scoring (hybrid: live API with timeout → precomputed fallback) ─────── */
async function selectPreset(i) {
  el("chips")
    .querySelectorAll(".chip")
    .forEach((c, j) => c.setAttribute("aria-pressed", String(j === i)));
  el("gauge-cap").textContent = "scoring…";
  const { result, live } = await scoreTransaction(PRESETS[i]);
  renderResult(result, live);
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
    } catch (e) {
      clearTimeout(timer);
    }
  }
  return { result: preset.expected, live: false };
}

function positionLimit() {
  const x = clamp(THRESHOLD * 100, 0, 100);
  el("gauge-limit").style.left = x + "%";
  el("gauge-flag").style.left = x + "%";
  const tag = el("gauge-limit-tag");
  if (tag) tag.textContent = `review threshold ${(THRESHOLD * 100).toFixed(1)}%`;
}

function renderResult(r, live) {
  const p = Number(r.fraud_probability);
  const isFlag = r.decision ? r.decision === "review" : p >= THRESHOLD;
  const target = clamp(p * 100, 0.6, 99.4);

  const marker = el("gauge-marker");
  marker.hidden = false;
  marker.className = "gauge-marker " + (isFlag ? "is-flag" : "is-clear");
  el("gauge-marker-tag").textContent = fmtPct(p);
  if (firstScore) {
    marker.style.left = "0%";
    requestAnimationFrame(() => requestAnimationFrame(() => (marker.style.left = target + "%")));
    firstScore = false;
  } else {
    marker.style.left = target + "%";
  }
  el("gauge-cap").textContent = isFlag
    ? "crosses the review threshold → flagged for review"
    : "under the review threshold → cleared";
  el("gauge-scale").setAttribute("role", "img");
  el("gauge-scale").setAttribute(
    "aria-label",
    `Calibrated fraud probability ${fmtPct(p)}, review threshold ${(THRESHOLD * 100).toFixed(1)}%, ` +
      `decision ${isFlag ? "review" : "approve"}.`
  );

  el("readout").hidden = false;
  const stamp = el("verdict-stamp");
  stamp.textContent = isFlag ? "Review" : "Approve";
  stamp.className = "stamp " + (isFlag ? "review" : "approve");
  el("verdict-line").innerHTML =
    `calibrated P(fraud) <b>${fmtPct(p)}</b> against the review threshold ` +
    `<b>${(THRESHOLD * 100).toFixed(1)}%</b>`;

  renderBars(r.top_factors || []);

  const parts = [];
  if (r.model_version) parts.push("model v" + r.model_version);
  if (r.calibration) parts.push(r.calibration + " calibration");
  if (r.provenance) parts.push(r.provenance + " data");
  parts.push(live ? "scored live" : "cached response");
  el("model-meta").textContent = parts.join("  ·  ");
}

function renderBars(factors) {
  const bars = el("bars");
  bars.innerHTML = "";
  const top = factors.slice(0, 6);
  const maxAbs = Math.max(0.001, ...top.map((f) => Math.abs(Number(f.shap))));
  top.forEach((f) => {
    const shap = Number(f.shap);
    const w = (Math.abs(shap) / maxAbs) * 48;
    const dir = shap >= 0 ? "pos" : "neg";
    const li = document.createElement("li");
    li.className = "bar-row";
    li.innerHTML =
      `<span class="bar-feat" title="${escapeHtml(f.feature)} = ${fmtNum(f.value)}">` +
      `${escapeHtml(f.feature)}</span>` +
      `<span class="bar-track"><span class="bar-fill ${dir}" style="width:${w}%"></span></span>` +
      `<span class="bar-val ${dir}">${shap >= 0 ? "+" : "−"}${Math.abs(shap).toFixed(2)}</span>`;
    bars.appendChild(li);
  });
}

/* ── vanilla-SVG chart engine ──────────────────────────────────────────── */
const SVGNS = "http://www.w3.org/2000/svg";
function svgTag(tag, attrs, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  return e;
}
function svgRoot(w, h, title, desc) {
  const s = svgTag("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" });
  s.appendChild(svgTag("title", {}, title));
  s.appendChild(svgTag("desc", {}, desc));
  return s;
}
const linScale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) * (r1 - r0)) / (d1 - d0);
const money = (v) => (v < 0 ? "-$" : "$") + Math.round(Math.abs(v) / 1000) + "k";

async function loadCharts() {
  try {
    const [nv, lk, rel] = await Promise.all(
      ["net_value", "leakage", "reliability"].map((n) =>
        fetch(`curves/${n}.json`, { cache: "no-store" }).then((r) => r.json())
      )
    );
    chartNetValue(nv);
    chartLeakage(lk);
    chartReliability(rel);
  } catch (e) {
    /* charts are enhancement; page still works without them */
  }
}

function table(host, headers, rows) {
  const h = "<tr>" + headers.map((x) => `<th>${escapeHtml(x)}</th>`).join("") + "</tr>";
  const b = rows
    .map((r) => "<tr>" + r.map((c) => `<td>${escapeHtml(String(c))}</td>`).join("") + "</tr>")
    .join("");
  host.innerHTML = `<table>${h}${b}</table>`;
}

function chartNetValue(d) {
  const W = 680, H = 300, m = { l: 58, r: 70, t: 20, b: 34 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xd = [0, 0.5], yd = [-100000, 300000];
  const x = linScale(xd[0], xd[1], m.l, m.l + iw);
  const y = linScale(yd[0], yd[1], m.t + ih, m.t);
  const opV = d.op.net_val, opH = d.op.net_holdout, opT = d.op.t;
  const s = svgRoot(
    W, H,
    "Net value versus decision threshold",
    `Net dollars per 100k transactions across thresholds on the validation set; peaks at ${money(opV)} at the frozen threshold ${(opT * 100).toFixed(1)}%, which returns ${money(opH)} on holdout.`
  );
  const defs = svgTag("defs", {});
  const clip = svgTag("clipPath", { id: "clip-nv" });
  clip.appendChild(svgTag("rect", { x: m.l, y: m.t, width: iw, height: ih }));
  defs.appendChild(clip);
  s.appendChild(defs);

  [-100000, 0, 100000, 200000, 300000].forEach((v) => {
    const yy = y(v);
    s.appendChild(svgTag("line", { class: "c-grid", x1: m.l, y1: yy, x2: m.l + iw, y2: yy }));
    s.appendChild(svgTag("text", { class: "c-axis-text", "text-anchor": "end", x: m.l - 8, y: yy + 4 }, money(v)));
  });
  [0, 0.1, 0.2, 0.3, 0.4, 0.5].forEach((v) => {
    s.appendChild(svgTag("text", { class: "c-axis-text", "text-anchor": "middle", x: x(v), y: m.t + ih + 20 }, (v * 100).toFixed(0) + "%"));
  });
  s.appendChild(svgTag("line", { class: "c-axis", x1: m.l, y1: m.t + ih, x2: m.l + iw, y2: m.t + ih }));

  const pts = d.series.filter((p) => p.t <= xd[1] + 0.01);
  const dPath = pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)} ${y(p.net).toFixed(1)}`).join(" ");
  s.appendChild(svgTag("path", { class: "c-line", d: dPath, "clip-path": "url(#clip-nv)" }));

  // frozen threshold marker + the val peak and holdout points
  s.appendChild(svgTag("line", { class: "c-marker", x1: x(opT), y1: m.t, x2: x(opT), y2: m.t + ih }));
  s.appendChild(svgTag("circle", { class: "c-dot", cx: x(opT), cy: y(opV), r: 4 }));
  s.appendChild(svgTag("text", { class: "c-label-strong", "text-anchor": "start", x: x(opT) + 8, y: y(opV) - 4 }, `${money(opV)} val`));
  s.appendChild(svgTag("circle", { class: "c-dot-flag", cx: x(opT), cy: y(opH), r: 4 }));
  s.appendChild(svgTag("text", { class: "c-label-flag", "text-anchor": "start", x: x(opT) + 8, y: y(opH) + 14 }, `${money(opH)} holdout`));

  el("svg-netvalue").replaceChildren(s);
  table(
    el("tbl-netvalue"),
    ["threshold", "net / 100k (val)"],
    d.series.filter((_, i) => i % 6 === 0).map((p) => [(p.t * 100).toFixed(1) + "%", money(p.net)])
      .concat([["→ frozen " + (opT * 100).toFixed(1) + "%", money(opV) + " val · " + money(opH) + " holdout"]])
  );
}

function chartLeakage(d) {
  const rows = d.rows;
  const W = 680, H = 300, m = { l: 44, r: 20, t: 24, b: 46 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const yd = [0, 0.8];
  const y = linScale(yd[0], yd[1], m.t + ih, m.t);
  const s = svgRoot(
    W, H,
    "Validation PR-AUC: temporal split versus random cross-validation",
    "Random cross-validation inflates PR-AUC over the honest temporal split for the tree models by 17 to 23 percent; the linear model barely moves."
  );
  [0, 0.2, 0.4, 0.6, 0.8].forEach((v) => {
    const yy = y(v);
    s.appendChild(svgTag("line", { class: "c-grid", x1: m.l, y1: yy, x2: m.l + iw, y2: yy }));
    s.appendChild(svgTag("text", { class: "c-axis-text", "text-anchor": "end", x: m.l - 8, y: yy + 4 }, v.toFixed(1)));
  });
  s.appendChild(svgTag("line", { class: "c-axis", x1: m.l, y1: m.t + ih, x2: m.l + iw, y2: m.t + ih }));

  const gw = iw / rows.length, bw = Math.min(46, gw / 2.6);
  rows.forEach((r, i) => {
    const cx = m.l + gw * (i + 0.5);
    const bars = [
      { v: r.temporal, cls: "c-bar-temporal", dx: -bw - 3 },
      { v: r.random_cv, cls: "c-bar-random", dx: 3 },
    ];
    bars.forEach((b) => {
      const bh = m.t + ih - y(b.v);
      s.appendChild(svgTag("rect", { class: b.cls, x: cx + b.dx, y: y(b.v), width: bw, height: bh }));
    });
    s.appendChild(svgTag("text", { class: "c-axis-text", "text-anchor": "middle", x: cx, y: m.t + ih + 18 }, r.model));
    const infl = (r.inflation >= 0 ? "+" : "") + r.inflation.toFixed(1) + "%";
    s.appendChild(svgTag("text", { class: r.inflation > 0 ? "c-label-flag" : "c-label", "text-anchor": "middle", x: cx, y: y(Math.max(r.temporal, r.random_cv)) - 7 }, infl));
  });
  // legend
  s.appendChild(svgTag("rect", { class: "c-bar-temporal", x: m.l, y: m.t + ih + 30, width: 11, height: 11 }));
  s.appendChild(svgTag("text", { class: "c-label", x: m.l + 16, y: m.t + ih + 40 }, "temporal (honest)"));
  s.appendChild(svgTag("rect", { class: "c-bar-random", x: m.l + 150, y: m.t + ih + 30, width: 11, height: 11 }));
  s.appendChild(svgTag("text", { class: "c-label-flag", x: m.l + 166, y: m.t + ih + 40 }, "random-CV (anti-pattern)"));

  el("svg-leakage").replaceChildren(s);
  table(
    el("tbl-leakage"),
    ["model", "temporal", "random-CV", "inflation"],
    rows.map((r) => [r.model, r.temporal.toFixed(3), r.random_cv.toFixed(3), (r.inflation >= 0 ? "+" : "") + r.inflation.toFixed(1) + "%"])
  );
}

function chartReliability(d) {
  const W = 680, H = 320, m = { l: 48, r: 20, t: 20, b: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const dom = [0, 0.6];
  const x = linScale(dom[0], dom[1], m.l, m.l + iw);
  const y = linScale(dom[0], dom[1], m.t + ih, m.t);
  const s = svgRoot(
    W, H,
    "Calibration reliability: predicted probability versus observed fraud rate",
    "The uncalibrated model sits below the diagonal (over-confident); isotonic calibration pulls it onto the diagonal so a stated probability matches the observed rate."
  );
  [0, 0.15, 0.3, 0.45, 0.6].forEach((v) => {
    s.appendChild(svgTag("line", { class: "c-grid", x1: m.l, y1: y(v), x2: m.l + iw, y2: y(v) }));
    s.appendChild(svgTag("text", { class: "c-axis-text", "text-anchor": "end", x: m.l - 8, y: y(v) + 4 }, v.toFixed(2)));
    s.appendChild(svgTag("text", { class: "c-axis-text", "text-anchor": "middle", x: x(v), y: m.t + ih + 18 }, v.toFixed(2)));
  });
  s.appendChild(svgTag("line", { class: "c-axis", x1: m.l, y1: m.t + ih, x2: m.l + iw, y2: m.t + ih }));
  s.appendChild(svgTag("line", { class: "c-ref", x1: x(0), y1: y(0), x2: x(0.6), y2: y(0.6) }));
  s.appendChild(svgTag("text", { class: "c-axis-text", "text-anchor": "middle", x: x(0.5), y: y(0.55) }, "perfect"));

  const series = (arr, cls, dotCls) => {
    const path = arr.map((p, i) => `${i ? "L" : "M"}${x(clamp(p.pred, 0, 0.6)).toFixed(1)} ${y(clamp(p.emp, 0, 0.6)).toFixed(1)}`).join(" ");
    s.appendChild(svgTag("path", { class: cls, d: path }));
    arr.forEach((p) => s.appendChild(svgTag("circle", { class: dotCls, cx: x(clamp(p.pred, 0, 0.6)), cy: y(clamp(p.emp, 0, 0.6)), r: 3 })));
  };
  series(d.uncalibrated, "c-line-2", "c-dot-flag");
  series(d.isotonic, "c-line", "c-dot");

  // axis titles + legend
  s.appendChild(svgTag("text", { class: "c-label", "text-anchor": "middle", x: m.l + iw / 2, y: H - 4 }, "predicted probability"));
  s.appendChild(svgTag("rect", { class: "c-dot", x: m.l + 6, y: m.t + 4, width: 10, height: 10 }));
  s.appendChild(svgTag("text", { class: "c-label-strong", x: m.l + 20, y: m.t + 13 }, "isotonic (on the line)"));
  s.appendChild(svgTag("rect", { class: "c-dot-flag", x: m.l + 170, y: m.t + 4, width: 10, height: 10 }));
  s.appendChild(svgTag("text", { class: "c-label-flag", x: m.l + 184, y: m.t + 13 }, "uncalibrated"));

  el("svg-reliability").replaceChildren(s);
  table(
    el("tbl-reliability"),
    ["predicted", "empirical (uncal)", "empirical (isotonic)"],
    d.uncalibrated.map((p, i) => [
      p.pred.toFixed(3),
      p.emp.toFixed(3),
      (d.isotonic[i] ? d.isotonic[i].emp.toFixed(3) : "—"),
    ])
  );
}

/* ── helpers ───────────────────────────────────────────────────────────── */
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
function escapeHtml(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );
}

boot();
