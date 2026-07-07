"use strict";

/* The validation worksheet — demo controller.
   Hybrid scoring: try the live API with a short timeout, fall back to the precomputed
   preset so the page never hangs on a cold/absent endpoint. The decision is drawn as a
   control-limit gauge (probability vs the frozen review threshold) and settled with a
   REVIEW / APPROVE stamp. */

const CONFIG = window.FRAUD_CONFIG || {};
const API_BASE = (CONFIG.API_BASE || "").replace(/\/$/, "");
const API_LIVE = API_BASE && !API_BASE.includes("__FRAUD_API_BASE__");
const SCORE_TIMEOUT_MS = 3000;

// Optional GA4 (same property as projects 1–2).
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
}

/* ── masthead fields + results ledger ──────────────────────────────────── */
async function loadMetrics() {
  let m;
  try {
    m = await (await fetch("metrics.json", { cache: "no-store" })).json();
  } catch (e) {
    el("stats").innerHTML = '<li class="entry"><div class="entry-label">Metrics unavailable.</div></li>';
    return;
  }
  if (typeof m.threshold === "number") THRESHOLD = m.threshold;
  positionLimit();

  const fields = [
    ["Model", `${m.calibration}-calibrated ${m.model}`],
    ["Registry", `v${m.registry_version}`],
    ["Data", `${m.provenance} IEEE-CIS`],
    ["Review cost", `$${Number(m.review_cost).toFixed(0)} / alert`],
  ];
  el("fields").innerHTML = fields
    .map(([k, v]) => `<div><dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd></div>`)
    .join("");

  el("stats").innerHTML = (m.headline || [])
    .map(
      (h) =>
        `<li class="entry"><div class="entry-value">${withAccentUnit(h.value)}</div>` +
        `<div class="entry-label">${escapeHtml(h.label)}</div></li>`
    )
    .join("");
}

// Mute the unit/sign glyphs of a headline figure (keep the digits full ink).
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
      clearTimeout(timer); // timeout / network / cold start → fall through
    }
  }
  return { result: preset.expected, live: false };
}

/* ── control-limit gauge + verdict + factors ───────────────────────────── */
function positionLimit() {
  const x = pct(THRESHOLD);
  el("gauge-limit").style.left = x + "%";
  el("gauge-flag").style.left = x + "%";
  const tag = el("gauge-limit-tag");
  if (tag) tag.textContent = `review threshold ${(THRESHOLD * 100).toFixed(1)}%`;
}
function pct(p) {
  return clamp(Number(p) * 100, 0, 100);
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
    ? "crosses the control limit → flagged for review"
    : "under the control limit → cleared";
  el("gauge-scale").setAttribute(
    "role",
    "img"
  );
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
