"use strict";

/* The analyst's terminal — demo controller.
   Hybrid scoring: try the live API with a short timeout, fall back to the precomputed
   preset result so the page never hangs or breaks on a cold/absent endpoint. The
   decision is rendered as a point on a number line crossing the frozen threshold. */

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
let selectedIndex = null;
let firstScore = true;

async function boot() {
  await loadMetrics();
  await loadPresets();
  setModeTag(API_LIVE ? "live" : "offline");
}

/* ── results card ──────────────────────────────────────────────────────── */
async function loadMetrics() {
  try {
    const m = await (await fetch("metrics.json", { cache: "no-store" })).json();
    if (typeof m.threshold === "number") THRESHOLD = m.threshold;
    el("model-tag").textContent = `${m.model} v${m.registry_version}`;
    positionThreshold();
    const ol = el("stats");
    ol.innerHTML = "";
    (m.headline || []).forEach((h) => {
      const li = document.createElement("li");
      li.className = "stat";
      li.innerHTML =
        `<div class="stat-value">${withAccentUnit(h.value)}</div>` +
        `<div class="stat-label">${escapeHtml(h.label)}</div>`;
      ol.appendChild(li);
    });
  } catch (e) {
    el("stats").innerHTML = '<li class="stat"><div class="stat-label">Metrics unavailable.</div></li>';
  }
}

// Colour the unit/sign glyph of a headline number with the data accent.
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
  const chips = el("chips");
  chips.innerHTML = "";
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
    chips.appendChild(btn);
  });
}

/* ── scoring (hybrid: live API with timeout → precomputed fallback) ─────── */
async function selectPreset(i) {
  selectedIndex = i;
  const chips = el("chips").querySelectorAll(".chip");
  chips.forEach((c, j) => c.setAttribute("aria-pressed", String(j === i)));
  el("nl-caption").textContent = "scoring…";

  const preset = PRESETS[i];
  const { result, live } = await scoreTransaction(preset);
  setModeTag(live ? "live" : "offline");
  renderResult(result);
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

/* ── the number line + verdict + factors ───────────────────────────────── */
function positionThreshold() {
  el("nl-threshold").style.left = pct(THRESHOLD) + "%";
  el("nl-zone-alert").style.left = pct(THRESHOLD) + "%";
  const tag = el("nl-threshold").querySelector(".nl-threshold-tag");
  if (tag) tag.textContent = `threshold ${(THRESHOLD * 100).toFixed(1)}%`;
}

function pct(p) {
  return clamp(Number(p) * 100, 0, 100);
}

function renderResult(r) {
  const p = Number(r.fraud_probability);
  const isAlert = r.decision ? r.decision === "review" : p >= THRESHOLD;
  const target = clamp(p * 100, 0.7, 99.3);

  // number line
  const point = el("nl-point");
  point.hidden = false;
  point.className = "nl-point " + (isAlert ? "is-alert" : "is-approve");
  el("nl-point-tag").textContent = fmtPct(p);
  if (firstScore) {
    point.style.left = "0%";
    requestAnimationFrame(() => requestAnimationFrame(() => (point.style.left = target + "%")));
    firstScore = false;
  } else {
    point.style.left = target + "%";
  }
  el("nl-caption").textContent = isAlert
    ? "above the threshold → sent for review"
    : "below the threshold → approved";
  el("numberline").setAttribute(
    "aria-label",
    `Calibrated fraud probability ${fmtPct(p)}, threshold ${(THRESHOLD * 100).toFixed(1)}%, ` +
      `decision ${isAlert ? "review" : "approve"}.`
  );

  // verdict
  el("readout").hidden = false;
  const badge = el("verdict-badge");
  badge.textContent = isAlert ? "ALERT · REVIEW" : "APPROVE";
  badge.className = "verdict-badge " + (isAlert ? "alert" : "approve");
  el("verdict-detail").innerHTML =
    `calibrated P(fraud) <b>${fmtPct(p)}</b> vs frozen threshold ` +
    `<b>${(THRESHOLD * 100).toFixed(1)}%</b>`;

  renderBars(r.top_factors || []);

  const parts = [];
  if (r.model_version) parts.push("model v" + r.model_version);
  if (r.calibration) parts.push(r.calibration + " calibration");
  if (r.provenance) parts.push(r.provenance + " data");
  el("model-meta").textContent = parts.join(" · ");
}

function renderBars(factors) {
  const bars = el("bars");
  bars.innerHTML = "";
  const top = factors.slice(0, 6);
  const maxAbs = Math.max(0.001, ...top.map((f) => Math.abs(Number(f.shap))));
  top.forEach((f) => {
    const shap = Number(f.shap);
    const w = (Math.abs(shap) / maxAbs) * 48; // half-track max width (%)
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

/* ── small helpers ─────────────────────────────────────────────────────── */
function setModeTag(mode) {
  const t = el("mode-tag");
  if (!API_LIVE) {
    t.textContent = "cached";
    t.className = "tag tag-live is-offline";
    t.title = "scoring source: precomputed (no live endpoint in this build)";
    return;
  }
  const live = mode === "live";
  t.textContent = live ? "live API" : "cached";
  t.className = "tag tag-live " + (live ? "is-live" : "is-offline");
  t.title = live
    ? "scored just now by the live AWS endpoint"
    : "endpoint cold or unreachable — showing the precomputed result";
}

function fmtPct(p) {
  const v = Number(p) * 100;
  if (v >= 99.95) return "100%";
  if (v < 0.1) return v.toFixed(3) + "%";
  if (v < 10) return v.toFixed(1) + "%";
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
