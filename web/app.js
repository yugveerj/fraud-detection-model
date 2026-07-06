"use strict";

const CONFIG = window.FRAUD_CONFIG || {};
const API_BASE = (CONFIG.API_BASE || "").replace(/\/$/, "");
const API_LIVE = API_BASE && !API_BASE.includes("__FRAUD_API_BASE__");

// Optional GA4 (same property as projects 1–2).
if (CONFIG.GA_MEASUREMENT_ID) {
  const s = document.createElement("script");
  s.async = true;
  s.src = "https://www.googletagmanager.com/gtag/js?id=" + CONFIG.GA_MEASUREMENT_ID;
  document.head.appendChild(s);
  window.dataLayer = window.dataLayer || [];
  window.gtag = function () { window.dataLayer.push(arguments); };
  window.gtag("js", new Date());
  window.gtag("config", CONFIG.GA_MEASUREMENT_ID);
}

const el = (id) => document.getElementById(id);
let PRESETS = [];

async function boot() {
  const banner = el("mode-banner");
  if (!API_LIVE) {
    banner.hidden = false;
    banner.textContent =
      "Offline preview — the live scoring API is not wired in this build; showing precomputed model responses.";
  }
  try {
    const res = await fetch("presets.json", { cache: "no-store" });
    const data = await res.json();
    PRESETS = data.presets || [];
    renderPresets(data);
  } catch (e) {
    el("presets").innerHTML = "<p>Could not load example transactions.</p>";
  }
}

function renderPresets(data) {
  const container = el("presets");
  container.innerHTML = "";
  PRESETS.forEach((p, i) => {
    const card = document.createElement("button");
    card.className = "card";
    card.type = "button";
    const d = p.display || {};
    card.innerHTML = `
      <h4>${escapeHtml(p.label || p.name)}</h4>
      <dl>
        <dt>amount</dt><dd>$${fmtNum(d.TransactionAmt)}</dd>
        <dt>product</dt><dd>${escapeHtml(d.ProductCD || "—")}</dd>
        <dt>card</dt><dd>${escapeHtml(d.card4 || "—")} / ${escapeHtml(d.card6 || "—")}</dd>
        <dt>email</dt><dd>${escapeHtml(d.P_emaildomain || "—")}</dd>
      </dl>`;
    card.addEventListener("click", () => scorePreset(i));
    container.appendChild(card);
  });
}

async function scorePreset(i) {
  const preset = PRESETS[i];
  let result = preset.expected;
  if (API_LIVE) {
    try {
      const res = await fetch(API_BASE + "/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(preset.payload),
      });
      if (res.ok) result = await res.json();
    } catch (e) {
      /* fall back to precomputed */
    }
  }
  renderResult(result);
}

function renderResult(r) {
  el("result").hidden = false;
  el("prob-value").textContent = (r.fraud_probability * 100).toFixed(1) + "%";
  el("threshold-value").textContent = Number(r.threshold).toFixed(3);
  const badge = el("decision-badge");
  badge.textContent = r.decision === "review" ? "REVIEW" : "APPROVE";
  badge.className = "badge " + r.decision;

  const tbody = el("factors").querySelector("tbody");
  tbody.innerHTML = "";
  (r.top_factors || []).forEach((f) => {
    const tr = document.createElement("tr");
    const cls = f.shap >= 0 ? "pos" : "neg";
    tr.innerHTML = `<td>${escapeHtml(f.feature)}</td><td>${fmtNum(f.value)}</td>` +
      `<td class="${cls}">${f.shap >= 0 ? "+" : ""}${Number(f.shap).toFixed(3)}</td>`;
    tbody.appendChild(tr);
  });

  const parts = [];
  if (r.model_version) parts.push("model v" + r.model_version);
  if (r.calibration) parts.push(r.calibration + " calibration");
  if (r.provenance) parts.push(r.provenance + " data");
  el("model-meta").textContent = parts.join(" · ");
  el("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function fmtNum(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  return Number.isInteger(n) ? n.toLocaleString() : n.toFixed(2);
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

boot();
