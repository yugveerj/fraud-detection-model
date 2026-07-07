// Deploy-time config. The deploy workflow rewrites FRAUD_API_BASE to the live API
// Gateway URL. Left as a placeholder here so the demo runs offline (it falls back to
// the precomputed preset responses and shows an "offline" banner).
window.FRAUD_CONFIG = {
  API_BASE: "https://mrx6i8np6k.execute-api.us-east-2.amazonaws.com", // e.g. https://abc123.execute-api.us-west-2.amazonaws.com
  GA_MEASUREMENT_ID: "", // same GA4 property as projects 1–2; set to enable analytics
};
