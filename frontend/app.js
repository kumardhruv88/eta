/**
 * app.js — Driver ETA Predictor frontend logic
 * Talks to FastAPI at localhost:8000
 */

// API base URL — change this to your Render URL when deployed
// e.g. const API = "https://driver-eta-api.onrender.com";
const API = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://127.0.0.1:8000"
  : "https://driver-eta-api.onrender.com";  // ← replace with your actual Render URL after deploying


// ────────────────────────────────────────────────
// State
// ────────────────────────────────────────────────
const state = {
  operator: 0,       // 0=Uber, 1=Lyft
  zones: [],
  shapeChart: null,
};

// ────────────────────────────────────────────────
// Utility: fetch wrapper
// ────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ────────────────────────────────────────────────
// Health check + badge update
// ────────────────────────────────────────────────
async function checkHealth() {
  try {
    const data = await apiFetch("/health");
    const dot  = document.getElementById("model-dot");
    const txt  = document.getElementById("model-status-text");
    if (data.model_loaded) {
      dot.className = "badge-dot online";
      txt.textContent = "Model Online";
    } else {
      dot.className = "badge-dot offline";
      txt.textContent = "Model Not Loaded";
    }
  } catch {
    document.getElementById("model-dot").className = "badge-dot offline";
    document.getElementById("model-status-text").textContent = "API Offline";
  }
}

// ────────────────────────────────────────────────
// Load zones into selects
// ────────────────────────────────────────────────
async function loadZones() {
  try {
    const data = await apiFetch("/zones");
    state.zones = data.zones;

    const puSel = document.getElementById("pu-zone");
    const doSel = document.getElementById("do-zone");

    const emptyOpt = '<option value="">Select zone...</option>';
    const opts = data.zones.map(z =>
      `<option value="${z.id}">${z.id} — ${z.name}</option>`
    ).join("");

    puSel.innerHTML = emptyOpt + opts;
    doSel.innerHTML = emptyOpt + opts;

    // Defaults: Midtown Center (161) → Upper East Side N (236)
    puSel.value = "161";
    doSel.value = "236";
  } catch {
    document.getElementById("pu-zone").innerHTML = '<option value="">Zone list unavailable</option>';
    document.getElementById("do-zone").innerHTML = '<option value="">Zone list unavailable</option>';
  }
}

// ────────────────────────────────────────────────
// Populate hour dropdown
// ────────────────────────────────────────────────
function populateHours() {
  const sel = document.getElementById("pickup-hour");
  const labels = {
    0:"Midnight", 1:"1am", 2:"2am", 3:"3am", 4:"4am", 5:"5am",
    6:"6am", 7:"7am", 8:"8am (rush)", 9:"9am (rush)",
    10:"10am", 11:"11am", 12:"Noon", 13:"1pm", 14:"2pm", 15:"3pm", 16:"4pm",
    17:"5pm (rush)", 18:"6pm (rush)", 19:"7pm", 20:"8pm", 21:"9pm",
    22:"10pm (night)", 23:"11pm (night)"
  };
  sel.innerHTML = '<option value="">Select hour...</option>' +
    Object.entries(labels).map(([v, l]) =>
      `<option value="${v}">${l}</option>`
    ).join("");
  // Default: 8am
  sel.value = "8";
}

// ────────────────────────────────────────────────
// Auto-compute derived flags from hour + day
// ────────────────────────────────────────────────
function updateAutoFlags() {
  const hour = parseInt(document.getElementById("pickup-hour").value) || 0;
  const dow  = parseInt(document.getElementById("pickup-dow").value);

  const isWeekend  = dow === 5 || dow === 6;
  const isRushHour = [7, 8, 9, 17, 18, 19].includes(hour);
  const isNight    = [22, 23, 0, 1, 2, 3, 4].includes(hour);

  setFlag("flag-weekend", isWeekend,  "Weekend");
  setFlag("flag-rush",    isRushHour, "Rush Hour");
  setFlag("flag-night",   isNight,    "Night");

  const hintMap = {
    8: "Morning rush — expect higher ETA",
    9: "Morning rush — expect higher ETA",
    17: "Evening rush — peak congestion",
    18: "Evening rush — peak congestion",
    19: "Evening rush — winding down",
    22: "Late night — fewer drivers",
    23: "Late night — fewer drivers",
    0:  "Midnight — very low supply",
    2:  "Early hours — minimal traffic",
  };
  document.getElementById("hour-hint").textContent = hintMap[hour] || "";
}

function setFlag(id, active, label) {
  const el = document.getElementById(id);
  if (active) el.classList.add("active"); else el.classList.remove("active");
}

// ────────────────────────────────────────────────
// Operator toggle
// ────────────────────────────────────────────────
function setupOperatorToggle() {
  document.querySelectorAll(".toggle-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".toggle-btn[data-field='operator']").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.operator = parseInt(btn.dataset.value);
    });
  });
}

// ────────────────────────────────────────────────
// Build feature pills summary
// ────────────────────────────────────────────────
function buildFeaturePills(req) {
  const pills = [
    { label: `Hour ${req.pickup_hour}`, active: true },
    { label: req.is_weekend  ? "Weekend"   : "Weekday",   active: req.is_weekend  === 1 },
    { label: req.is_rush_hour? "Rush Hour" : "Off-Peak",  active: req.is_rush_hour === 1 },
    { label: req.is_night    ? "Night"     : "Day",       active: req.is_night     === 1 },
    { label: req.operator === 0 ? "Uber" : "Lyft",        active: true },
    { label: `PU: Zone ${req.PULocationID}`,              active: true },
    { label: `DO: Zone ${req.DOLocationID}`,              active: true },
    { label: req.shared_request_flag ? "Shared" : "Private", active: req.shared_request_flag === 1 },
    { label: req.wav_request_flag ? "WAV" : "",           active: req.wav_request_flag === 1 },
  ].filter(p => p.label);

  document.getElementById("feature-pills").innerHTML = pills
    .map(p => `<span class="feature-pill${p.active ? " active" : ""}">${p.label}</span>`)
    .join("");
}

// ────────────────────────────────────────────────
// Animate ETA ring
// ────────────────────────────────────────────────
function animateRing(etaSec) {
  // Map ETA 0–1800s onto ring fill (0–1)
  const pct = Math.min(etaSec / 1200, 1.0);
  const circumference = 2 * Math.PI * 85; // r=85
  const fill = document.getElementById("ring-fill-el");
  fill.style.strokeDashoffset = circumference * (1 - pct);

  // Inject SVG gradient
  const svg = fill.closest("svg");
  if (!svg.querySelector("defs")) {
    svg.insertAdjacentHTML("afterbegin", `
      <defs>
        <linearGradient id="etaGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stop-color="#6c63ff"/>
          <stop offset="100%" stop-color="#00d4aa"/>
        </linearGradient>
      </defs>
    `);
  }
}

// Count-up animation
function countUp(el, target, unit = "", duration = 1000) {
  const start = performance.now();
  const from  = 0;
  function step(now) {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(from + (target - from) * eased) + unit;
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ────────────────────────────────────────────────
// Show result
// ────────────────────────────────────────────────
function showResult(data, req) {
  document.getElementById("empty-state").classList.add("hidden");
  document.getElementById("error-state").classList.add("hidden");
  const display = document.getElementById("result-display");
  display.classList.remove("hidden");

  // Animate seconds
  const secEl = document.getElementById("eta-seconds");
  countUp(secEl, Math.round(data.predicted_eta_seconds));

  // Minutes
  document.getElementById("eta-minutes").textContent =
    `≈ ${data.predicted_eta_minutes.toFixed(1)} min`;

  // Ring
  setTimeout(() => animateRing(data.predicted_eta_seconds), 100);

  // Confidence chip
  const chip = document.getElementById("confidence-chip");
  const conf = data.confidence.toLowerCase();
  chip.className = `result-chip ${conf}`;
  document.getElementById("confidence-label").textContent = data.confidence;
  document.getElementById("interpretation-text").textContent = data.interpretation;

  // Range bar
  const low    = data.eta_range_low_s;
  const high   = data.eta_range_high_s;
  const center = data.predicted_eta_seconds;
  document.getElementById("range-low").textContent  = `${Math.round(low)}s`;
  document.getElementById("range-high").textContent = `${Math.round(high)}s`;

  const pct = Math.max(5, Math.min(95, ((center - low) / (high - low + 1)) * 100));
  document.getElementById("range-fill").style.width  = `${pct}%`;
  document.getElementById("range-thumb").style.left  = `${pct}%`;

  // Feature pills
  buildFeaturePills(req);
}

function showError(msg) {
  document.getElementById("empty-state").classList.add("hidden");
  document.getElementById("result-display").classList.add("hidden");
  const errDiv = document.getElementById("error-state");
  errDiv.classList.remove("hidden");
  document.getElementById("error-msg").textContent = msg;
}

// ────────────────────────────────────────────────
// Form submit → predict
// ────────────────────────────────────────────────
document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const hour = parseInt(document.getElementById("pickup-hour").value);
  const dow  = parseInt(document.getElementById("pickup-dow").value);

  const req = {
    pickup_hour:         hour,
    pickup_day_of_week:  dow,
    is_weekend:          (dow === 5 || dow === 6) ? 1 : 0,
    is_rush_hour:        [7,8,9,17,18,19].includes(hour) ? 1 : 0,
    is_night:            [22,23,0,1,2,3,4].includes(hour) ? 1 : 0,
    PULocationID:        parseInt(document.getElementById("pu-zone").value),
    DOLocationID:        parseInt(document.getElementById("do-zone").value),
    operator:            state.operator,
    shared_request_flag: document.getElementById("shared-flag").checked ? 1 : 0,
    wav_request_flag:    document.getElementById("wav-flag").checked    ? 1 : 0,
  };

  const btn     = document.getElementById("predict-btn");
  const btnText = document.getElementById("btn-text");
  const spinner = document.getElementById("btn-spinner");
  const arrow   = document.getElementById("btn-arrow");

  btn.disabled = true;
  btnText.textContent = "Predicting...";
  spinner.classList.remove("hidden");
  arrow.classList.add("hidden");

  try {
    const result = await apiFetch("/predict", {
      method: "POST",
      body: JSON.stringify(req),
    });
    showResult(result, req);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btnText.textContent = "Predict ETA";
    spinner.classList.add("hidden");
    arrow.classList.remove("hidden");
  }
});

// ────────────────────────────────────────────────
// Load metrics dashboard
// ────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const data = await apiFetch("/metrics");
    const m    = data.metrics || data;

    const cards = [
      { label: "MAE", value: `${(m.MAE_seconds||0).toFixed(1)}s`,  color: "#6c63ff", note: "Avg error (seconds)" },
      { label: "RMSE", value: `${(m.RMSE_seconds||0).toFixed(1)}s`, color: "#00d4aa", note: "Penalizes large errors" },
      { label: "MAPE", value: `${(m.MAPE_pct||0).toFixed(1)}%`,    color: "#ffd166", note: "Percentage error" },
      { label: "R²",   value: (m.R2||0).toFixed(3),                 color: "#ff9ff3", note: "Variance explained" },
      { label: "p90 Error", value: `${(m.p90_error_s||0).toFixed(0)}s`, color: "#ff6b6b", note: "90th percentile error" },
    ];

    // Hero stats
    if (m.MAE_seconds) {
      document.getElementById("stat-mae").textContent  = (m.MAE_seconds).toFixed(1) + "s";
      document.getElementById("stat-r2").textContent   = (m.R2||0).toFixed(3);
      document.getElementById("stat-mape").textContent = (m.MAPE_pct||0).toFixed(1) + "%";
      if (m.n_test) {
        document.getElementById("stat-trips").textContent =
          ((m.n_test || 60000)/1000).toFixed(0) + "k";
      }
    }

    document.getElementById("metrics-grid").innerHTML = cards.map(c => `
      <div class="metric-card">
        <div class="metric-value" style="color:${c.color}">${c.value}</div>
        <div class="metric-label">${c.label}</div>
        <div class="metric-note">${c.note}</div>
      </div>
    `).join("");
  } catch {
    document.getElementById("metrics-grid").innerHTML =
      '<p style="color:var(--text-muted);font-size:.85rem;grid-column:1/-1;text-align:center;">' +
      'Run the pipeline first: <code>python src/pipeline/full_pipeline.py</code></p>';
  }
}

// ────────────────────────────────────────────────
// Load SHAP chart
// ────────────────────────────────────────────────
async function loadShapChart() {
  try {
    const data = await apiFetch("/model-info");
    const shap = data.shap_importance;

    if (!shap || shap.length === 0) {
      document.getElementById("shap-chart").closest(".shap-chart-container").innerHTML =
        '<p style="color:var(--text-muted);font-size:.85rem;text-align:center;padding:2rem;">' +
        'SHAP data available after running Notebook 06.</p>';
      return;
    }

    // Color by feature type
    const zoneFeats   = ["PULocationID", "DOLocationID"];
    const timeFeats   = ["pickup_hour", "pickup_day_of_week", "is_rush_hour", "is_night", "is_weekend"];
    const getColor = (name) => {
      if (zoneFeats.includes(name))   return "rgba(108,99,255,0.85)";
      if (timeFeats.includes(name))   return "rgba(0,212,170,0.85)";
      return "rgba(255,209,102,0.85)";
    };

    const sorted = [...shap].sort((a, b) => b.mean_abs_shap - a.mean_abs_shap);
    const labels = sorted.map(s => s.feature);
    const values = sorted.map(s => parseFloat(s.mean_abs_shap.toFixed(4)));
    const colors = sorted.map(s => getColor(s.feature));

    const ctx = document.getElementById("shap-chart").getContext("2d");
    if (state.shapChart) state.shapChart.destroy();

    state.shapChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Mean |SHAP|",
          data: values,
          backgroundColor: colors,
          borderRadius: 6,
          borderSkipped: false,
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#1a1a2e",
            borderColor: "rgba(108,99,255,0.3)",
            borderWidth: 1,
            titleColor: "#f0f0ff",
            bodyColor: "#9090b0",
            callbacks: {
              label: (ctx) => ` Mean |SHAP|: ${ctx.parsed.x.toFixed(4)}`
            }
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(255,255,255,0.04)" },
            ticks: { color: "#9090b0", font: { family: "'JetBrains Mono', monospace", size: 11 } },
          },
          y: {
            grid: { display: false },
            ticks: { color: "#f0f0ff", font: { family: "'Inter', sans-serif", size: 12 } },
          }
        }
      }
    });
  } catch {
    // No SHAP data yet — show placeholder
  }
}

// ────────────────────────────────────────────────
// Init
// ────────────────────────────────────────────────
async function init() {
  populateHours();
  setupOperatorToggle();

  // Update flags whenever hour/day changes
  document.getElementById("pickup-hour").addEventListener("change", updateAutoFlags);
  document.getElementById("pickup-dow").addEventListener("change",  updateAutoFlags);
  updateAutoFlags();

  // Default day: Tuesday (index 1)
  document.getElementById("pickup-dow").value = "1";
  updateAutoFlags();

  await Promise.allSettled([
    checkHealth(),
    loadZones(),
    loadMetrics(),
    loadShapChart(),
  ]);

  // Recheck health every 30s
  setInterval(checkHealth, 30_000);
}

document.addEventListener("DOMContentLoaded", init);
