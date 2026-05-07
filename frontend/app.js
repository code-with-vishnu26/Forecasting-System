/* ═══════════════════════════════════
   SalesCast AI — Application Logic
   ═══════════════════════════════════ */

const API = 'http://localhost:8000';
const MODEL_COLORS = {
  ARIMA:   '#7c3aed',
  Prophet: '#3b82f6',
  XGBoost: '#10b981',
  LSTM:    '#f59e0b',
};

let allStates = [];
let perfData = null;
let chartInstances = {};

// ──────────────────────────────────────────
// INIT
// ──────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await checkHealth();
  await loadStates();
  await loadDashboard();
});

// ──────────────────────────────────────────
// NAVIGATION
// ──────────────────────────────────────────
function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`view-${name}`).classList.add('active');
  document.getElementById(`nav-${name}`).classList.add('active');

  const titles = {
    dashboard:   ['Dashboard', 'Overview of the forecasting system'],
    forecast:    ['State Forecast', '8-week sales prediction by state'],
    compare:     ['Model Comparison', 'Compare ARIMA, Prophet, XGBoost & LSTM'],
    performance: ['Performance', 'Validation metrics across all states'],
    api:         ['API Explorer', 'Interactive REST API documentation'],
  };
  document.getElementById('page-title').textContent = titles[name][0];
  document.getElementById('page-sub').textContent   = titles[name][1];

  if (name === 'performance') loadPerformanceView();
}

// ──────────────────────────────────────────
// HEALTH CHECK
// ──────────────────────────────────────────
async function checkHealth() {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  try {
    const res = await fetch(`${API}/health`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    if (data.status === 'healthy') {
      dot.className  = 'status-dot healthy';
      text.textContent = `API Healthy · ${data.total_states_trained} states`;
    } else {
      dot.className  = 'status-dot degraded';
      text.textContent = 'API Degraded';
    }
  } catch {
    dot.className  = 'status-dot error';
    text.textContent = 'API Offline';
    document.getElementById('api-badge').style.background = 'rgba(239,68,68,0.1)';
    document.getElementById('api-badge').style.color = '#ef4444';
    document.getElementById('api-badge').style.borderColor = 'rgba(239,68,68,0.25)';
    document.getElementById('api-badge').querySelector('.pulse-dot').style.background = '#ef4444';
    document.getElementById('api-badge').querySelector('span:last-child') || null;
    document.getElementById('api-badge').lastChild.textContent = 'API Offline';
  }
}

// ──────────────────────────────────────────
// LOAD STATES
// ──────────────────────────────────────────
async function loadStates() {
  try {
    const res = await fetch(`${API}/forecast/states`);
    const data = await res.json();
    allStates = data.states || [];
    document.getElementById('stat-states-val').textContent = allStates.length;

    const selectors = ['qs-state','forecast-state','compare-state','api-state-sel','api-state-sel2'];
    selectors.forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;
      const hasDefault = el.querySelector('option[value=""]');
      allStates.forEach(state => {
        const opt = document.createElement('option');
        opt.value = state;
        opt.textContent = state;
        el.appendChild(opt);
      });
    });
  } catch (e) {
    console.warn('Could not load states:', e.message);
  }
}

// ──────────────────────────────────────────
// DASHBOARD
// ──────────────────────────────────────────
async function loadDashboard() {
  try {
    const res = await fetch(`${API}/models/performance`);
    const data = await res.json();
    perfData = data;
    renderModelDistChart(data.states);
    renderHeatmap(data.states);
  } catch (e) {
    console.warn('Dashboard load error:', e.message);
  }
}

function renderModelDistChart(states) {
  const counts = { ARIMA: 0, Prophet: 0, XGBoost: 0, LSTM: 0 };
  states.forEach(s => { if (counts[s.best_model] !== undefined) counts[s.best_model]++; });

  destroyChart('model-dist-chart');
  const ctx = document.getElementById('model-dist-chart').getContext('2d');
  chartInstances['model-dist-chart'] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: Object.keys(counts),
      datasets: [{
        data: Object.values(counts),
        backgroundColor: Object.keys(counts).map(m => MODEL_COLORS[m] + 'cc'),
        borderColor: Object.keys(counts).map(m => MODEL_COLORS[m]),
        borderWidth: 2,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `${ctx.label}: ${ctx.parsed} states` } },
      },
      cutout: '65%',
    },
  });

  // Legend
  const legend = document.getElementById('model-dist-legend');
  legend.innerHTML = Object.entries(counts).map(([model, cnt]) => `
    <div class="legend-item">
      <span class="legend-dot" style="background:${MODEL_COLORS[model]}"></span>
      ${model}: <strong>${cnt}</strong>
    </div>`).join('');
}

function renderHeatmap(states) {
  const container = document.getElementById('heatmap-container');
  const models = ['ARIMA', 'Prophet', 'XGBoost', 'LSTM'];

  // Compute color scale per model
  const mapeValues = {};
  models.forEach(m => {
    mapeValues[m] = states.map(s => s[`${m}_mape`] || 0).filter(v => v < 1e10);
  });
  const getColor = (val, vals) => {
    if (!vals || vals.length === 0) return '#1f1f2e';
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const t = mx === mn ? 0 : (val - mn) / (mx - mn);
    const r = Math.round(10 + t * 120);
    const g = Math.round(185 - t * 160);
    const b = Math.round(129 - t * 100);
    return `rgb(${r},${g},${b})`;
  };

  let html = `<div class="heatmap-header">${models.map(m => `<div class="heatmap-col-label">${m}</div>`).join('')}</div>`;
  html += '<div class="heatmap-grid">';

  states.slice(0, 43).forEach(s => {
    html += `<div class="heatmap-row">
      <div class="heatmap-state-label">${s.state}</div>
      <div class="heatmap-cells">`;
    models.forEach(m => {
      const val = s[`${m}_mape`] || 0;
      const color = getColor(val, mapeValues[m]);
      const text = val > 0 && val < 1e10 ? val.toFixed(1) : '—';
      const textColor = val > 20 ? '#fff' : '#000';
      html += `<div class="heatmap-cell" style="background:${color};color:${textColor}" title="${s.state} - ${m}: MAPE ${text}%">${text}</div>`;
    });
    html += `</div></div>`;
  });

  html += '</div>';
  container.innerHTML = html;
}

// ──────────────────────────────────────────
// QUICK FORECAST (Dashboard)
// ──────────────────────────────────────────
async function quickForecast() {
  const state = document.getElementById('qs-state').value;
  if (!state) { alert('Please select a state.'); return; }

  show('quick-loading');
  hide('quick-result');

  try {
    const res = await fetch(`${API}/forecast/${encodeURIComponent(state)}?weeks=8`);
    const data = await res.json();

    document.getElementById('qr-state-name').textContent = data.state;
    document.getElementById('qr-model').textContent = data.best_model;

    const tbody = document.getElementById('qr-tbody');
    tbody.innerHTML = data.forecast.map((f, i) => `
      <tr>
        <td>Week ${i + 1}</td>
        <td>${f.date}</td>
        <td>$${formatNum(f.predicted_sales)}</td>
      </tr>`).join('');

    hide('quick-loading');
    show('quick-result');
  } catch (e) {
    hide('quick-loading');
    alert('Error: ' + e.message);
  }
}

// ──────────────────────────────────────────
// FORECAST VIEW
// ──────────────────────────────────────────
async function loadForecastView() {
  const state = document.getElementById('forecast-state').value;
  const weeks = parseInt(document.getElementById('forecast-weeks').value);
  if (!state) return;

  hide('forecast-view-content');
  hide('forecast-view-empty');
  show('forecast-view-loading');

  try {
    const [histRes, fcastRes] = await Promise.all([
      fetch(`${API}/forecast/${encodeURIComponent(state)}/history`),
      fetch(`${API}/forecast/${encodeURIComponent(state)}?weeks=${weeks}`),
    ]);
    const histData  = await histRes.json();
    const fcastData = await fcastRes.json();

    // Metrics
    const best = fcastData.best_model;
    const m = fcastData.model_comparison[best] || {};
    document.getElementById('fv-best-model').textContent = best;
    document.getElementById('fv-rmse').textContent = m.rmse ? '$' + formatNum(m.rmse) : '—';
    document.getElementById('fv-mae').textContent  = m.mae  ? '$' + formatNum(m.mae)  : '—';
    document.getElementById('fv-mape').textContent = m.mape ? m.mape.toFixed(2) + '%'  : '—';

    // Chart data
    const histDates  = histData.val_dates || [];
    const histActual = histData.val_actual || [];
    const histPreds  = (histData.val_predictions || {})[best] || [];
    const fcastDates = fcastData.forecast.map(f => f.date);
    const fcastVals  = fcastData.forecast.map(f => f.predicted_sales);

    destroyChart('forecast-chart');
    const ctx = document.getElementById('forecast-chart').getContext('2d');
    const datasets = [];

    if (histDates.length > 0) {
      datasets.push({
        label: 'Actual (Validation)',
        data: histDates.map((d, i) => ({ x: d, y: histActual[i] })),
        borderColor: '#94a3b8',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 3,
        tension: 0.3,
      });
      if (histPreds.length > 0) {
        datasets.push({
          label: `${best} (Validation Fit)`,
          data: histDates.map((d, i) => ({ x: d, y: histPreds[i] })),
          borderColor: MODEL_COLORS[best],
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [5, 3],
          pointRadius: 3,
          tension: 0.3,
        });
      }
    }

    datasets.push({
      label: `${best} Forecast (Next ${weeks} Weeks)`,
      data: fcastDates.map((d, i) => ({ x: d, y: fcastVals[i] })),
      borderColor: MODEL_COLORS[best] || '#7c3aed',
      backgroundColor: (MODEL_COLORS[best] || '#7c3aed') + '22',
      borderWidth: 2.5,
      pointRadius: 5,
      pointBackgroundColor: MODEL_COLORS[best] || '#7c3aed',
      fill: true,
      tension: 0.3,
    });

    chartInstances['forecast-chart'] = new Chart(ctx, {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { type: 'time', time: { unit: 'week' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b8ba8' } },
          y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b8ba8', callback: v => '$' + formatNum(v) } },
        },
        plugins: {
          legend: { labels: { color: '#f0f0fa', font: { size: 12 } } },
          tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: $${formatNum(ctx.parsed.y)}` } },
        },
        interaction: { mode: 'index', intersect: false },
      },
    });

    // Table
    const tbody = document.getElementById('forecast-table-body');
    tbody.innerHTML = fcastData.forecast.map((f, i) => {
      const prev = i > 0 ? fcastData.forecast[i - 1].predicted_sales : f.predicted_sales;
      const diff = f.predicted_sales - prev;
      const pct  = prev !== 0 ? (diff / prev * 100).toFixed(1) : '0.0';
      const cls  = diff >= 0 ? 'change-up' : 'change-down';
      const sign = diff >= 0 ? '▲' : '▼';
      return `<tr>
        <td>Week ${i + 1}</td>
        <td>${f.date}</td>
        <td>$${formatNum(f.predicted_sales)}</td>
        <td class="${cls}">${i === 0 ? '—' : `${sign} ${Math.abs(pct)}%`}</td>
      </tr>`;
    }).join('');

    hide('forecast-view-loading');
    hide('forecast-view-empty');
    show('forecast-view-content');
  } catch (e) {
    hide('forecast-view-loading');
    show('forecast-view-empty');
    console.error(e);
  }
}

// ──────────────────────────────────────────
// COMPARE VIEW
// ──────────────────────────────────────────
async function loadCompareView() {
  const state = document.getElementById('compare-state').value;
  if (!state) return;

  hide('compare-content');
  hide('compare-empty');
  show('compare-loading');

  try {
    const res = await fetch(`${API}/forecast/${encodeURIComponent(state)}/all-models?weeks=8`);
    const data = await res.json();

    const dates  = data.future_dates || [];
    const models = data.all_models  || {};

    destroyChart('compare-chart');
    const ctx = document.getElementById('compare-chart').getContext('2d');
    const datasets = Object.entries(models).map(([name, info]) => ({
      label: name,
      data: (info.predictions || []).map((v, i) => ({ x: dates[i], y: v })),
      borderColor: MODEL_COLORS[name] || '#888',
      backgroundColor: 'transparent',
      borderWidth: name === data.best_model ? 3 : 1.5,
      pointRadius: 4,
      tension: 0.3,
      borderDash: name === data.best_model ? [] : [4, 3],
    }));

    chartInstances['compare-chart'] = new Chart(ctx, {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { type: 'time', time: { unit: 'week' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b8ba8' } },
          y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b8ba8', callback: v => '$' + formatNum(v) } },
        },
        plugins: {
          legend: { labels: { color: '#f0f0fa', font: { size: 12 } } },
          tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: $${formatNum(ctx.parsed.y)}` } },
        },
        interaction: { mode: 'index', intersect: false },
      },
    });

    // Metrics cards
    const mgrid = document.getElementById('metrics-comparison-table');
    mgrid.innerHTML = Object.entries(models).map(([name, info]) => {
      const m = info.metrics || {};
      const isBest = name === data.best_model;
      return `
        <div class="metric-card ${isBest ? 'best' : ''}">
          ${isBest ? '<span class="best-badge">Best</span>' : ''}
          <div class="metric-model-name">
            <span class="model-dot" style="background:${MODEL_COLORS[name]}"></span>
            ${name}
          </div>
          <div class="metric-row"><span class="label">RMSE</span><span class="value">${m.rmse ? '$'+formatNum(m.rmse) : '—'}</span></div>
          <div class="metric-row"><span class="label">MAE</span> <span class="value">${m.mae  ? '$'+formatNum(m.mae)  : '—'}</span></div>
          <div class="metric-row"><span class="label">MAPE</span><span class="value">${m.mape ? m.mape.toFixed(2)+'%' : '—'}</span></div>
        </div>`;
    }).join('');

    hide('compare-loading');
    hide('compare-empty');
    show('compare-content');
  } catch (e) {
    hide('compare-loading');
    show('compare-empty');
    console.error(e);
  }
}

// ──────────────────────────────────────────
// PERFORMANCE VIEW
// ──────────────────────────────────────────
let perfTableData = [];

async function loadPerformanceView() {
  show('perf-loading');
  hide('perf-content');

  try {
    if (!perfData) {
      const res = await fetch(`${API}/models/performance`);
      perfData = await res.json();
    }
    perfTableData = perfData.states || [];

    // Bar chart — best RMSE per state (top 20)
    const top20 = [...perfTableData]
      .sort((a, b) => {
         const aBest = Math.min(...['ARIMA','Prophet','XGBoost','LSTM'].map(m => a[`${m}_rmse`]||1e18).filter(v=>v>0));
         const bBest = Math.min(...['ARIMA','Prophet','XGBoost','LSTM'].map(m => b[`${m}_rmse`]||1e18).filter(v=>v>0));
         return aBest - bBest;
      })
      .slice(0, 20);

    destroyChart('perf-chart');
    const ctx = document.getElementById('perf-chart').getContext('2d');
    const models = ['ARIMA', 'Prophet', 'XGBoost', 'LSTM'];
    chartInstances['perf-chart'] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: top20.map(s => s.state),
        datasets: models.map(m => ({
          label: m,
          data: top20.map(s => {
             const v = s[`${m}_rmse`] || 0;
             return v > 1e10 ? 0 : v;
          }),
          backgroundColor: MODEL_COLORS[m] + '99',
          borderColor: MODEL_COLORS[m],
          borderWidth: 1,
          borderRadius: 3,
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b8ba8', font: { size: 10 }, maxRotation: 45 } },
          y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b8ba8', callback: v => '$' + formatNum(v) } },
        },
        plugins: {
          legend: { labels: { color: '#f0f0fa', font: { size: 12 } } },
          tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: $${formatNum(ctx.parsed.y)}` } },
        },
      },
    });

    renderPerfTable(perfTableData);
    hide('perf-loading');
    show('perf-content');
  } catch (e) {
    hide('perf-loading');
    console.error(e);
  }
}

function renderPerfTable(data) {
  const tbody = document.getElementById('perf-tbody');
  tbody.innerHTML = data.map(s => {
    const rm = {
      ARIMA:   s.ARIMA_rmse   || 0,
      Prophet: s.Prophet_rmse || 0,
      XGBoost: s.XGBoost_rmse || 0,
      LSTM:    s.LSTM_rmse    || 0,
    };
    const bestRmse = Math.min(...Object.values(rm).filter(v => v > 0));
    const modelColor = MODEL_COLORS[s.best_model] || '#888';
    return `<tr>
      <td><strong>${s.state}</strong></td>
      <td><span style="color:${modelColor};font-weight:700">${s.best_model}</span></td>
      ${['ARIMA','Prophet','XGBoost','LSTM'].map(m => {
        const v = rm[m];
        const isBest = v === bestRmse && v > 0;
        return `<td style="${isBest ? 'color:var(--green);font-weight:700' : ''}">${v > 0 ? '$'+formatNum(v) : '—'}</td>`;
      }).join('')}
      <td style="color:var(--green);font-weight:700">${bestRmse > 0 ? '$'+formatNum(bestRmse) : '—'}</td>
    </tr>`;
  }).join('');
}

function filterPerfTable() {
  const q    = document.getElementById('perf-search').value.toLowerCase();
  const sort = document.getElementById('perf-sort').value;
  let data = [...(perfData?.states || [])].filter(s => s.state.toLowerCase().includes(q));

  if (sort === 'rmse_asc') {
    data.sort((a, b) => {
      const av = Math.min(...['ARIMA','Prophet','XGBoost','LSTM'].map(m => a[`${m}_rmse`]||1e18).filter(v=>v>0));
      const bv = Math.min(...['ARIMA','Prophet','XGBoost','LSTM'].map(m => b[`${m}_rmse`]||1e18).filter(v=>v>0));
      return av - bv;
    });
  } else if (sort === 'rmse_desc') {
    data.sort((a, b) => {
      const av = Math.min(...['ARIMA','Prophet','XGBoost','LSTM'].map(m => a[`${m}_rmse`]||1e18).filter(v=>v>0));
      const bv = Math.min(...['ARIMA','Prophet','XGBoost','LSTM'].map(m => b[`${m}_rmse`]||1e18).filter(v=>v>0));
      return bv - av;
    });
  } else {
    data.sort((a, b) => a.state.localeCompare(b.state));
  }

  renderPerfTable(data);
}

// ──────────────────────────────────────────
// API EXPLORER
// ──────────────────────────────────────────
async function tryEndpoint(path, containerId) {
  const el = document.getElementById(containerId);
  el.textContent = 'Loading...';
  el.classList.remove('hidden');
  try {
    const res  = await fetch(API + path);
    const data = await res.json();
    el.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    el.textContent = `Error: ${e.message}`;
  }
}

async function tryStateForecast() {
  const state = document.getElementById('api-state-sel').value;
  if (!state) return;
  await tryEndpoint(`/forecast/${encodeURIComponent(state)}?weeks=8`, 'forecast-res');
}

async function tryAllModels() {
  const state = document.getElementById('api-state-sel2').value;
  if (!state) return;
  await tryEndpoint(`/forecast/${encodeURIComponent(state)}/all-models?weeks=8`, 'all-models-res');
}

async function tryBatch() {
  const el = document.getElementById('batch-res');
  el.textContent = 'Loading...';
  el.classList.remove('hidden');
  const sample = allStates.slice(0, 3);
  try {
    const res = await fetch(`${API}/forecast/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ states: sample, weeks: 8 }),
    });
    const data = await res.json();
    el.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    el.textContent = `Error: ${e.message}`;
  }
}

// ──────────────────────────────────────────
// HELPERS
// ──────────────────────────────────────────
function formatNum(n) {
  if (!n && n !== 0) return '—';
  return Number(n).toLocaleString('en-US', { maximumFractionDigits: 0 });
}

function show(id) { document.getElementById(id)?.classList.remove('hidden'); }
function hide(id) { document.getElementById(id)?.classList.add('hidden'); }

function destroyChart(id) {
  if (chartInstances[id]) {
    chartInstances[id].destroy();
    delete chartInstances[id];
  }
}
