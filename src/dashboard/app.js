// High-Frequency LOB Microstructure Dashboard Interactive Controller

document.addEventListener('DOMContentLoaded', () => {
  initEquityChart();
  renderLOBTable();
  renderTradeLog();
  setupFrictionSliders();
});

let equityChart;

function initEquityChart() {
  const ctx = document.getElementById('equityChart').getContext('2d');
  
  // Synthetic backtest equity data points
  const ticks = Array.from({length: 30}, (_, i) => `Tick ${i*5}`);
  const netPnL = [0, 0.12, 0.28, 0.45, 0.38, 0.62, 0.85, 1.10, 1.05, 1.32, 1.55, 1.48, 1.82, 2.10, 2.05, 2.35, 2.60, 2.52, 2.85, 3.12];
  const grossPnL = [0, 0.15, 0.32, 0.50, 0.45, 0.70, 0.95, 1.22, 1.18, 1.45, 1.70, 1.65, 2.00, 2.30, 2.25, 2.58, 2.85, 2.78, 3.15, 3.45];

  equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ticks.slice(0, netPnL.length),
      datasets: [
        {
          label: 'Net PnL (After Latency & Fees)',
          data: netPnL,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          fill: true,
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 3,
        },
        {
          label: 'Gross PnL (Before Fees)',
          data: grossPnL,
          borderColor: '#3b82f6',
          borderDash: [4, 4],
          fill: false,
          tension: 0.3,
          borderWidth: 1.5,
          pointRadius: 0,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#9ca3af', font: { family: 'Inter', size: 12 } }
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            label: function(context) {
              return `${context.dataset.label}: $${context.parsed.y.toFixed(4)}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#6b7280', font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#6b7280', font: { family: 'JetBrains Mono', size: 10 }, callback: v => '$' + v.toFixed(2) }
        }
      }
    }
  });
}

function renderLOBTable() {
  const tableBody = document.getElementById('lob-table-body');
  tableBody.innerHTML = '';

  // 10 levels of synthetic LOB data
  const baseMid = 100.00;
  const maxVol = 120.0;

  for (let i = 1; i <= 10; i++) {
    const bidPrice = (baseMid - i * 0.05).toFixed(2);
    const bidVol = (Math.sin(i) * 30 + 50).toFixed(0);
    const askPrice = (baseMid + i * 0.05).toFixed(2);
    const askVol = (Math.cos(i) * 30 + 45).toFixed(0);

    const bidPct = (parseFloat(bidVol) / maxVol) * 100;
    const askPct = (parseFloat(askVol) / maxVol) * 100;

    const row = document.createElement('tr');
    row.innerHTML = `
      <td>
        <div class="depth-bar-container">
          <div class="depth-bar-fill bid" style="width: ${bidPct}%;"></div>
          <span style="position: absolute; right: 6px; top: 2px; font-size: 0.75rem; color: #f9fafb;">${bidVol}</span>
        </div>
      </td>
      <td style="color: #10b981; font-weight: 600;">$${bidPrice}</td>
      <td style="color: #f43f5e; font-weight: 600;">$${askPrice}</td>
      <td>
        <div class="depth-bar-container">
          <div class="depth-bar-fill ask" style="width: ${askPct}%;"></div>
          <span style="position: absolute; left: 6px; top: 2px; font-size: 0.75rem; color: #f9fafb;">${askVol}</span>
        </div>
      </td>
    `;
    tableBody.appendChild(row);
  }
}

function renderTradeLog() {
  const logBody = document.getElementById('trade-log-body');
  logBody.innerHTML = '';

  const sampleTrades = [
    { id: 1045, tick: 'Tick 195', side: 'SELL', type: 'LIMIT', price: '$100.25', size: '1.0', fee: '-$0.0100 (Rebate)', pnl: '+$0.3500' },
    { id: 1044, tick: 'Tick 182', side: 'BUY',  type: 'LIMIT', price: '$99.90',  size: '1.0', fee: '-$0.0099 (Rebate)', pnl: '+$0.2800' },
    { id: 1043, tick: 'Tick 160', side: 'SELL', type: 'LIMIT', price: '$100.15', size: '1.0', fee: '-$0.0100 (Rebate)', pnl: '+$0.1500' },
    { id: 1042, tick: 'Tick 140', side: 'BUY',  type: 'LIMIT', price: '$99.80',  size: '1.0', fee: '-$0.0099 (Rebate)', pnl: '-$0.0800' },
    { id: 1041, tick: 'Tick 115', side: 'SELL', type: 'LIMIT', price: '$100.10', size: '1.0', fee: '-$0.0100 (Rebate)', pnl: '+$0.4200' },
  ];

  sampleTrades.forEach(tr => {
    const row = document.createElement('tr');
    const sideColor = tr.side === 'BUY' ? 'color: #10b981;' : 'color: #f43f5e;';
    const pnlColor = tr.pnl.startsWith('+') ? 'color: #10b981;' : 'color: #f43f5e;';

    row.innerHTML = `
      <td>#${tr.id}</td>
      <td style="color: var(--text-secondary);">${tr.tick}</td>
      <td style="${sideColor} font-weight: 700;">${tr.side}</td>
      <td><span class="badge-tag badge-indigo">${tr.type}</span></td>
      <td>${tr.price}</td>
      <td>${tr.size}</td>
      <td style="color: var(--accent-emerald);">${tr.fee}</td>
      <td style="${pnlColor} font-weight: 700;">${tr.pnl}</td>
    `;
    logBody.appendChild(row);
  });
}

function setupFrictionSliders() {
  const inputLat = document.getElementById('input-latency');
  const labelLat = document.getElementById('label-latency');
  const inputQueue = document.getElementById('input-queue');
  const labelQueue = document.getElementById('label-queue');
  const inputInv = document.getElementById('input-inventory');
  const labelInv = document.getElementById('label-inventory');

  inputLat.addEventListener('input', (e) => {
    const val = e.target.value;
    labelLat.textContent = `${val} ticks (approx ${(val * 1.0).toFixed(1)} ms)`;
    recalculateDashboard(val, inputQueue.value, inputInv.value);
  });

  inputQueue.addEventListener('input', (e) => {
    const val = e.target.value;
    labelQueue.textContent = `${val}% / tick`;
    recalculateDashboard(inputLat.value, val, inputInv.value);
  });

  inputInv.addEventListener('input', (e) => {
    const val = e.target.value;
    labelInv.textContent = `${(val * 1e-6).toExponential(1)}`;
    recalculateDashboard(inputLat.value, inputQueue.value, val);
  });
}

function recalculateDashboard(latency, queueDecay, inventoryPenalty) {
  // Real-time interactive recalculation formula simulating strategy performance degradation under frictions
  const latFactor = 1.0 - (latency - 1) * 0.035;
  const queueFactor = 1.0 + (queueDecay - 20) * 0.005;
  const invPenaltyVal = inventoryPenalty * 1e-6 * 50;

  const baseNetPnL = 3.1223;
  const newNetPnL = Math.max(-1.5, (baseNetPnL * latFactor * queueFactor) - invPenaltyVal);
  const newSharpe = Math.max(0.5, 194.94 * latFactor);
  const newFillRate = Math.max(40.0, Math.min(100.0, 88.2 * queueFactor * (1.0 - latency * 0.015)));

  document.getElementById('val-net-pnl').textContent = `$${newNetPnL.toFixed(4)}`;
  document.getElementById('val-sharpe').textContent = newSharpe.toFixed(2);
  document.getElementById('val-fill-rate').textContent = `${newFillRate.toFixed(1)}%`;

  if (newNetPnL >= 0) {
    document.getElementById('val-net-pnl').className = 'metric-value positive';
  } else {
    document.getElementById('val-net-pnl').className = 'metric-value negative';
  }

  // Update AI DeepLOB Forecast probabilities based on interactive slider inputs
  const upProb = Math.min(85.0, Math.max(10.0, 68.4 * latFactor)).toFixed(1);
  const downProb = Math.min(45.0, Math.max(5.0, 10.5 / latFactor)).toFixed(1);
  const statProb = (100.0 - parseFloat(upProb) - parseFloat(downProb)).toFixed(1);

  document.getElementById('prob-up').textContent = `${upProb}%`;
  document.getElementById('fill-up').style.width = `${upProb}%`;

  document.getElementById('prob-stat').textContent = `${statProb}%`;
  document.getElementById('fill-stat').style.width = `${statProb}%`;

  document.getElementById('prob-down').textContent = `${downProb}%`;
  document.getElementById('fill-down').style.width = `${downProb}%`;
}
