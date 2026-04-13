document.addEventListener('DOMContentLoaded', () => {
  // ===== ACTIVE PAGE HIGHLIGHT =====
  const path = window.location.pathname || '/';
  document.querySelectorAll('.nav a').forEach(link => {
    const href = link.getAttribute('href');
    const li = link.closest('li');
    if (!li) return;

    if (href === path || (href === '/' && path === '/')) li.classList.add('active');
    else li.classList.remove('active');
  });

  // ===== MOBILE SIDEBAR TOGGLE =====
  const menuToggle = document.querySelector('.menu-toggle');
  const sidebar = document.querySelector('.sidebar');

  if (menuToggle && sidebar) {
    menuToggle.addEventListener('click', () => {
      if (window.innerWidth <= 1024) {
        const isOpen = (sidebar.style.display === 'flex');
        sidebar.style.display = isOpen ? 'none' : 'flex';
        if (!isOpen) {
          sidebar.style.position = 'fixed';
          sidebar.style.zIndex = '1000';
          sidebar.style.width = '260px';
          sidebar.style.left = '0';
          sidebar.style.top = '0';
        }
      }
    });
  }

  // ===== PAGE INIT =====
  if (path.startsWith('/simulation')) initSimulationPage();
  else if (path.startsWith('/data')) initDataPage();
  else if (path.startsWith('/graph')) initGraphPage();
  else {
    initDashboardStats();
    initRecentRunsTable();
    setInterval(initRecentRunsTable, 5000);
    if (typeof Chart !== "undefined") initChart();
  }
});

async function api(path, method = "GET", body = null) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  const t = await r.text();
  try { return JSON.parse(t); }
  catch { return { raw: t, status: r.status }; }
}

// ===============================
// Dashboard chart (demo)
// ===============================
function initChart() {
  const canvas = document.getElementById('salesChart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['00:00','02:00','04:00','06:00','08:00','10:00','12:00','14:00','16:00','18:00','20:00','22:00'],
      datasets: [{
        label: 'Vehicles per hour',
        data: [210,150,130,380,1120,1450,1380,1520,1840,2110,1750,980],
        backgroundColor: 'rgba(56, 189, 248, 0.1)',
        borderColor: '#38bdf8',
        tension: 0.3,
        fill: true
      }]
    },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } } }
  });
}

// ===============================
// Dashboard live cards
// ===============================
function initDashboardStats() {
  const elTotalMaps = document.getElementById('statTotalMaps');
  const elTotalMapsSub = document.getElementById('statTotalMapsSub');
  const elQ = document.getElementById('statQPerf');
  const elQSub = document.getElementById('statQPerfSub');
  const elDQN = document.getElementById('statDQNPerf');
  const elDQNSub = document.getElementById('statDQNPerfSub');
  const elRun = document.getElementById('statRunStatus');
  const elRunSub = document.getElementById('statRunStatusSub');
  if (!elTotalMaps || !elQ || !elDQN || !elRun) return;

  const fmt = (n) => {
    if (n === null || n === undefined) return "–";
    const x = Number(n);
    if (Number.isNaN(x)) return "–";
    return x.toFixed(2);
  };
  const setText = (el, txt) => { if (el) el.textContent = txt; };

  async function tick() {
    const m = await api('/dashboard/metrics');

    // Card 1
    if (m && typeof m.total_maps !== "undefined") {
      setText(elTotalMaps, String(m.total_maps));
      setText(elTotalMapsSub, `Configs: ${m.total_configs ?? 0}`);
    } else {
      setText(elTotalMaps, "–");
      setText(elTotalMapsSub, "Configs: –");
    }

    // Card 4
    const st = (m && m.status) ? m.status : null;
    if (st) {
      setText(elRun, st.running ? "RUNNING" : "STOPPED");
      setText(elRunSub, st.running
        ? `Scenario: ${st.scenario || "–"} | SimTime: ${fmt(st.sim_time)} | Vehicles: ${st.vehicle_count ?? 0}`
        : (st.last_outputs_url ? `Last run: ${st.last_outputs_url}` : "No active simulation"));
    } else {
      setText(elRun, "–");
      setText(elRunSub, "Status unavailable");
    }

    // Cards 2 & 3
    const q = m ? m.q : null;
    const dqn = m ? m.dqn : null;

    if (q) {
      setText(elQ, fmt(q.final_cumulative_reward));
      setText(elQSub, `Avg Queue: ${fmt(q.avg_total_queue)} | Steps: ${q.total_steps ?? 0}`);
    } else {
      setText(elQ, "–");
      setText(elQSub, "No previous run found");
    }

    if (dqn) {
      setText(elDQN, fmt(dqn.final_cumulative_reward));
      setText(elDQNSub, `Avg Queue: ${fmt(dqn.avg_total_queue)} | Steps: ${dqn.total_steps ?? 0}`);
    } else {
      setText(elDQN, "–");
      setText(elDQNSub, "No runs found yet");
    }
  }

  tick();
  setInterval(tick, 2000);
}

// ===============================
// Simulation Controls Page
// ===============================
async function initSimulationPage() {
  const scenarioSel = document.getElementById('scenarioSelect');
  const configSel = document.getElementById('configSelect');
  const guiSel = document.getElementById('guiSelect');
  const statusBox = document.getElementById('statusBox');
  const warningsBox = document.getElementById('warningsBox');
  const delaySlider = document.getElementById('delaySlider');
  const delayVal = document.getElementById('delayVal');
  const lastRunLink = document.getElementById('lastRunLink');
  if (!scenarioSel || !configSel || !guiSel) return;

  const setStatus = (obj) => {
    if (statusBox) statusBox.textContent = JSON.stringify(obj, null, 2);
    if (obj && obj.outputs_url && lastRunLink) {
      lastRunLink.href = obj.outputs_url;
      lastRunLink.style.display = "inline-flex";
    }
  };

  async function loadScenarios() {
    const data = await api('/scenarios');
    const list = data.scenarios || [];

    scenarioSel.innerHTML = "";
    list.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.name;
      opt.textContent = s.name;
      scenarioSel.appendChild(opt);
    });

    function fillConfigs() {
      const sName = scenarioSel.value;
      const obj = list.find(x => x.name === sName);
      const cfgs = (obj && obj.configs) ? obj.configs : [];
      configSel.innerHTML = "";
      cfgs.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c;
        configSel.appendChild(opt);
      });
    }

    scenarioSel.addEventListener('change', fillConfigs);
    fillConfigs();
  }

  async function refreshStatus() {
    setStatus(await api('/status'));
  }

  document.getElementById('btnStart')?.addEventListener('click', async () => {
    const scenario = scenarioSel.value;
    const config = configSel.value;
    const gui = (guiSel.value === "true");
    setStatus(await api('/start', 'POST', { scenario, config, gui, extra_args: [] }));
    await refreshStatus();
  });

  document.getElementById('btnStop')?.addEventListener('click', async () => {
    setStatus(await api('/stop', 'POST'));
    await refreshStatus();
  });

  document.getElementById('btnPause')?.addEventListener('click', async () => {
    setStatus(await api('/pause', 'POST'));
    await refreshStatus();
  });

  document.getElementById('btnResume')?.addEventListener('click', async () => {
    setStatus(await api('/resume', 'POST'));
    await refreshStatus();
  });

  document.getElementById('btnFaster')?.addEventListener('click', async () => {
    setStatus(await api('/speed/up', 'POST'));
    await refreshStatus();
  });

  document.getElementById('btnSlower')?.addEventListener('click', async () => {
    setStatus(await api('/speed/down', 'POST'));
    await refreshStatus();
  });

  document.getElementById('btnSetSpeed')?.addEventListener('click', async () => {
    const delay_sec = parseFloat(delaySlider?.value || "0.1");
    setStatus(await api('/speed/set', 'POST', { delay_sec }));
    await refreshStatus();
  });

  delaySlider?.addEventListener('input', () => {
    if (delayVal) delayVal.textContent = parseFloat(delaySlider.value).toFixed(2);
  });

  document.getElementById('btnAddVeh')?.addEventListener('click', async () => {
    const count = parseInt(document.getElementById('addCount')?.value || "1", 10);
    setStatus(await api('/vehicles/add', 'POST', { count }));
    await refreshStatus();
  });

  document.getElementById('btnRemVeh')?.addEventListener('click', async () => {
    const count = parseInt(document.getElementById('remCount')?.value || "1", 10);
    setStatus(await api('/vehicles/remove', 'POST', { count }));
    await refreshStatus();
  });

  document.getElementById('btnRefreshStatus')?.addEventListener('click', refreshStatus);

  await loadScenarios();
  await refreshStatus();

  // Live warnings (SSE)
  if (warningsBox) {
    warningsBox.textContent = "Waiting for warnings... (start simulation)\n";
    const es = new EventSource('/warnings/stream');
    const lines = [];
    const MAX = 300;

    es.onmessage = (e) => {
      lines.push(e.data);
      if (lines.length > MAX) lines.splice(0, lines.length - MAX);
      warningsBox.textContent = lines.join("\n");
      warningsBox.scrollTop = warningsBox.scrollHeight;
    };

    es.onerror = () => {
      warningsBox.textContent += "\n[stream disconnected - refresh page]\n";
    };
  }
}

// ===============================
// Data Page
// ===============================
async function initDataPage() {
  const mapSel = document.getElementById('dataMapSelect');
  const runSel = document.getElementById('dataRunSelect');
  const box = document.getElementById('csvFilesBox');
  const openRunBtn = document.getElementById('openRunFolderBtn');
  if (!mapSel || !runSel || !box) return;

  const setError = (msg) => {
    box.innerHTML = `<div style="padding:12px;border:1px solid #fecaca;border-radius:12px;background:#fff;">
      <b style="color:#b91c1c;">${msg}</b>
    </div>`;
    if (openRunBtn) openRunBtn.style.display = "none";
  };

  const setNoData = () => {
    box.innerHTML = `<div style="padding:12px;border:1px solid #e2e8f0;border-radius:12px;background:#fff;">
      <b>No available data yet!!</b>
    </div>`;
    if (openRunBtn) openRunBtn.style.display = "none";
  };

  let payload;
  try {
    payload = await api('/api/outputs');
  } catch (e) {
    setError("Failed to load outputs API. Check if /api/outputs works.");
    return;
  }

  const maps = (payload && payload.maps) ? payload.maps : [];
  if (maps.length === 0) {
    mapSel.innerHTML = `<option value="">No maps found</option>`;
    runSel.innerHTML = `<option value="">No runs</option>`;
    setNoData();
    return;
  }

  // Fill map dropdown
  mapSel.innerHTML = "";
  maps.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.map;
    opt.textContent = m.map;
    mapSel.appendChild(opt);
  });

  const formatBytes = (bytes) => {
    const units = ["B","KB","MB","GB"];
    let i = 0;
    let b = Number(bytes || 0);
    while (b >= 1024 && i < units.length - 1) { b /= 1024; i++; }
    return `${b.toFixed(1)} ${units[i]}`;
  };

  function renderFiles(mapName, runId) {
    const m = maps.find(x => x.map === mapName);
    const runs = (m && m.runs) ? m.runs : [];
    const r = runs.find(x => x.run_id === runId);
    if (!r) { setNoData(); return; }

    if (openRunBtn) {
      openRunBtn.href = r.url;
      openRunBtn.style.display = "inline-flex";
    }

    const files = r.csv_files || [];
    if (files.length === 0) { setNoData(); return; }

    box.innerHTML = `
      <div style="display:flex;flex-direction:column;gap:10px;">
        ${files.map(f => `
          <a href="${f.url}" target="_blank"
             style="display:flex;justify-content:space-between;align-items:center;
                    padding:12px;border:1px solid #e2e8f0;border-radius:12px;background:#fff;text-decoration:none;">
            <span><i class="fas fa-file-csv"></i> ${f.name}</span>
            <span style="color:#64748b;font-size:0.9rem;">${formatBytes(f.size_bytes || 0)}</span>
          </a>
        `).join("")}
      </div>
    `;
  }

  function renderRunsForMap(mapName) {
    const m = maps.find(x => x.map === mapName);
    const runs = (m && m.runs) ? m.runs : [];
    runSel.innerHTML = "";

    if (runs.length === 0) {
      runSel.innerHTML = `<option value="">No runs</option>`;
      setNoData();
      return;
    }

    runs.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.run_id;
      opt.textContent = r.run_id + (r.has_csv ? "" : " (no csv)");
      runSel.appendChild(opt);
    });

    renderFiles(mapName, runSel.value);
  }

  mapSel.addEventListener('change', () => renderRunsForMap(mapSel.value));
  runSel.addEventListener('change', () => renderFiles(mapSel.value, runSel.value));

  renderRunsForMap(mapSel.value);
}

async function initGraphPage() {
  const mapSel = document.getElementById('graphMapSelect');
  const runSel = document.getElementById('graphRunSelect');
  const box = document.getElementById('plotsBox');
  const openPlotsBtn = document.getElementById('openPlotsBtn');

  if (!mapSel || !runSel || !box) return;

  const setError = (msg) => {
    box.innerHTML = `<div style="padding:12px;border:1px solid #fecaca;border-radius:12px;background:#fff;">
      <b style="color:#b91c1c;">${msg}</b>
    </div>`;
    if (openPlotsBtn) openPlotsBtn.style.display = "none";
  };

  const setNoData = (msg = "No plots found yet!") => {
    box.innerHTML = `<div style="padding:12px;border:1px solid #e2e8f0;border-radius:12px;background:#fff;">
      <b>${msg}</b>
    </div>`;
    if (openPlotsBtn) openPlotsBtn.style.display = "none";
  };

  let payload;
  try {
    payload = await api('/api/plots');
  } catch (e) {
    setError("Failed to load plots API. Check if /api/plots works.");
    return;
  }

  const maps = (payload && payload.maps) ? payload.maps : [];
  if (maps.length === 0) {
    mapSel.innerHTML = `<option value="">No maps</option>`;
    runSel.innerHTML = `<option value="">No runs</option>`;
    setNoData("No maps found in outputs folder.");
    return;
  }

  // Fill map dropdown
  mapSel.innerHTML = "";
  maps.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.map;
    opt.textContent = m.map;
    mapSel.appendChild(opt);
  });

  function renderPlots(mapName, runId) {
    const m = maps.find(x => x.map === mapName);
    const runs = (m && m.runs) ? m.runs : [];
    const r = runs.find(x => x.run_id === runId);
    if (!r) { setNoData(); return; }

    const plots = r.plots || [];
    if (plots.length === 0) { setNoData("This run has no plots."); return; }

    if (openPlotsBtn) {
      openPlotsBtn.href = r.plots_url;
      openPlotsBtn.style.display = "inline-flex";
    }

    // Simple responsive grid
    box.innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;">
        ${plots.map(p => `
          <div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:12px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
              <b style="color:#0f172a;font-size:0.95rem;">
                <i class="fas fa-chart-area"></i> ${p.name}
              </b>
              <a class="btn" href="${p.url}" target="_blank" style="padding:6px 10px;font-size:0.8rem;">
                <i class="fas fa-up-right-from-square"></i> Open
              </a>
            </div>
            <a href="${p.url}" target="_blank" style="text-decoration:none;">
              <img src="${p.url}" alt="${p.name}" style="width:100%;border-radius:12px;border:1px solid #e2e8f0;">
            </a>
          </div>
        `).join("")}
      </div>
    `;
  }

  function renderRunsForMap(mapName) {
    const m = maps.find(x => x.map === mapName);
    const runs = (m && m.runs) ? m.runs : [];

    runSel.innerHTML = "";
    if (runs.length === 0) {
      runSel.innerHTML = `<option value="">No runs</option>`;
      setNoData("No runs found for this map.");
      return;
    }

    runs.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.run_id;
      opt.textContent = r.run_id + (r.has_plots ? "" : " (no plots)");
      runSel.appendChild(opt);
    });

    renderPlots(mapName, runSel.value);
  }

  mapSel.addEventListener('change', () => renderRunsForMap(mapSel.value));
  runSel.addEventListener('change', () => renderPlots(mapSel.value, runSel.value));

  renderRunsForMap(mapSel.value);
}

async function initRecentRunsTable(){
  const body = document.getElementById("recentRunsBody");
  if (!body) return;

  const data = await api("/api/recent-runs?limit=20");
  const runs = data?.runs || [];

  if (!runs.length){
    body.innerHTML = `<tr><td colspan="6">No runs found yet.</td></tr>`;
    return;
  }

  body.innerHTML = runs.map(r => `
    <tr>
      <td>${r.time}</td>
      <td>${r.map}</td>
      <td>${r.run_id}</td>
      <td>—</td>
      <td>
        ${r.has_csv ? `<span class="status toll-pass">CSV</span>` : ``}
        ${r.has_plots ? `<span class="status accident">PLOTS</span>` : ``}
      </td>
      <td><a class="btn" href="${r.url}" target="_blank"><i class="fas fa-folder-open"></i> Open</a></td>
    </tr>
  `).join("");
}