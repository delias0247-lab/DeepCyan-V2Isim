document.addEventListener("DOMContentLoaded", () => {
  const path = window.location.pathname || "/";
  highlightActivePage(path);
  initMobileSidebar();

  if (path.startsWith("/graph/comparison")) initComparisonGraphPage();
  else if (path.startsWith("/simulation")) initSimulationPage();
  else if (path.startsWith("/data")) initDataPage();
  else if (path.startsWith("/graph")) initGraphPage();
  else initDashboardPage();
});

async function api(path, method = "GET", body = null) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const response = await fetch(path, opts);
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text, status: response.status };
  }
}

function highlightActivePage(path) {
  document.querySelectorAll(".nav a").forEach((link) => {
    const href = link.getAttribute("href");
    const li = link.closest("li");
    if (!li) return;
    const isActive = href === path || (href === "/" && path === "/") || (href !== "/" && path.startsWith(`${href}/`));
    if (isActive) li.classList.add("active");
    else li.classList.remove("active");
  });
}

function initMobileSidebar() {
  const menuToggle = document.querySelector(".menu-toggle");
  const sidebar = document.querySelector(".sidebar");
  if (!menuToggle || !sidebar) return;

  menuToggle.addEventListener("click", () => {
    if (window.innerWidth > 1024) return;
    const isOpen = sidebar.style.display === "flex";
    sidebar.style.display = isOpen ? "none" : "flex";
    if (!isOpen) {
      sidebar.style.position = "fixed";
      sidebar.style.zIndex = "1000";
      sidebar.style.width = "260px";
      sidebar.style.left = "0";
      sidebar.style.top = "0";
    }
  });
}

function formatNum(value) {
  if (value === null || value === undefined || value === "") return "-";
  const n = Number(value);
  if (Number.isNaN(n)) return "-";
  return n.toFixed(2);
}

function initDashboardPage() {
  const tick = async () => {
    await Promise.all([
      updateDashboardStats(),
      initRecentRunsTable(),
      initComparisonTable(),
    ]);
  };

  tick();
  setInterval(tick, 5000);
}

async function updateDashboardStats() {
  const data = await api("/dashboard/metrics");
  const status = data?.status || null;

  const setText = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  };

  setText("statTotalMaps", String(data?.total_maps ?? 0));
  setText("statTotalMapsSub", `Models: ${data?.total_models ?? 0}`);

  if (data?.q) {
    setText("statQPerf", formatNum(data.q.final_cumulative_reward));
    setText("statQPerfSub", `Avg Queue: ${formatNum(data.q.avg_total_queue)} | Steps: ${data.q.total_steps ?? 0}`);
  } else {
    setText("statQPerf", "-");
    setText("statQPerfSub", "Waiting for a completed traci6 run");
  }

  if (data?.dqn) {
    setText("statDQNPerf", formatNum(data.dqn.final_cumulative_reward));
    setText("statDQNPerfSub", `Avg Queue: ${formatNum(data.dqn.avg_total_queue)} | Steps: ${data.dqn.total_steps ?? 0}`);
  } else {
    setText("statDQNPerf", "-");
    setText("statDQNPerfSub", "Waiting for a completed traci7 run");
  }

  if (status?.running) {
    setText("statRunStatus", "RUNNING");
    setText("statRunStatusSub", `${status.scenario || "-"} | ${status.model_label || status.model || "-"} | Steps: ${status.steps ?? 0}`);
  } else if (status?.outputs_url) {
    setText("statRunStatus", (status.run_status || "idle").toUpperCase());
    setText("statRunStatusSub", `Last run: ${status.scenario || "-"} | ${status.model_label || status.model || "-"}`);
  } else {
    setText("statRunStatus", "IDLE");
    setText("statRunStatusSub", "No active model run");
  }
}

async function initSimulationPage() {
  const scenarioSel = document.getElementById("scenarioSelect");
  const modelSel = document.getElementById("modelSelect");
  const guiSel = document.getElementById("guiSelect");
  const statusBox = document.getElementById("statusBox");
  const warningsBox = document.getElementById("warningsBox");
  const lastRunLink = document.getElementById("lastRunLink");
  const delaySlider = document.getElementById("delaySlider");
  const delayInput = document.getElementById("delayInput");
  const delayVal = document.getElementById("delayVal");
  const speedStepInput = document.getElementById("speedStepInput");
  const vehicleCurrentVal = document.getElementById("vehicleCurrentVal");
  const vehicleTargetInput = document.getElementById("vehicleTargetInput");
  const vehicleStepInput = document.getElementById("vehicleStepInput");
  const perfState = document.getElementById("perfState");
  const perfSteps = document.getElementById("perfSteps");
  const perfVehicles = document.getElementById("perfVehicles");
  const perfQueue = document.getElementById("perfQueue");
  const perfReward = document.getElementById("perfReward");
  const perfCumReward = document.getElementById("perfCumReward");
  const perfAvgQueue = document.getElementById("perfAvgQueue");
  const perfControl = document.getElementById("perfControl");
  if (!scenarioSel || !modelSel) return;

  let scenarios = [];
  let currentVehicleCount = null;
  let vehicleTargetInitialized = false;

  const setMetric = (el, value) => {
    if (el) el.textContent = value;
  };

  const getMaxDelay = () => Number(delaySlider?.max || delayInput?.max || 10);
  const clampDelay = (value) => {
    const n = Number(value);
    if (Number.isNaN(n)) return 0.10;
    return Math.min(getMaxDelay(), Math.max(0, n));
  };
  const syncDelayInputs = (value) => {
    const normalized = clampDelay(value).toFixed(2);
    if (delaySlider) delaySlider.value = normalized;
    if (delayInput) delayInput.value = normalized;
    if (delayVal) delayVal.textContent = normalized;
  };
  const getDelayStep = () => {
    const n = Number(speedStepInput?.value || "0.25");
    if (Number.isNaN(n) || n <= 0) return 0.25;
    return n;
  };
  const applyDelay = async (value) => {
    const delay_sec = clampDelay(value);
    syncDelayInputs(delay_sec);
    setStatus(await api("/speed/set", "POST", { delay_sec }));
    await refreshStatus();
  };
  const parseVehicleCount = (value, fallback = 0) => {
    const n = parseInt(value, 10);
    if (Number.isNaN(n) || n < 0) return fallback;
    return n;
  };
  const getVehicleStep = () => {
    const n = parseVehicleCount(vehicleStepInput?.value || "5", 5);
    return n > 0 ? n : 5;
  };
  const syncVehicleTarget = (value, force = false) => {
    if (!vehicleTargetInput) return;
    if (!force && document.activeElement === vehicleTargetInput) return;
    vehicleTargetInput.value = String(parseVehicleCount(value, 0));
  };
  const queueVehicleChange = async (delta) => {
    const count = Math.abs(parseVehicleCount(delta, 0));
    if (!count) {
      return refreshStatus();
    }
    const response = delta > 0
      ? await api("/vehicles/add", "POST", { count })
      : await api("/vehicles/remove", "POST", { count });
    setStatus(response);
    const status = await waitForCommandAck(response?.command_id);
    if (status?.vehicle_count !== undefined && status?.vehicle_count !== null) {
      syncVehicleTarget(status.vehicle_count, true);
    }
    return status;
  };

  const setStatus = (obj) => {
    if (statusBox) statusBox.textContent = JSON.stringify(obj, null, 2);
    if (obj?.outputs_url && lastRunLink) {
      lastRunLink.href = obj.outputs_url;
      lastRunLink.style.display = "inline-flex";
    } else if (lastRunLink) {
      lastRunLink.style.display = "none";
    }
    if (delayVal && obj?.delay_sec !== undefined && obj?.delay_sec !== null) {
      syncDelayInputs(obj.delay_sec);
    }

    setMetric(perfState, obj?.running ? (obj?.run_status || "running").toUpperCase() : (obj?.run_status || "idle").toUpperCase());
    setMetric(perfSteps, String(obj?.runtime_step ?? obj?.steps ?? 0));
    setMetric(perfVehicles, obj?.vehicle_count ?? "-");
    setMetric(perfQueue, formatNum(obj?.latest_total_queue));
    setMetric(perfReward, formatNum(obj?.latest_reward));
    setMetric(perfCumReward, formatNum(obj?.latest_cumulative_reward));
    setMetric(perfAvgQueue, formatNum(obj?.avg_total_queue));
    setMetric(perfControl, obj?.last_command ? `${obj.last_command} (${obj?.last_command_status || "pending"})` : "-");
    if (obj?.vehicle_count !== undefined && obj?.vehicle_count !== null) {
      currentVehicleCount = parseVehicleCount(obj.vehicle_count, 0);
      setMetric(vehicleCurrentVal, String(currentVehicleCount));
      if (!vehicleTargetInitialized) {
        syncVehicleTarget(currentVehicleCount, true);
        vehicleTargetInitialized = true;
      }
    }
  };

  const fillModels = () => {
    const scenario = scenarios.find((item) => item.name === scenarioSel.value);
    const models = scenario?.models || [];
    modelSel.innerHTML = "";
    models.forEach((model) => {
      const opt = document.createElement("option");
      opt.value = model.id;
      opt.textContent = model.display_name;
      modelSel.appendChild(opt);
    });
  };

  const loadScenarios = async () => {
    const data = await api("/scenarios");
    scenarios = data?.scenarios || [];
    scenarioSel.innerHTML = "";
    scenarios.forEach((scenario) => {
      const opt = document.createElement("option");
      opt.value = scenario.name;
      opt.textContent = scenario.name;
      scenarioSel.appendChild(opt);
    });
    scenarioSel.addEventListener("change", fillModels);
    fillModels();
  };

  const refreshStatus = async () => {
    const data = await api("/status");
    setStatus(data);
    return data;
  };

  const waitForCommandAck = async (commandId, attempts = 10) => {
    if (!commandId) return refreshStatus();
    for (let i = 0; i < attempts; i += 1) {
      const status = await refreshStatus();
      if ((status?.last_command_id ?? 0) >= commandId) return status;
      await new Promise((resolve) => setTimeout(resolve, 300));
    }
    return refreshStatus();
  };

  document.getElementById("btnStart")?.addEventListener("click", async () => {
    const payload = {
      scenario: scenarioSel.value,
      model: modelSel.value,
      gui: guiSel ? guiSel.value === "true" : true,
      extra_args: [],
    };
    setStatus(await api("/start", "POST", payload));
    await refreshStatus();
  });

  document.getElementById("btnStop")?.addEventListener("click", async () => {
    setStatus(await api("/stop", "POST"));
    await refreshStatus();
  });

  document.getElementById("btnRefreshStatus")?.addEventListener("click", async () => {
    await refreshStatus();
  });

  document.getElementById("btnFaster")?.addEventListener("click", async () => {
    await applyDelay((Number(delayInput?.value || delaySlider?.value || "0.10")) - getDelayStep());
  });

  document.getElementById("btnSlower")?.addEventListener("click", async () => {
    await applyDelay((Number(delayInput?.value || delaySlider?.value || "0.10")) + getDelayStep());
  });

  document.getElementById("btnSetSpeed")?.addEventListener("click", async () => {
    await applyDelay(delayInput?.value || delaySlider?.value || "0.10");
  });

  delaySlider?.addEventListener("input", () => {
    syncDelayInputs(delaySlider.value);
  });

  delayInput?.addEventListener("input", () => {
    syncDelayInputs(delayInput.value);
  });

  document.getElementById("btnAddVeh")?.addEventListener("click", async () => {
    await queueVehicleChange(getVehicleStep());
  });

  document.getElementById("btnRemVeh")?.addEventListener("click", async () => {
    await queueVehicleChange(-getVehicleStep());
  });

  document.getElementById("btnSetVeh")?.addEventListener("click", async () => {
    const liveCount = currentVehicleCount ?? parseVehicleCount(vehicleCurrentVal?.textContent || "0", 0);
    const targetCount = parseVehicleCount(vehicleTargetInput?.value || String(liveCount), liveCount);
    await queueVehicleChange(targetCount - liveCount);
  });

  await loadScenarios();
  await refreshStatus();

  setInterval(refreshStatus, 2000);

  if (warningsBox) {
    warningsBox.textContent = "Waiting for warnings... start a model run.\n";
    const es = new EventSource("/warnings/stream");
    const lines = [];
    const maxLines = 300;

    es.onmessage = (event) => {
      lines.push(event.data);
      if (lines.length > maxLines) lines.splice(0, lines.length - maxLines);
      warningsBox.textContent = lines.join("\n");
      warningsBox.scrollTop = warningsBox.scrollHeight;
    };

    es.onerror = () => {
      warningsBox.textContent += "\n[warning stream disconnected]\n";
    };
  }
}

async function initDataPage() {
  const mapSel = document.getElementById("dataMapSelect");
  const runSel = document.getElementById("dataRunSelect");
  const box = document.getElementById("csvFilesBox");
  const openRunBtn = document.getElementById("openRunFolderBtn");
  if (!mapSel || !runSel || !box) return;

  const payload = await api("/api/outputs");
  const maps = payload?.maps || [];
  if (!maps.length) {
    box.innerHTML = "<div style=\"padding:12px;border:1px solid #e2e8f0;border-radius:12px;background:#fff;\"><b>No available data yet.</b></div>";
    return;
  }

  mapSel.innerHTML = "";
  maps.forEach((map) => {
    const opt = document.createElement("option");
    opt.value = map.map;
    opt.textContent = map.map;
    mapSel.appendChild(opt);
  });

  const formatBytes = (bytes) => {
    const units = ["B", "KB", "MB", "GB"];
    let index = 0;
    let size = Number(bytes || 0);
    while (size >= 1024 && index < units.length - 1) {
      size /= 1024;
      index += 1;
    }
    return `${size.toFixed(1)} ${units[index]}`;
  };

  const renderFiles = (mapName, runId) => {
    const map = maps.find((item) => item.map === mapName);
    const run = (map?.runs || []).find((item) => item.run_id === runId);
    if (!run) return;
    if (openRunBtn) {
      openRunBtn.href = run.url;
      openRunBtn.style.display = "inline-flex";
    }
    const files = run.csv_files || [];
    if (!files.length) {
      box.innerHTML = "<div style=\"padding:12px;border:1px solid #e2e8f0;border-radius:12px;background:#fff;\"><b>No CSV files found for this run.</b></div>";
      return;
    }
    box.innerHTML = `<div style="display:flex;flex-direction:column;gap:10px;">${files.map((file) => `
      <a href="${file.url}" target="_blank" style="display:flex;justify-content:space-between;align-items:center;padding:12px;border:1px solid #e2e8f0;border-radius:12px;background:#fff;text-decoration:none;">
        <span><i class="fas fa-file-csv"></i> ${file.name}</span>
        <span style="color:#64748b;font-size:0.9rem;">${formatBytes(file.size_bytes || 0)}</span>
      </a>`).join("")}</div>`;
  };

  const renderRunsForMap = (mapName) => {
    const map = maps.find((item) => item.map === mapName);
    const runs = map?.runs || [];
    runSel.innerHTML = "";
    runs.forEach((run) => {
      const opt = document.createElement("option");
      opt.value = run.run_id;
      opt.textContent = run.model_label ? `${run.run_id} (${run.model_label})` : run.run_id;
      runSel.appendChild(opt);
    });
    if (runs.length) renderFiles(mapName, runSel.value);
  };

  mapSel.addEventListener("change", () => renderRunsForMap(mapSel.value));
  runSel.addEventListener("change", () => renderFiles(mapSel.value, runSel.value));
  renderRunsForMap(mapSel.value);
}

async function initGraphPage() {
  const mapSel = document.getElementById("graphMapSelect");
  const runSel = document.getElementById("graphRunSelect");
  const box = document.getElementById("plotsBox");
  const openPlotsBtn = document.getElementById("openPlotsBtn");
  if (!mapSel || !runSel || !box) return;

  const payload = await api("/api/plots");
  const maps = payload?.maps || [];
  if (!maps.length) {
    box.innerHTML = "<div style=\"padding:12px;border:1px solid #e2e8f0;border-radius:12px;background:#fff;\"><b>No plots found yet.</b></div>";
    return;
  }

  mapSel.innerHTML = "";
  maps.forEach((map) => {
    const opt = document.createElement("option");
    opt.value = map.map;
    opt.textContent = map.map;
    mapSel.appendChild(opt);
  });

  const renderPlots = (mapName, runId) => {
    const map = maps.find((item) => item.map === mapName);
    const run = (map?.runs || []).find((item) => item.run_id === runId);
    if (!run) return;
    const plots = run.plots || [];
    if (openPlotsBtn) {
      openPlotsBtn.href = run.plots_url;
      openPlotsBtn.style.display = plots.length ? "inline-flex" : "none";
    }
    if (!plots.length) {
      box.innerHTML = "<div style=\"padding:12px;border:1px solid #e2e8f0;border-radius:12px;background:#fff;\"><b>This run has no plots.</b></div>";
      return;
    }
    box.innerHTML = `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;">${plots.map((plot) => `
      <div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <b style="color:#0f172a;font-size:0.95rem;"><i class="fas fa-chart-area"></i> ${plot.name}</b>
          <a class="btn" href="${plot.url}" target="_blank" style="padding:6px 10px;font-size:0.8rem;"><i class="fas fa-up-right-from-square"></i> Open</a>
        </div>
        <a href="${plot.url}" target="_blank" style="text-decoration:none;">
          <img src="${plot.url}" alt="${plot.name}" style="width:100%;border-radius:12px;border:1px solid #e2e8f0;">
        </a>
      </div>`).join("")}</div>`;
  };

  const renderRunsForMap = (mapName) => {
    const map = maps.find((item) => item.map === mapName);
    const runs = map?.runs || [];
    runSel.innerHTML = "";
    runs.forEach((run) => {
      const opt = document.createElement("option");
      opt.value = run.run_id;
      opt.textContent = run.model_label ? `${run.run_id} (${run.model_label})` : run.run_id;
      runSel.appendChild(opt);
    });
    if (runs.length) renderPlots(mapName, runSel.value);
  };

  mapSel.addEventListener("change", () => renderRunsForMap(mapSel.value));
  runSel.addEventListener("change", () => renderPlots(mapSel.value, runSel.value));
  renderRunsForMap(mapSel.value);
}

function createComparisonChart(canvas, title, datasets, yLabel) {
  if (!canvas || typeof Chart === "undefined") return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  new Chart(ctx, {
    type: "line",
    data: {
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "nearest",
        intersect: false,
      },
      plugins: {
        legend: {
          position: "top",
        },
        title: {
          display: true,
          text: title,
        },
      },
      scales: {
        x: {
          type: "linear",
          title: {
            display: true,
            text: "Step",
          },
        },
        y: {
          title: {
            display: true,
            text: yLabel,
          },
        },
      },
    },
  });
}

async function initComparisonGraphPage() {
  const box = document.getElementById("comparisonGraphsBox");
  if (!box) return;

  const payload = await api("/api/comparison-graphs?max_points=320");
  const items = payload?.comparisons || [];
  if (!items.length) {
    box.innerHTML = "<div class=\"map-card\"><b>No traci5/traci6 comparison graphs are available yet.</b><p style=\"margin-top:10px;color:#64748b;\">Run both models on the same map from the dashboard, then come back here.</p></div>";
    return;
  }

  const runStatusText = (run) => run?.status ? `${run.model_label || run.model_id}: ${String(run.status).toUpperCase()}` : null;

  box.innerHTML = items.map((item, index) => `
    <div class="map-card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:14px;flex-wrap:wrap;">
        <div>
          <h3><i class="fas fa-map"></i> ${item.map}</h3>
          <p style="margin-top:8px;color:#64748b;font-size:0.95rem;">
            Winner: <b>${item.winner || "Need both runs"}</b> |
            Queue Gain: <b>${item.queue_gain === null || item.queue_gain === undefined ? "-" : formatNum(item.queue_gain)}</b> |
            Reward Gain: <b>${item.reward_gain === null || item.reward_gain === undefined ? "-" : formatNum(item.reward_gain)}</b>
          </p>
          <p style="margin-top:6px;color:#64748b;font-size:0.9rem;">
            ${runStatusText(item.fixed) || "traci5: no run yet"} |
            ${runStatusText(item.q_learning) || "traci6: no run yet"}
          </p>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          ${item.fixed?.url ? `<a class="btn" href="${item.fixed.url}" target="_blank"><i class="fas fa-folder-open"></i> traci5 Run</a>` : ""}
          ${item.q_learning?.url ? `<a class="btn" href="${item.q_learning.url}" target="_blank"><i class="fas fa-folder-open"></i> traci6 Run</a>` : ""}
        </div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-top:14px;">
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:12px;">
          <div style="color:#64748b;font-size:0.85rem;">traci5 Avg Queue</div>
          <div style="font-weight:700;color:#0f172a;margin-top:4px;">${formatNum(item.fixed?.avg_total_queue)}</div>
        </div>
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:12px;">
          <div style="color:#64748b;font-size:0.85rem;">traci6 Avg Queue</div>
          <div style="font-weight:700;color:#0f172a;margin-top:4px;">${formatNum(item.q_learning?.avg_total_queue)}</div>
        </div>
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:12px;">
          <div style="color:#64748b;font-size:0.85rem;">traci5 Final Reward</div>
          <div style="font-weight:700;color:#0f172a;margin-top:4px;">${formatNum(item.fixed?.final_cumulative_reward)}</div>
        </div>
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:12px;">
          <div style="color:#64748b;font-size:0.85rem;">traci6 Final Reward</div>
          <div style="font-weight:700;color:#0f172a;margin-top:4px;">${formatNum(item.q_learning?.final_cumulative_reward)}</div>
        </div>
      </div>

      ${(item.has_reward_graph || item.has_queue_graph) ? `
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px;margin-top:16px;">
          <div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:12px;">
            <div style="height:320px;">
              <canvas id="rewardCompareChart${index}"></canvas>
            </div>
          </div>
          <div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:12px;">
            <div style="height:320px;">
              <canvas id="queueCompareChart${index}"></canvas>
            </div>
          </div>
        </div>
      ` : `
        <div style="margin-top:16px;padding:14px;border:1px dashed #cbd5e1;border-radius:14px;background:#f8fafc;color:#475569;">
          ${item.fixed && item.q_learning
            ? "Both runs exist, but one of them does not have enough time-series data to draw the comparison charts."
            : "Run both traci5 and traci6 for this map from the dashboard to unlock the comparison graphs here."}
        </div>
      `}
    </div>
  `).join("");

  items.forEach((item, index) => {
    if (item.has_reward_graph) {
      createComparisonChart(
        document.getElementById(`rewardCompareChart${index}`),
        `${item.map} - Cumulative Reward`,
        [
          {
            label: "traci5 - Constant",
            data: item.fixed?.reward_points || [],
            borderColor: "#0f172a",
            backgroundColor: "rgba(15, 23, 42, 0.08)",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.15,
          },
          {
            label: "traci6 - Q-Learning",
            data: item.q_learning?.reward_points || [],
            borderColor: "#16a34a",
            backgroundColor: "rgba(22, 163, 74, 0.08)",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.15,
          },
        ],
        "Cumulative Reward",
      );
    }

    if (item.has_queue_graph) {
      createComparisonChart(
        document.getElementById(`queueCompareChart${index}`),
        `${item.map} - Total Queue`,
        [
          {
            label: "traci5 - Constant",
            data: item.fixed?.queue_points || [],
            borderColor: "#ef4444",
            backgroundColor: "rgba(239, 68, 68, 0.08)",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.15,
          },
          {
            label: "traci6 - Q-Learning",
            data: item.q_learning?.queue_points || [],
            borderColor: "#0ea5e9",
            backgroundColor: "rgba(14, 165, 233, 0.08)",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.15,
          },
        ],
        "Total Queue",
      );
    }
  });
}

async function initRecentRunsTable() {
  const body = document.getElementById("recentRunsBody");
  if (!body) return;
  const data = await api("/api/recent-runs?limit=20");
  const runs = data?.runs || [];
  if (!runs.length) {
    body.innerHTML = "<tr><td colspan=\"6\">No runs found yet.</td></tr>";
    return;
  }
  body.innerHTML = runs.map((run) => `
    <tr>
      <td>${run.time}</td>
      <td>${run.map}</td>
      <td>${run.run_id}</td>
      <td>${run.model_label || run.model_id || "-"}</td>
      <td>
        ${run.has_csv ? '<span class="status toll-pass">CSV</span>' : ""}
        ${run.has_plots ? '<span class="status accident">PLOTS</span>' : ""}
      </td>
      <td><a class="btn" href="${run.url}" target="_blank"><i class="fas fa-folder-open"></i> Open</a></td>
    </tr>
  `).join("");
}

async function initComparisonTable() {
  const body = document.getElementById("comparisonBody");
  if (!body) return;
  const data = await api("/api/comparisons");
  const items = data?.comparisons || [];
  if (!items.length) {
    body.innerHTML = "<tr><td colspan=\"6\">Run traci5 and traci6 from the dashboard to unlock comparisons.</td></tr>";
    return;
  }
  body.innerHTML = items.map((item) => {
    const fixed = item.fixed;
    const q = item.q_learning;
    const winner = item.winner || "Need both runs";
    const links = [
      fixed ? `<a class="btn" href="${fixed.url}" target="_blank" style="margin-right:6px;">traci5</a>` : "",
      q ? `<a class="btn" href="${q.url}" target="_blank">traci6</a>` : "",
    ].join("");
    return `
      <tr>
        <td>${item.map}</td>
        <td>${fixed ? formatNum(fixed.avg_total_queue) : "-"}</td>
        <td>${q ? formatNum(q.avg_total_queue) : "-"}</td>
        <td>${item.queue_gain === null || item.queue_gain === undefined ? "-" : formatNum(item.queue_gain)}</td>
        <td>${winner}</td>
        <td>${links || "-"}</td>
      </tr>
    `;
  }).join("");
}
