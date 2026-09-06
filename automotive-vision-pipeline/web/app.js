/**
 * DriveSense AI — Real-Time Unified Dashboard Frontend Logic (fixed)
 *
 * Fixes vs previous version:
 * 1. Rendering no longer depends on the <video> element successfully playing.
 *    A virtual frame clock (setInterval) drives all HUD/gauge/chart updates,
 *    so a broken/missing video no longer freezes the entire dashboard.
 * 2. Added video error/loadeddata handling with a visible on-screen message
 *    instead of silent failure.
 * 3. Model A/B/C switching now calls the real backend /predict endpoint
 *    (throttled) instead of faking risk scores with local JS formulas.
 */

document.addEventListener("DOMContentLoaded", () => {
  const video = document.getElementById("dashcamVideo");
  const overlayCanvas = document.getElementById("telemetryOverlayCanvas");
  const overlayCtx = overlayCanvas.getContext("2d");

  const timelineCanvas = document.getElementById("riskTimelineCanvas");
  const timelineCtx = timelineCanvas.getContext("2d");

  const btnPlayPause = document.getElementById("btnPlayPause");
  const playIcon = document.getElementById("playIcon");
  const btnReset = document.getElementById("btnReset");

  const chkLane = document.getElementById("chkLane");
  const chkBoxes = document.getElementById("chkBoxes");
  const chkTtc = document.getElementById("chkTtc");

  const hudFrameNumber = document.getElementById("hudFrameNumber");
  const hudMinTtc = document.getElementById("hudMinTtc");
  const hudLaneOffset = document.getElementById("hudLaneOffset");
  const hudTtcAlert = document.getElementById("hudTtcAlert");

  const riskPercentValue = document.getElementById("riskPercentValue");
  const riskSeverityBadge = document.getElementById("riskSeverityBadge");
  const gaugeProgress = document.getElementById("gaugeProgress");
  const shapFactorsList = document.getElementById("shapFactorsList");
  const latencyVal = document.getElementById("latencyVal");
  const decisionZoneVal = document.getElementById("decisionZoneVal");

  const telemOffsetVal = document.getElementById("telemOffsetVal");
  const offsetThumb = document.getElementById("offsetThumb");
  const telemDensityVal = document.getElementById("telemDensityVal");
  const telemVruVal = document.getElementById("telemVruVal");
  const telemConfVal = document.getElementById("telemConfVal");

  const modalityButtons = document.querySelectorAll(".btn-modality");
  const activeModelPill = document.getElementById("activeModelLabel");

  let currentModel = "C";
  let telemetryData = [];
  let telemetryLoaded = false;
  let currentFrameIdx = 0;
  let riskHistory = [];
  let isPlaying = false;
  let clockTimer = null;

  const maxHistory = 150;
  const TOTAL_FRAMES = 300;
  const FRAME_INTERVAL_MS = 100; // 10fps virtual clock, independent of <video>
  const PREDICT_EVERY_MS = 500;  // throttle real backend calls
  let lastPredictAt = 0;

  function resizeCanvases() {
    overlayCanvas.width = overlayCanvas.parentElement.clientWidth;
    overlayCanvas.height = overlayCanvas.parentElement.clientHeight;
    timelineCanvas.width = timelineCanvas.parentElement.clientWidth;
    timelineCanvas.height = timelineCanvas.parentElement.clientHeight;
  }
  window.addEventListener("resize", resizeCanvases);
  resizeCanvases();

  // ── Video error handling (previously missing entirely) ─────────────────
  const viewport = document.getElementById("viewportContainer");
  const videoErrorBanner = document.createElement("div");
  videoErrorBanner.style.cssText =
    "position:absolute;inset:0;display:none;align-items:center;justify-content:center;" +
    "color:#F87171;background:rgba(15,20,28,0.92);font:14px 'JetBrains Mono',monospace;" +
    "text-align:center;padding:24px;z-index:5;";
  videoErrorBanner.textContent =
    "Video failed to load. Confirm dashboard.py is running (not opening index.html directly) " +
    "and that data/real_road.mp4.f401.mp4 exists.";
  viewport.appendChild(videoErrorBanner);

  video.addEventListener("error", () => {
    console.error("[DriveSense] Video failed to load:", video.error);
    videoErrorBanner.style.display = "flex";
  });
  video.addEventListener("loadeddata", () => {
    videoErrorBanner.style.display = "none";
  });

  // ── Load pre-computed telemetry from the real pipeline run ─────────────
  fetch("/api/telemetry")
    .then(r => r.json())
    .then(data => {
      if (Array.isArray(data) && data.length > 0) {
        telemetryData = data;
        telemetryLoaded = true;
        console.log(`[DriveSense] Loaded ${telemetryData.length} real telemetry frames.`);
      } else {
        console.warn("[DriveSense] /api/telemetry empty — run pipeline.py first. Using procedural fallback.");
      }
    })
    .catch(err => {
      console.warn("[DriveSense] Could not reach /api/telemetry. Using procedural fallback.", err);
    });

  // ── Play / Pause (drives the virtual clock, not just the <video>) ──────
  btnPlayPause.addEventListener("click", () => {
    isPlaying = !isPlaying;
    if (isPlaying) {
      video.play().catch(e => console.warn("[DriveSense] video.play() rejected:", e));
      playIcon.textContent = "⏸ Pause";
      startClock();
    } else {
      video.pause();
      playIcon.textContent = "▶ Play";
      stopClock();
    }
  });

  btnReset.addEventListener("click", () => {
    currentFrameIdx = 0;
    riskHistory = [];
    try { video.currentTime = 0; } catch (e) { /* ignore if not seekable */ }
    renderFrame();
  });

  // ── Model switching now actually queries the backend ────────────────────
  modalityButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      modalityButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentModel = btn.dataset.model;

      const labels = {
        A: "Model A: Crash Baseline (Tabular Only)",
        B: "Model B: Vision Only (Kinematics)",
        C: "Model C: Fused Multi-Modal (XGBoost ★)"
      };
      activeModelPill.textContent = labels[currentModel];

      lastPredictAt = 0; // force an immediate re-prediction on switch
      renderFrame();
    });
  });

  // ── Virtual frame clock: independent of whether <video> actually plays ──
  function startClock() {
    if (clockTimer) return;
    clockTimer = setInterval(() => {
      currentFrameIdx = (currentFrameIdx + 1) % TOTAL_FRAMES;
      renderFrame();
    }, FRAME_INTERVAL_MS);
  }
  function stopClock() {
    clearInterval(clockTimer);
    clockTimer = null;
  }

  function renderFrame() {
    const frameData = getBaseFrameTelemetry(currentFrameIdx);

    drawOverlays(frameData);

    hudFrameNumber.textContent = `${currentFrameIdx} / ${TOTAL_FRAMES}`;
    hudLaneOffset.textContent =
      `${frameData.lane.lane_offset_m >= 0 ? '+' : ''}${frameData.lane.lane_offset_m.toFixed(2)} m`;

    const leadObj = frameData.objects[0];
    hudMinTtc.textContent = leadObj ? `${leadObj.ttc_sec.toFixed(1)} s` : "-- s";

    if (leadObj && leadObj.ttc_sec < 2.5) {
      hudTtcAlert.style.borderColor = "var(--accent-red)";
      hudTtcAlert.style.background = "rgba(239, 68, 68, 0.3)";
    } else {
      hudTtcAlert.style.borderColor = "rgba(255, 255, 255, 0.1)";
      hudTtcAlert.style.background = "rgba(15, 20, 28, 0.75)";
    }

    telemOffsetVal.textContent = `${frameData.lane.lane_offset_m.toFixed(2)} m`;
    const sliderPercent = Math.min(100, Math.max(0, 50 + (frameData.lane.lane_offset_m / 1.5) * 50));
    offsetThumb.style.left = `${sliderPercent}%`;

    telemDensityVal.textContent = `${frameData.scene.traffic_density} objs`;
    telemVruVal.textContent = `${Math.round(frameData.scene.vru_ratio * 100)}%`;
    telemConfVal.textContent = `${Math.round(frameData.lane.lane_confidence * 100)}%`;

    const now = performance.now();
    if (now - lastPredictAt > PREDICT_EVERY_MS) {
      lastPredictAt = now;
      requestRiskFromBackend(frameData);
    }

    drawTimelineChart();
  }

  // ── Base per-frame kinematics: real telemetry if loaded, else procedural ─
  function getBaseFrameTelemetry(idx) {
    if (telemetryLoaded && telemetryData.length > 0) {
      return telemetryData[idx % telemetryData.length];
    }
    const progress = (Math.sin(idx * 0.05) + 1.0) / 2.0;
    const leadDist = 12.0 + (1.0 - progress) * 45.0;
    const closingVel = progress > 0.6 ? 4.5 : -1.2;
    const ttc = closingVel > 0.5 ? Math.max(1.1, leadDist / closingVel) : 10.0;
    const offset = Math.sin(idx * 0.03) * 0.42;
    return {
      frame_id: idx,
      lane: { lane_offset_m: offset, lane_curvature_rad_m: 0.0012, lane_confidence: 0.92, lane_detected: true },
      objects: [{
        id: 1, class_name: "car",
        bbox: [
          Math.floor(overlayCanvas.width * (0.42 + progress * 0.15)),
          Math.floor(overlayCanvas.height * (0.55 + progress * 0.20)),
          Math.floor(overlayCanvas.width * (0.58 + progress * 0.25)),
          Math.floor(overlayCanvas.height * (0.75 + progress * 0.18))
        ],
        distance_m: leadDist, ttc_sec: ttc, approach_rate_mps: closingVel
      }],
      scene: { traffic_density: 1, vru_ratio: 0.0, luminance_mean: 145.0 }
    };
  }

  // ── The actual live call to your XGBoost models via FastAPI ────────────
  async function requestRiskFromBackend(frameData) {
    const leadObj = frameData.objects[0] || {};
    const payload = {
      model_type: currentModel,
      // NOTE: fixed demo scenario context. Wire these to real inputs/UI
      // controls (time-of-day, weather selector, etc.) as a next step.
      hour: 14, month: 6, day_week: 3, weather: 1, lgt_cond: 1,
      speed_limit_mph: 55.0,
      lane_offset_m: frameData.lane.lane_offset_m,
      lane_curvature_rad_m: frameData.lane.lane_curvature_rad_m,
      lane_confidence: frameData.lane.lane_confidence,
      min_ttc_sec: leadObj.ttc_sec ?? 10.0,
      nearest_obj_distance_m: leadObj.distance_m ?? 50.0,
      max_approach_rate_mps: leadObj.approach_rate_mps ?? 0.0,
      traffic_density: frameData.scene.traffic_density,
      vru_ratio: frameData.scene.vru_ratio
    };

    const t0 = performance.now();
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      updateRiskPanel(data, performance.now() - t0);
    } catch (err) {
      console.error("[DriveSense] /predict call failed:", err);
    }
  }

  function updateRiskPanel(data, latencyMs) {
    const pct = data.risk_score;
    riskPercentValue.textContent = `${pct}%`;
    riskSeverityBadge.textContent = `${data.risk_level} RISK`;
    riskSeverityBadge.className = `risk-badge badge-${data.risk_level.toLowerCase()}`;

    const maxCircumference = 427.25; // 2 * PI * r(68)
    gaugeProgress.style.strokeDashoffset = maxCircumference * (1 - pct / 100);

    if (data.risk_level === "HIGH") {
      gaugeProgress.style.stroke = "var(--accent-red)";
      decisionZoneVal.textContent = "Intervention Threshold";
      decisionZoneVal.className = "stat-val text-red";
    } else if (data.risk_level === "MEDIUM") {
      gaugeProgress.style.stroke = "var(--accent-yellow)";
      decisionZoneVal.textContent = "Advisory Alert Zone";
      decisionZoneVal.className = "stat-val text-yellow";
    } else {
      gaugeProgress.style.stroke = "var(--accent-green)";
      decisionZoneVal.textContent = "Safe Operating Envelope";
      decisionZoneVal.className = "stat-val text-green";
    }

    latencyVal.textContent = `${latencyMs.toFixed(1)} ms`;

    if (data.top_3_factors && data.top_3_factors.length > 0) {
      renderShapFactors(data.top_3_factors);
    }

    riskHistory.push(pct);
    if (riskHistory.length > maxHistory) riskHistory.shift();
  }

  function renderShapFactors(factors) {
    shapFactorsList.innerHTML = factors.map(f => {
      const isPositive = f.direction === "INCREASING_RISK";
      const colorClass = isPositive ? (f.shap_value > 0.25 ? "fill-red" : "fill-yellow") : "fill-green";
      const textClass = isPositive ? (f.shap_value > 0.25 ? "text-red" : "text-yellow") : "text-green";
      const barWidth = Math.min(100, Math.max(10, Math.abs(f.shap_value) * 180));
      return `
        <div class="factor-row">
          <div class="factor-header">
            <span class="factor-title">${f.human_name}</span>
            <span class="factor-impact ${textClass}">${f.shap_value >= 0 ? '+' : ''}${f.shap_value.toFixed(2)}</span>
          </div>
          <div class="factor-bar-bg"><div class="factor-bar-fill ${colorClass}" style="width: ${barWidth}%"></div></div>
          <div class="factor-desc">${f.description}</div>
        </div>`;
    }).join("");
  }

  function drawOverlays(data) {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    const w = overlayCanvas.width, h = overlayCanvas.height;

    if (chkLane.checked && data.lane.lane_detected) {
      overlayCtx.save();
      overlayCtx.beginPath();
      overlayCtx.moveTo(w * 0.12, h);
      overlayCtx.lineTo(w * 0.45, h * 0.55);
      overlayCtx.lineTo(w * 0.55, h * 0.55);
      overlayCtx.lineTo(w * 0.88, h);
      overlayCtx.closePath();
      const offsetAbs = Math.abs(data.lane.lane_offset_m);
      overlayCtx.fillStyle = offsetAbs > 0.4 ? "rgba(245, 158, 11, 0.22)" : "rgba(16, 185, 129, 0.22)";
      overlayCtx.fill();
      overlayCtx.lineWidth = 4;
      overlayCtx.strokeStyle = "rgba(239, 68, 68, 0.8)";
      overlayCtx.stroke();
      overlayCtx.restore();
    }

    if (chkBoxes.checked && data.objects) {
      data.objects.forEach(obj => {
        const [x1, y1, x2, y2] = obj.bbox;
        const ttc = obj.ttc_sec, dist = obj.distance_m;
        overlayCtx.save();
        const boxColor = ttc < 2.5 ? "#EF4444" : ttc < 4.0 ? "#F59E0B" : "#10B981";
        overlayCtx.strokeStyle = boxColor;
        overlayCtx.lineWidth = 3;
        overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const label = `ID #${obj.id} ${obj.class_name.toUpperCase()} | ${dist.toFixed(0)}m`;
        overlayCtx.font = "bold 11px 'JetBrains Mono', monospace";
        const txtWidth = overlayCtx.measureText(label).width;
        overlayCtx.fillStyle = boxColor;
        overlayCtx.fillRect(x1, Math.max(0, y1 - 20), txtWidth + 12, 20);
        overlayCtx.fillStyle = "#FFFFFF";
        overlayCtx.fillText(label, x1 + 6, Math.max(14, y1 - 5));

        if (chkTtc.checked) {
          const ttcLabel = ttc < 10.0 ? `TTC: ${ttc.toFixed(1)}s` : `STABLE`;
          overlayCtx.fillStyle = "rgba(15, 20, 28, 0.85)";
          overlayCtx.fillRect(x1, y2, 80, 18);
          overlayCtx.fillStyle = boxColor;
          overlayCtx.fillText(ttcLabel, x1 + 4, y2 + 13);
        }
        overlayCtx.restore();
      });
    }
  }

  function drawTimelineChart() {
    timelineCtx.clearRect(0, 0, timelineCanvas.width, timelineCanvas.height);
    const w = timelineCanvas.width, h = timelineCanvas.height;

    timelineCtx.strokeStyle = "rgba(239, 68, 68, 0.35)";
    timelineCtx.lineWidth = 1;
    timelineCtx.setLineDash([4, 4]);
    const yHigh = h * (1 - 0.70);
    timelineCtx.beginPath(); timelineCtx.moveTo(0, yHigh); timelineCtx.lineTo(w, yHigh); timelineCtx.stroke();

    const yMod = h * (1 - 0.45);
    timelineCtx.strokeStyle = "rgba(245, 158, 11, 0.35)";
    timelineCtx.beginPath(); timelineCtx.moveTo(0, yMod); timelineCtx.lineTo(w, yMod); timelineCtx.stroke();
    timelineCtx.setLineDash([]);

    if (riskHistory.length < 2) return;

    timelineCtx.beginPath();
    timelineCtx.lineWidth = 2.5;
    timelineCtx.strokeStyle = "#3B82F6";
    const step = w / (maxHistory - 1);
    riskHistory.forEach((val, i) => {
      const x = i * step, y = h * (1 - val / 100);
      if (i === 0) timelineCtx.moveTo(x, y); else timelineCtx.lineTo(x, y);
    });
    timelineCtx.stroke();

    const grad = timelineCtx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(59, 130, 246, 0.25)");
    grad.addColorStop(1, "rgba(59, 130, 246, 0.0)");
    timelineCtx.lineTo(w, h);
    timelineCtx.lineTo(0, h);
    timelineCtx.closePath();
    timelineCtx.fillStyle = grad;
    timelineCtx.fill();
  }

  // Draw an initial frame immediately so the dashboard isn't blank pre-Play
  renderFrame();
});