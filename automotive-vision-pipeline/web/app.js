/**
 * DriveSense AI — Dashboard frontend
 *
 * - Overlays are driven by the <video> playback time (frame = floor(t * SRC_FPS))
 * - Telemetry bboxes are in source-pixel space and get scaled to the canvas
 * - Lane polygon comes from the detected lane lines, not a hardcoded shape
 * - Risk gauge uses the lowest-TTC object + real scene photometry
 * - No fake data: if telemetry is missing/mismatched the page says so
 */
document.addEventListener("DOMContentLoaded", () => {
  // ── Config ───────────────────────────────────────────────────────────────
  const SRC_FPS = 10;            // MUST match the ffmpeg -framerate used to build the video
  const SPEED_LIMIT_MPH = 55.0;  // same tabular prior pipeline.py uses
  const PREDICT_EVERY_MS = 500;
  const TICK_MS = 50;
  const MAX_HISTORY = 150;

  // ── DOM ──────────────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);
  const video = $("dashcamVideo");
  const viewport = $("viewportContainer");
  const overlayCanvas = $("telemetryOverlayCanvas");
  const overlayCtx = overlayCanvas.getContext("2d");
  const timelineCanvas = $("riskTimelineCanvas");
  const timelineCtx = timelineCanvas.getContext("2d");

  const btnPlayPause = $("btnPlayPause");
  const playIcon = $("playIcon");
  const btnReset = $("btnReset");

  const chkLane = $("chkLane");
  const chkBoxes = $("chkBoxes");
  const chkTtc = $("chkTtc");
  const chkHud = $("chkHud");
  const hudLayer = document.querySelector(".hud-layer");

  const hudFrameNumber = $("hudFrameNumber");
  const hudMinTtc = $("hudMinTtc");
  const hudLaneOffset = $("hudLaneOffset");
  const hudTtcAlert = $("hudTtcAlert");

  const riskPercentValue = $("riskPercentValue");
  const riskSeverityBadge = $("riskSeverityBadge");
  const gaugeProgress = $("gaugeProgress");
  const shapFactorsList = $("shapFactorsList");
  const latencyVal = $("latencyVal");
  const decisionZoneVal = $("decisionZoneVal");
  const modelConfVal = $("modelConfVal");

  const telemOffsetVal = $("telemOffsetVal");
  const offsetThumb = $("offsetThumb");
  const telemDensityVal = $("telemDensityVal");
  const telemVruVal = $("telemVruVal");
  const telemConfVal = $("telemConfVal");

  const modalityButtons = document.querySelectorAll(".btn-modality");
  const activeModelPill = $("activeModelLabel");
  $("fpsCounter").textContent = `Source: ${SRC_FPS} fps`;

  // ── State ────────────────────────────────────────────────────────────────
  let telemetry = [];
  let telemetryLoaded = false;
  let videoFailed = false;
  let currentIdx = 0;
  let riskHistory = [];
  let isPlaying = false;
  let timer = null;
  let tick = 0;
  let lastPredictAt = 0;
  let predictInFlight = false;
  let currentModel = "C";
  let scaleX = 1, scaleY = 1;

  const EMPTY_FRAME = {
    lane: {
      lane_offset_m: 0, lane_curvature_rad_m: 0, lane_confidence: 0,
      lane_detected: false, left_line: null, right_line: null
    },
    objects: [],
    scene: {
      traffic_density: 0, vru_ratio: 0, luminance_mean: 128,
      contrast_rms: 50, glare_index: 0, rain_fog_scatter_index: 0
    }
  };

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ── On-screen messages (no silent failures) ──────────────────────────────
  function makeMsg(css) {
    const el = document.createElement("div");
    el.style.cssText = css + "display:none;position:absolute;z-index:6;font:12px 'JetBrains Mono',monospace;";
    viewport.appendChild(el);
    return el;
  }
  const videoBanner = makeMsg(
    "inset:0;align-items:center;justify-content:center;color:#F87171;" +
    "background:rgba(15,20,28,0.92);text-align:center;padding:24px;");
  videoBanner.textContent =
    "Video failed to load. Confirm dashboard.py is running and that " +
    "outputs/real_road_risk_web.mp4 exists (H.264 .mp4).";
  const notice = makeMsg(
    "left:50%;bottom:52px;transform:translateX(-50%);max-width:90%;color:#FBBF24;" +
    "background:rgba(15,20,28,0.9);border:1px solid rgba(251,191,36,0.4);" +
    "padding:6px 12px;border-radius:6px;text-align:center;pointer-events:none;");

  function showNotice(text) { notice.textContent = text; notice.style.display = "block"; }
  function hideNotice() { notice.style.display = "none"; }

  // ── Frame bookkeeping ────────────────────────────────────────────────────
  function frameCount() {
    if (telemetryLoaded) return telemetry.length;
    return isFinite(video.duration) ? Math.round(video.duration * SRC_FPS) : 0;
  }
  function getFrame(idx) {
    return telemetryLoaded ? telemetry[Math.min(idx, telemetry.length - 1)] : EMPTY_FRAME;
  }
  function leadObject(objs) {
    if (!objs || !objs.length) return null;
    return objs.reduce((a, b) => (b.ttc_sec < a.ttc_sec ? b : a));
  }

  function checkSync() {
    if (!telemetryLoaded || !isFinite(video.duration) || video.duration === 0) return;
    const videoFrames = Math.round(video.duration * SRC_FPS);
    if (Math.abs(videoFrames - telemetry.length) > 2) {
      showNotice(`Telemetry has ${telemetry.length} frames but the video has ~${videoFrames}. ` +
        `Re-run pipeline.py on this same video.`);
    } else {
      hideNotice();
    }
  }

  // ── Sizing: keep canvas + video + scale in sync ──────────────────────────
  function updateScale() {
    const srcW = video.videoWidth || 1392;
    const srcH = video.videoHeight || 512;
    viewport.style.aspectRatio = `${srcW} / ${srcH}`;
    overlayCanvas.width = overlayCanvas.clientWidth;
    overlayCanvas.height = overlayCanvas.clientHeight;
    timelineCanvas.width = timelineCanvas.clientWidth;
    timelineCanvas.height = timelineCanvas.clientHeight;
    scaleX = overlayCanvas.width / srcW;
    scaleY = overlayCanvas.height / srcH;
    drawTimeline();
    renderFrame();
  }

  // ── Video events ─────────────────────────────────────────────────────────
  function onVideoError() {
    console.error("[DriveSense] Video failed to load:", video.error);
    videoFailed = true;
    videoBanner.style.display = "flex";
  }
  video.addEventListener("error", onVideoError);
  video.addEventListener("loadedmetadata", () => { updateScale(); checkSync(); });
  video.addEventListener("loadeddata", () => {
    videoFailed = false;
    videoBanner.style.display = "none";
    updateScale();
    checkSync();
  });
  window.addEventListener("resize", updateScale);
  if (video.error) onVideoError();

  // ── Telemetry ────────────────────────────────────────────────────────────
  fetch("/api/telemetry")
    .then((r) => r.json())
    .then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        telemetry = data;
        telemetryLoaded = true;
        console.log(`[DriveSense] Loaded ${telemetry.length} telemetry frames.`);
        checkSync();
      } else {
        showNotice("No telemetry found. Run pipeline.py to generate outputs/demo_telemetry.json.");
      }
      renderFrame();
    })
    .catch((err) => {
      console.warn("[DriveSense] Could not reach /api/telemetry", err);
      showNotice("Could not reach /api/telemetry. Is dashboard.py running?");
    });

  // ── Controls ─────────────────────────────────────────────────────────────
  btnPlayPause.addEventListener("click", () => {
    isPlaying = !isPlaying;
    if (isPlaying) {
      video.play().catch((e) => console.warn("[DriveSense] video.play() rejected:", e));
      playIcon.textContent = "⏸ Pause";
      startClock();
    } else {
      video.pause();
      playIcon.textContent = "▶ Play";
      stopClock();
    }
  });

  btnReset.addEventListener("click", () => {
    currentIdx = 0;
    riskHistory = [];
    try { video.currentTime = 0; } catch (e) { /* not seekable yet */ }
    drawTimeline();
    renderFrame();
  });

  [[chkLane], [chkBoxes], [chkTtc], [chkHud]].forEach(([chk]) => {
    chk.addEventListener("change", () => {
      chk.closest(".toggle-chip").classList.toggle("active", chk.checked);
      hudLayer.style.display = chkHud.checked ? "" : "none";
      renderFrame();
    });
  });

  modalityButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      modalityButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentModel = btn.dataset.model;
      activeModelPill.textContent = {
        A: "Model A: Crash Baseline (Tabular Only)",
        B: "Model B: Vision Only (Kinematics)",
        C: "Model C: Fused Multi-Modal (XGBoost ★)",
      }[currentModel];
      riskHistory = [];
      lastPredictAt = 0;
      renderFrame();
    });
  });

  // ── Clock: frame index follows the video's own playback time ─────────────
  function startClock() {
    if (timer) return;
    timer = setInterval(() => {
      tick++;
      const n = frameCount();
      if (!n) return;
      if (!videoFailed && video.readyState >= 2) {
        currentIdx = Math.min(n - 1, Math.floor(video.currentTime * SRC_FPS + 1e-3));
      } else if (tick % 2 === 0) {
        currentIdx = (currentIdx + 1) % n;   // video broken: still step through telemetry
      }
      renderFrame();
    }, TICK_MS);
  }
  function stopClock() { clearInterval(timer); timer = null; }

  // ── Per-frame render ─────────────────────────────────────────────────────
  function renderFrame() {
    const f = getFrame(currentIdx);
    const lane = f.lane || EMPTY_FRAME.lane;
    const scene = f.scene || EMPTY_FRAME.scene;

    drawOverlays(f);

    const n = frameCount();
    hudFrameNumber.textContent = `${currentIdx + 1} / ${n || "--"}`;

    const off = lane.lane_offset_m;
    hudLaneOffset.textContent = `${off >= 0 ? "+" : ""}${off.toFixed(2)} m`;

    const lead = leadObject(f.objects);
    hudMinTtc.textContent = lead ? `${lead.ttc_sec.toFixed(1)} s` : "-- s";
    if (lead && lead.ttc_sec < 2.5) {
      hudTtcAlert.style.borderColor = "var(--accent-red)";
      hudTtcAlert.style.background = "rgba(239, 68, 68, 0.3)";
    } else {
      hudTtcAlert.style.borderColor = "rgba(255, 255, 255, 0.1)";
      hudTtcAlert.style.background = "rgba(15, 20, 28, 0.75)";
    }

    telemOffsetVal.textContent = `${off.toFixed(2)} m`;
    offsetThumb.style.left = `${Math.min(100, Math.max(0, 50 + (off / 1.5) * 50))}%`;
    telemDensityVal.textContent = `${scene.traffic_density} objs`;
    telemVruVal.textContent = `${Math.round(scene.vru_ratio * 100)}%`;
    telemConfVal.textContent = `${Math.round(lane.lane_confidence * 100)}%`;

    const now = performance.now();
    if (!predictInFlight && now - lastPredictAt > PREDICT_EVERY_MS) {
      lastPredictAt = now;
      requestRisk(f);
    }
  }

  // ── Risk prediction (same features pipeline.py builds) ───────────────────
  async function requestRisk(f) {
    const objs = f.objects || [];
    const lane = f.lane || EMPTY_FRAME.lane;
    const scene = f.scene || EMPTY_FRAME.scene;

    const payload = {
      model_type: currentModel,
      hour: 14, month: 6, day_week: 3, weather: 1, lgt_cond: 1,
      speed_limit_mph: SPEED_LIMIT_MPH,
      lane_offset_m: lane.lane_offset_m,
      lane_curvature_rad_m: lane.lane_curvature_rad_m,
      lane_confidence: lane.lane_confidence,
      min_ttc_sec: objs.length ? Math.min(...objs.map((o) => o.ttc_sec)) : 10.0,
      nearest_obj_distance_m: objs.length ? Math.min(...objs.map((o) => o.distance_m)) : 60.0,
      max_approach_rate_mps: objs.length ? Math.max(...objs.map((o) => o.approach_rate_mps)) : 0.0,
      traffic_density: scene.traffic_density,
      vru_ratio: scene.vru_ratio,
      luminance_mean: scene.luminance_mean,
      contrast_rms: scene.contrast_rms,
      glare_index: scene.glare_index,
      rain_fog_scatter_index: scene.rain_fog_scatter_index,
    };

    const modelAtSend = currentModel;
    const t0 = performance.now();
    predictInFlight = true;
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      if (modelAtSend === currentModel) updateRiskPanel(data, performance.now() - t0);
    } catch (err) {
      console.error("[DriveSense] /predict call failed:", err);
    } finally {
      predictInFlight = false;
    }
  }

  function updateRiskPanel(data, latencyMs) {
    const pct = data.risk_score;               // integer 0-100
    const level = data.risk_level;             // LOW / MODERATE / HIGH / CRITICAL

    riskPercentValue.textContent = `${pct}%`;
    riskSeverityBadge.textContent = `${level} RISK`;
    riskSeverityBadge.className = `risk-badge badge-${level.toLowerCase()}`;
    gaugeProgress.style.strokeDashoffset = 427.25 * (1 - pct / 100);

    const zones = {
      CRITICAL: ["var(--accent-red)", "Critical — Immediate Intervention", "text-red"],
      HIGH: ["var(--accent-red)", "Intervention Threshold", "text-red"],
      MODERATE: ["var(--accent-yellow)", "Advisory Alert Zone", "text-yellow"],
      LOW: ["var(--accent-green)", "Safe Operating Envelope", "text-green"],
    };
    const [stroke, label, cls] = zones[level] || zones.LOW;
    gaugeProgress.style.stroke = stroke;
    decisionZoneVal.textContent = label;
    decisionZoneVal.className = `stat-val ${cls}`;

    latencyVal.textContent = `${latencyMs.toFixed(1)} ms`;
    modelConfVal.textContent = `Model ${data.model_used}${data.fallback ? " (fallback)" : ""}`;

    renderShap(data.top_3_factors);

    riskHistory.push(pct);
    if (riskHistory.length > MAX_HISTORY) riskHistory.shift();
    drawTimeline();
  }

  function renderShap(factors) {
    if (!factors || !factors.length) {
      shapFactorsList.innerHTML =
        `<div class="factor-row"><div class="factor-desc">No SHAP attribution for this prediction.</div></div>`;
      return;
    }
    const maxAbs = Math.max(...factors.map((f) => Math.abs(f.shap_value)), 1e-6);
    shapFactorsList.innerHTML = factors.map((f) => {
      const up = f.direction === "INCREASING_RISK";
      const strong = Math.abs(f.shap_value) > 0.5;
      const fill = up ? (strong ? "fill-red" : "fill-yellow") : "fill-green";
      const text = up ? (strong ? "text-red" : "text-yellow") : "text-green";
      const width = Math.max(8, (Math.abs(f.shap_value) / maxAbs) * 100);
      return `
        <div class="factor-row">
          <div class="factor-header">
            <span class="factor-title">${esc(f.human_name)}</span>
            <span class="factor-impact ${text}">${f.shap_value >= 0 ? "+" : ""}${f.shap_value.toFixed(2)}</span>
          </div>
          <div class="factor-bar-bg"><div class="factor-bar-fill ${fill}" style="width:${width}%"></div></div>
          <div class="factor-desc">${esc(f.description)}</div>
        </div>`;
    }).join("");
  }

  // ── Overlay drawing (source pixels → canvas pixels) ──────────────────────
  function drawOverlays(f) {
    const w = overlayCanvas.width, h = overlayCanvas.height;
    overlayCtx.clearRect(0, 0, w, h);
    const P = (x, y) => [x * scaleX, y * scaleY];

    const lane = f.lane;
    if (chkLane.checked && lane && lane.lane_detected && lane.left_line && lane.right_line) {
      const [[lxb, lyb], [lxt, lyt]] = lane.left_line;
      const [[rxb, ryb], [rxt, ryt]] = lane.right_line;

      overlayCtx.save();
      overlayCtx.beginPath();
      overlayCtx.moveTo(...P(lxb, lyb));
      overlayCtx.lineTo(...P(lxt, lyt));
      overlayCtx.lineTo(...P(rxt, ryt));
      overlayCtx.lineTo(...P(rxb, ryb));
      overlayCtx.closePath();
      overlayCtx.fillStyle = Math.abs(lane.lane_offset_m) > 0.4
        ? "rgba(245, 158, 11, 0.22)" : "rgba(16, 185, 129, 0.22)";
      overlayCtx.fill();

      overlayCtx.lineWidth = 4;
      overlayCtx.strokeStyle = "rgba(239, 68, 68, 0.85)";
      overlayCtx.beginPath();
      overlayCtx.moveTo(...P(lxb, lyb)); overlayCtx.lineTo(...P(lxt, lyt));
      overlayCtx.moveTo(...P(rxb, ryb)); overlayCtx.lineTo(...P(rxt, ryt));
      overlayCtx.stroke();
      overlayCtx.restore();
    }

    (f.objects || []).forEach((obj) => {
      const [bx1, by1, bx2, by2] = obj.bbox;
      const x1 = bx1 * scaleX, y1 = by1 * scaleY, x2 = bx2 * scaleX, y2 = by2 * scaleY;
      const ttc = obj.ttc_sec;
      const color = ttc < 2.5 ? "#EF4444" : ttc < 4.0 ? "#F59E0B" : "#10B981";

      overlayCtx.save();
      overlayCtx.font = "bold 11px 'JetBrains Mono', monospace";

      if (chkBoxes.checked) {
        overlayCtx.strokeStyle = color;
        overlayCtx.lineWidth = 2;
        overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const label = `#${obj.id} ${String(obj.class_name).toUpperCase()} | ${obj.distance_m.toFixed(0)}m`;
        const tw = overlayCtx.measureText(label).width;
        const ly = Math.max(0, y1 - 18);
        overlayCtx.fillStyle = color;
        overlayCtx.fillRect(x1, ly, tw + 10, 18);
        overlayCtx.fillStyle = "#FFFFFF";
        overlayCtx.fillText(label, x1 + 5, ly + 13);
      }

      if (chkTtc.checked) {
        const ttcLabel = ttc < 10.0 ? `TTC ${ttc.toFixed(1)}s` : "STABLE";
        const tw = overlayCtx.measureText(ttcLabel).width;
        overlayCtx.fillStyle = "rgba(15, 20, 28, 0.85)";
        overlayCtx.fillRect(x1, y2, tw + 10, 18);
        overlayCtx.fillStyle = color;
        overlayCtx.fillText(ttcLabel, x1 + 5, y2 + 13);
      }
      overlayCtx.restore();
    });
  }

  // ── Risk timeline ────────────────────────────────────────────────────────
  function drawTimeline() {
    const w = timelineCanvas.width, h = timelineCanvas.height;
    timelineCtx.clearRect(0, 0, w, h);

    timelineCtx.lineWidth = 1;
    timelineCtx.setLineDash([4, 4]);
    [[70, "rgba(239,68,68,0.35)"], [45, "rgba(245,158,11,0.35)"], [25, "rgba(250,204,21,0.2)"]]
      .forEach(([v, c]) => {
        const y = h * (1 - v / 100);
        timelineCtx.strokeStyle = c;
        timelineCtx.beginPath(); timelineCtx.moveTo(0, y); timelineCtx.lineTo(w, y); timelineCtx.stroke();
      });
    timelineCtx.setLineDash([]);

    if (riskHistory.length < 2) return;

    const step = w / (MAX_HISTORY - 1);
    timelineCtx.beginPath();
    riskHistory.forEach((val, i) => {
      const x = i * step, y = h * (1 - val / 100);
      if (i === 0) timelineCtx.moveTo(x, y); else timelineCtx.lineTo(x, y);
    });
    timelineCtx.lineWidth = 2.5;
    timelineCtx.strokeStyle = "#3B82F6";
    timelineCtx.stroke();

    const lastX = (riskHistory.length - 1) * step;
    timelineCtx.lineTo(lastX, h);
    timelineCtx.lineTo(0, h);
    timelineCtx.closePath();
    const grad = timelineCtx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(59, 130, 246, 0.25)");
    grad.addColorStop(1, "rgba(59, 130, 246, 0.0)");
    timelineCtx.fillStyle = grad;
    timelineCtx.fill();
  }

  // Initial layout + first frame
  updateScale();
  if (video.readyState >= 1) checkSync();
});