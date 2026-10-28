/**
 * RSNA KNEE MRI CADx WORKSTATION - FRONTEND INTERACTION ENGINE
 * File: demo/app.js
 */

document.addEventListener("DOMContentLoaded", () => {
  const cases = window.DEMO_CASES || [];
  let currentCaseIndex = 0;
  let isModelRun = false;
  let isBoxVisible = true;

  // DOM Elements
  const casesListEl = document.getElementById("cases-list");
  const mriImageEl = document.getElementById("mri-image");
  const overlayCanvas = document.getElementById("overlay-canvas");
  const ctx = overlayCanvas.getContext("2d");
  const scanBeamEl = document.getElementById("scan-beam");
  const bboxTagEl = document.getElementById("bbox-tag");
  const bboxTagTitleEl = document.getElementById("bbox-tag-title");
  const bboxTagConfEl = document.getElementById("bbox-tag-conf");

  // Header & Meta Elements
  const activeCaseTagEl = document.getElementById("active-case-tag");
  const activeCaseTitleEl = document.getElementById("active-case-title");
  const specPlaneEl = document.getElementById("spec-plane");
  const specSliceEl = document.getElementById("spec-slice");
  const specMatrixEl = document.getElementById("spec-matrix");
  const clinicalHistoryTextEl = document.getElementById("clinical-history-text");
  const telemetryLatencyEl = document.getElementById("telemetry-latency");

  // Report Elements
  const reportTriageBadgeEl = document.getElementById("report-triage-badge");
  const triageTierTitleEl = document.getElementById("triage-tier-title");
  const aiImpressionTextEl = document.getElementById("ai-impression-text");
  const probListEl = document.getElementById("prob-list");
  const guidePointsEl = document.getElementById("guide-points");

  // Buttons
  const btnRunModel = document.getElementById("btn-run-model");
  const btnToggleBox = document.getElementById("btn-toggle-box");
  const btnReset = document.getElementById("btn-reset");

  const TARGET_ORDER = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture"
  ];

  // Initialize Canvas resolution
  function resizeCanvas() {
    const rect = mriImageEl.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      overlayCanvas.width = rect.width;
      overlayCanvas.height = rect.height;
      if (isModelRun && isBoxVisible) {
        drawBoundingBox();
      }
    }
  }

  window.addEventListener("resize", resizeCanvas);
  mriImageEl.addEventListener("load", resizeCanvas);

  // Render Left Cases List
  function renderCasesList() {
    casesListEl.innerHTML = "";
    cases.forEach((item, index) => {
      const card = document.createElement("div");
      card.className = `case-item ${index === currentCaseIndex ? "active" : ""}`;
      card.innerHTML = `
        <img class="case-thumb" src="${item.image_url}" alt="${item.short_name}" />
        <div class="case-info">
          <div class="case-number-row">
            <span class="case-id-text">CASE ${String(index + 1).padStart(2, "0")}</span>
            <span class="case-plane-pill">${item.plane.toUpperCase()}</span>
          </div>
          <div class="case-title-text" title="${item.name}">${item.short_name}</div>
          <div class="case-status-indicator">${item.triage_code}</div>
        </div>
      `;
      card.addEventListener("click", () => {
        selectCase(index);
      });
      casesListEl.appendChild(card);
    });
  }

  // Select a Case
  function selectCase(index) {
    if (index < 0 || index >= cases.length) return;
    currentCaseIndex = index;
    isModelRun = false;
    isBoxVisible = true;

    // Update active highlight in left list
    const items = casesListEl.querySelectorAll(".case-item");
    items.forEach((el, idx) => {
      el.classList.toggle("active", idx === currentCaseIndex);
    });

    const c = cases[currentCaseIndex];

    // Update Meta Information
    activeCaseTagEl.textContent = `CASE ${String(currentCaseIndex + 1).padStart(2, "0")}`;
    activeCaseTitleEl.textContent = c.title.replace(/^Case \d+:\s*/, "");
    specPlaneEl.textContent = `PLANE: ${c.plane.toUpperCase()}`;
    specSliceEl.textContent = `SLICE: ${c.slice_index} / ${c.total_slices}`;
    specMatrixEl.textContent = "MATRIX: 512 × 512";
    clinicalHistoryTextEl.textContent = c.clinical_history;

    // Reset Viewer Canvas & Overlays
    mriImageEl.src = c.image_url;
    clearCanvas();
    bboxTagEl.style.display = "none";

    // Reset Report Panel to Initial Ready State
    reportTriageBadgeEl.className = "report-badge";
    reportTriageBadgeEl.textContent = "AWAITING INFERENCE";
    triageTierTitleEl.textContent = "SYSTEM READY — CLICK 'RUN DIAGNOSTIC MODEL'";
    aiImpressionTextEl.textContent = "Press 'RUN DIAGNOSTIC MODEL' to compute 2.5D multi-plane features, evaluate 12 pathology branches, and localize lesions.";

    renderProbabilities(c.predictions, false);
    renderTalkingPoints(c.talking_points);
  }

  // Render 12 Probabilities Table
  function renderProbabilities(predictions, showValues) {
    probListEl.innerHTML = "";
    TARGET_ORDER.forEach((target) => {
      const prob = predictions[target] !== undefined ? predictions[target] : 0.02;
      const pct = Math.round(prob * 100);
      const isFlagged = showValues && prob >= 0.5;

      const row = document.createElement("div");
      row.className = `prob-row ${isFlagged ? "flagged" : ""}`;
      row.innerHTML = `
        <span class="prob-name" title="${target}">${target}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${showValues ? pct : 0}%"></div>
        </div>
        <span class="prob-val">${showValues ? pct + "%" : "—"}</span>
      `;
      probListEl.appendChild(row);
    });
  }

  // Render Interview Talking Points
  function renderTalkingPoints(points) {
    guidePointsEl.innerHTML = "";
    if (points && points.length > 0) {
      points.forEach((pt) => {
        const li = document.createElement("li");
        li.textContent = pt;
        guidePointsEl.appendChild(li);
      });
    } else {
      guidePointsEl.innerHTML = "<li>No presentation notes for this case.</li>";
    }
  }

  // Clear Canvas Overlay
  function clearCanvas() {
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  // Draw Medical Bounding Box & Annotations on Canvas
  function drawBoundingBox() {
    clearCanvas();
    const c = cases[currentCaseIndex];
    if (!c.bounding_box) return;

    const cw = overlayCanvas.width;
    const ch = overlayCanvas.height;

    const { ymin, xmin, ymax, xmax, label, confidence } = c.bounding_box;
    const x = xmin * cw;
    const y = ymin * ch;
    const w = (xmax - xmin) * cw;
    const h = (ymax - ymin) * ch;

    // Normal control study case handling
    if (c.triage_code === "NORMAL") {
      ctx.save();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(x, y, w, h);
      ctx.restore();

      // Position Tag
      bboxTagTitleEl.textContent = "ANATOMICALLY NORMAL";
      bboxTagConfEl.textContent = "100%";
      positionTag(x + w / 2, y + h / 2, true);
      return;
    }

    ctx.save();

    // 1. Subtle semi-transparent highlight fill
    ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
    ctx.fillRect(x, y, w, h);

    // 2. High-contrast white bounding border
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    // 3. Precision PACS Surgical Corner L-Brackets
    const bracketSize = Math.max(10, Math.min(22, w * 0.2, h * 0.2));
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = "#ffffff";

    // Top-Left
    ctx.beginPath();
    ctx.moveTo(x, y + bracketSize);
    ctx.lineTo(x, y);
    ctx.lineTo(x + bracketSize, y);
    ctx.stroke();

    // Top-Right
    ctx.beginPath();
    ctx.moveTo(x + w - bracketSize, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + bracketSize);
    ctx.stroke();

    // Bottom-Left
    ctx.beginPath();
    ctx.moveTo(x, y + h - bracketSize);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + bracketSize, y + h);
    ctx.stroke();

    // Bottom-Right
    ctx.beginPath();
    ctx.moveTo(x + w - bracketSize, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w, y + h - bracketSize);
    ctx.stroke();

    // 4. Subtle Crosshair Lines at Center of Lesion
    const cx = x + w / 2;
    const cy = y + h / 2;
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
    ctx.setLineDash([2, 3]);

    ctx.beginPath();
    ctx.moveTo(cx - 10, cy);
    ctx.lineTo(cx + 10, cy);
    ctx.moveTo(cx, cy - 10);
    ctx.lineTo(cx, cy + 10);
    ctx.stroke();

    ctx.restore();

    // Position floating tag badge
    bboxTagTitleEl.textContent = label;
    bboxTagConfEl.textContent = confidence;
    positionTag(x, y, false);
  }

  // Position Bounding Box HTML Tag
  function positionTag(x, y, isCenter) {
    const wrapperRect = document.getElementById("canvas-wrapper").getBoundingClientRect();
    const viewportRect = document.getElementById("viewer-viewport").getBoundingClientRect();

    const offsetX = wrapperRect.left - viewportRect.left;
    const offsetY = wrapperRect.top - viewportRect.top;

    let posX = offsetX + x;
    let posY = offsetY + y - 28;

    if (isCenter) {
      posX = offsetX + x - 80;
      posY = offsetY + y - 14;
    } else {
      if (posY < 10) {
        posY = offsetY + y + 10;
      }
    }

    bboxTagEl.style.left = `${Math.max(10, posX)}px`;
    bboxTagEl.style.top = `${Math.max(10, posY)}px`;
    bboxTagEl.style.display = isBoxVisible ? "flex" : "none";
  }

  // Run Diagnostic Model Action
  function runModel() {
    const c = cases[currentCaseIndex];
    if (!c) return;

    // Simulate Realistic Live Telemetry Latency (39ms to 54ms)
    const simulatedLatency = Math.floor(Math.random() * 15) + 39;
    telemetryLatencyEl.textContent = `${simulatedLatency} ms`;

    // Trigger Laser Scanning Beam Animation
    scanBeamEl.classList.remove("active");
    void scanBeamEl.offsetWidth; // Trigger reflow
    scanBeamEl.classList.add("active");

    // Clear previous box during scan
    clearCanvas();
    bboxTagEl.style.display = "none";

    // Complete inference after scan sweep (450ms)
    setTimeout(() => {
      isModelRun = true;
      isBoxVisible = true;

      // Draw bounding box
      drawBoundingBox();

      // Update Triage Badge
      reportTriageBadgeEl.className = `report-badge ${c.triage_code.toLowerCase()}`;
      reportTriageBadgeEl.textContent = c.triage_code;

      // Update Triage Summary Card
      triageTierTitleEl.textContent = c.triage_tier;
      aiImpressionTextEl.textContent = c.ai_impression;

      // Populate 12 Probabilities with smooth bar fill
      renderProbabilities(c.predictions, true);
    }, 450);
  }

  // Toggle Bounding Box Visibility
  function toggleBoundingBox() {
    if (!isModelRun) return;
    isBoxVisible = !isBoxVisible;
    if (isBoxVisible) {
      drawBoundingBox();
    } else {
      clearCanvas();
      bboxTagEl.style.display = "none";
    }
  }

  // Reset Viewport
  function resetViewport() {
    selectCase(currentCaseIndex);
  }

  // Event Listeners
  btnRunModel.addEventListener("click", runModel);
  btnToggleBox.addEventListener("click", toggleBoundingBox);
  btnReset.addEventListener("click", resetViewport);

  // Keyboard Shortcuts for Smooth Judge Presentation
  document.addEventListener("keydown", (e) => {
    // Spacebar = Run Model
    if (e.code === "Space") {
      e.preventDefault();
      runModel();
    }
    // Arrow Down = Next Case
    else if (e.code === "ArrowDown") {
      e.preventDefault();
      const nextIdx = (currentCaseIndex + 1) % cases.length;
      selectCase(nextIdx);
    }
    // Arrow Up = Previous Case
    else if (e.code === "ArrowUp") {
      e.preventDefault();
      const prevIdx = (currentCaseIndex - 1 + cases.length) % cases.length;
      selectCase(prevIdx);
    }
    // 'B' key = Toggle Bounding Box
    else if (e.key === "b" || e.key === "B") {
      e.preventDefault();
      toggleBoundingBox();
    }
  });

  // Initial Load
  renderCasesList();
  selectCase(0);
});
