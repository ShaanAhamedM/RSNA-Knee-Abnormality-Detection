/**
 * KNEE MRI DIAGNOSTIC ASSISTANT - AESTHETIC FRONTEND LOGIC
 * File: demo/app.js
 */

document.addEventListener("DOMContentLoaded", () => {
  const cases = window.DEMO_CASES || [];
  let currentCaseIndex = 0;
  let isAnalyzed = false;
  let isBoxVisible = true;

  // DOM Elements
  const caseNavEl = document.getElementById("case-nav");
  const mriImageEl = document.getElementById("mri-image");
  const overlayCanvas = document.getElementById("overlay-canvas");
  const ctx = overlayCanvas.getContext("2d");
  const scanLineEl = document.getElementById("scan-line");
  const detectionTagEl = document.getElementById("detection-tag");
  const tagLabelEl = document.getElementById("tag-label");
  const tagConfEl = document.getElementById("tag-conf");

  // Meta Badges
  const caseNumBadgeEl = document.getElementById("case-num-badge");
  const casePlaneBadgeEl = document.getElementById("case-plane-badge");
  const sliceCounterEl = document.getElementById("slice-counter");

  // Findings Elements
  const findingsTitleEl = document.getElementById("findings-title");
  const stateEmptyEl = document.getElementById("state-empty");
  const stateActiveEl = document.getElementById("state-active");
  const conditionNameEl = document.getElementById("condition-name");
  const confidenceValEl = document.getElementById("confidence-val");
  const probBarFillEl = document.getElementById("prob-bar-fill");
  const triageStatusPillEl = document.getElementById("triage-status-pill");
  const explanationTextEl = document.getElementById("explanation-text");
  const patientHistoryTextEl = document.getElementById("patient-history-text");

  // Buttons
  const btnRunModel = document.getElementById("btn-run-model");
  const btnToggleBox = document.getElementById("btn-toggle-box");

  // Resize canvas to match the rendered image dimensions
  function resizeCanvas() {
    const rect = mriImageEl.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      overlayCanvas.width = rect.width;
      overlayCanvas.height = rect.height;
      if (isAnalyzed && isBoxVisible) {
        drawBoundingBox();
      }
    }
  }

  window.addEventListener("resize", resizeCanvas);
  mriImageEl.addEventListener("load", resizeCanvas);

  // Render Horizontal Case Selector Pills
  function renderCaseNavigation() {
    caseNavEl.innerHTML = "";
    cases.forEach((item, index) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `case-nav-btn ${index === currentCaseIndex ? "active" : ""}`;
      btn.innerHTML = `
        <span class="case-nav-num">${String(index + 1).padStart(2, "0")}</span>
        <span class="case-nav-title">${item.short_name}</span>
      `;
      btn.addEventListener("click", () => {
        selectCase(index);
      });
      caseNavEl.appendChild(btn);
    });
  }

  // Select a Case
  function selectCase(index) {
    if (index < 0 || index >= cases.length) return;
    currentCaseIndex = index;
    isAnalyzed = false;
    isBoxVisible = true;

    // Update active button state
    const buttons = caseNavEl.querySelectorAll(".case-nav-btn");
    buttons.forEach((btn, idx) => {
      btn.classList.toggle("active", idx === currentCaseIndex);
    });

    const c = cases[currentCaseIndex];

    // Update Badges
    caseNumBadgeEl.textContent = `Case ${String(currentCaseIndex + 1).padStart(2, "0")}`;
    casePlaneBadgeEl.textContent = `${c.plane} View`;
    sliceCounterEl.textContent = `Slice ${c.slice_index} of ${c.total_slices}`;

    // Load Scan Image
    mriImageEl.src = c.image_url;
    clearCanvas();
    detectionTagEl.style.display = "none";

    // Reset Right Findings Card
    findingsTitleEl.textContent = "Ready for Analysis";
    stateEmptyEl.style.display = "flex";
    stateActiveEl.style.display = "none";
    probBarFillEl.style.width = "0%";
  }

  // Clear Canvas Overlay
  function clearCanvas() {
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  // Draw Aesthetic Bounding Box on Canvas
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

    // If Normal Control Case
    if (c.triage_code === "NORMAL") {
      ctx.save();
      ctx.strokeStyle = "rgba(199, 160, 122, 0.4)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(x, y, w, h);
      ctx.restore();

      tagLabelEl.textContent = "Anatomically Normal Joint";
      tagConfEl.textContent = "No Lesion";
      positionTag(x + w / 2, y + h / 2, true);
      return;
    }

    ctx.save();

    // Subtle warm tinted fill
    ctx.fillStyle = "rgba(199, 160, 122, 0.08)";
    ctx.fillRect(x, y, w, h);

    // Drop shadow for clear contrast against any background
    ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
    ctx.shadowBlur = 6;

    // Elegant dual-tone border: Outer warm caramel (#C7A07A)
    ctx.strokeStyle = "#C7A07A";
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x, y, w, h);

    // Refined corner L-brackets for precision feel
    const corner = Math.max(10, Math.min(20, w * 0.2, h * 0.2));
    ctx.strokeStyle = "#FDFCE8";
    ctx.lineWidth = 3.5;

    // Top-Left
    ctx.beginPath();
    ctx.moveTo(x, y + corner);
    ctx.lineTo(x, y);
    ctx.lineTo(x + corner, y);
    ctx.stroke();

    // Top-Right
    ctx.beginPath();
    ctx.moveTo(x + w - corner, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + corner);
    ctx.stroke();

    // Bottom-Left
    ctx.beginPath();
    ctx.moveTo(x, y + h - corner);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + corner, y + h);
    ctx.stroke();

    // Bottom-Right
    ctx.beginPath();
    ctx.moveTo(x + w - corner, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w, y + h - corner);
    ctx.stroke();

    ctx.restore();

    // Position floating tag
    tagLabelEl.textContent = label;
    tagConfEl.textContent = confidence;
    positionTag(x, y, false);
  }

  // Position Bounding Box Tag
  function positionTag(x, y, isCenter) {
    const containerRect = document.getElementById("image-container").getBoundingClientRect();
    const frameRect = document.getElementById("mri-frame").getBoundingClientRect();

    const offsetX = containerRect.left - frameRect.left;
    const offsetY = containerRect.top - frameRect.top;

    let posX = offsetX + x;
    let posY = offsetY + y - 30;

    if (isCenter) {
      posX = offsetX + x - 90;
      posY = offsetY + y - 14;
    } else {
      if (posY < 8) {
        posY = offsetY + y + 10;
      }
    }

    detectionTagEl.style.left = `${Math.max(8, posX)}px`;
    detectionTagEl.style.top = `${Math.max(8, posY)}px`;
    detectionTagEl.style.display = isBoxVisible ? "flex" : "none";
  }

  // Analyze Scan Action
  function analyzeScan() {
    const c = cases[currentCaseIndex];
    if (!c) return;

    // Laser sweep animation
    scanLineEl.classList.remove("active");
    void scanLineEl.offsetWidth; // Trigger reflow
    scanLineEl.classList.add("active");

    clearCanvas();
    detectionTagEl.style.display = "none";

    // Reveal results after sweep (~450ms)
    setTimeout(() => {
      isAnalyzed = true;
      isBoxVisible = true;

      // Draw bounding box
      drawBoundingBox();

      // Show results card
      findingsTitleEl.textContent = c.short_name;
      stateEmptyEl.style.display = "none";
      stateActiveEl.style.display = "flex";

      // Condition name & confidence
      conditionNameEl.textContent = c.title.replace(/^Case \d+:\s*/, "");
      const conf = c.bounding_box ? c.bounding_box.confidence : "94%";
      confidenceValEl.textContent = conf;

      // Fill progress bar smoothly
      const pctNumber = parseFloat(conf) || 94;
      setTimeout(() => {
        probBarFillEl.style.width = `${pctNumber}%`;
      }, 50);

      // Triage status pill
      triageStatusPillEl.textContent = c.triage_tier;

      // Clinical explanation & history
      explanationTextEl.textContent = c.ai_impression;
      patientHistoryTextEl.textContent = c.clinical_history;
    }, 450);
  }

  // Toggle Bounding Box Visibility
  function toggleBoundingBox() {
    if (!isAnalyzed) return;
    isBoxVisible = !isBoxVisible;
    if (isBoxVisible) {
      drawBoundingBox();
    } else {
      clearCanvas();
      detectionTagEl.style.display = "none";
    }
  }

  // Event Listeners
  btnRunModel.addEventListener("click", analyzeScan);
  btnToggleBox.addEventListener("click", toggleBoundingBox);

  // Keyboard Shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space") {
      e.preventDefault();
      analyzeScan();
    } else if (e.code === "ArrowRight" || e.code === "ArrowDown") {
      e.preventDefault();
      const nextIdx = (currentCaseIndex + 1) % cases.length;
      selectCase(nextIdx);
    } else if (e.code === "ArrowLeft" || e.code === "ArrowUp") {
      e.preventDefault();
      const prevIdx = (currentCaseIndex - 1 + cases.length) % cases.length;
      selectCase(prevIdx);
    } else if (e.key === "b" || e.key === "B") {
      e.preventDefault();
      toggleBoundingBox();
    }
  });

  // Initial Load
  renderCaseNavigation();
  selectCase(0);
});
