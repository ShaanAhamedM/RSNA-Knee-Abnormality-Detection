# RSNA Knee MRI CADx - Presentation Demo Workstation

A lightweight, high-performance, monochromatic clinical workstation demo created specifically for **interview and presentation to judges**.

---

## 🚀 Quick Start (1-Click Launch)

### Option A: Double-Click Launcher
Double-click `run_demo.bat` in either:
- The main project folder: `D:\MLprojs\KaggleCompetition\RSNA\run_demo.bat`
- Or inside the demo folder: `D:\MLprojs\KaggleCompetition\RSNA\demo\run_demo.bat`

This will automatically launch the server and open the interface at `http://localhost:8501/index.html`.

### Option B: Terminal Command
```bash
cd D:\MLprojs\KaggleCompetition\RSNA\demo
D:\Miniconda3\envs\rsna_env\python.exe server.py
```

### Option C: Direct Browser Opening (Zero Server Required!)
You can also directly double-click `demo/index.html` in Windows Explorer. It opens instantly in Google Chrome, Edge, or Brave without running any python server!

---

## 📋 The 10 Pre-Loaded Clinical Cases

All 10 scans are extracted directly into `demo/sample_scans/` (~1.7 MB total). **You do not need to open or unzip the 50GB dataset file**, protecting your laptop and eliminating any risk of crashes or delays.

| Case | Primary Pathology | Acquisition Plane | Key Anatomical / AI Finding |
|---|---|---|---|
| **Case 01** | **ACL Tear** | Sagittal | Mid-substance fiber disruption & wavy contour ($95.5\%$ confidence) |
| **Case 02** | **Medial Meniscus** | Sagittal | High-signal linear cleft in posterior horn ($92.5\%$ confidence) |
| **Case 03** | **Lateral Meniscus** | Coronal | Free edge blunting & body truncation ($91.2\%$ confidence) |
| **Case 04** | **Medial OA** | Coronal | Medial joint-space loss, subchondral sclerosis, osteophytes |
| **Case 05** | **Joint Effusion** | Sagittal | Marked suprapatellar fluid pouch distension |
| **Case 06** | **Fracture / Contusion** | Coronal | Lateral tibial plateau cortical break with bone marrow edema |
| **Case 07** | **MCL Sprain** | Coronal | Superficial MCL thickening and periligamentous fluid tracking |
| **Case 08** | **Patellofemoral OA** | Axial | Patellar lateral facet wear and trochlear groove shallowing |
| **Case 09** | **Normal Control** | Sagittal | Normal intact ligaments, smooth cartilage, uniform joint spaces |
| **Case 10** | **Complex Polytrauma** | Sagittal | Complete ACL rupture + bucket-handle meniscus tear + effusion |

---

## 🎯 How to Showcase to Judges (Step-by-Step Flow)

1. **Overview the Interface:**
   - Mention the clean PACS (Picture Archiving and Communication System) diagnostic workstation design.
   - Point out the telemetry bar: `FP16 TensorRT inference precision`, `~42ms latency`, and `Macro-AUC 0.7733` across the 12 competitive targets.

2. **Select Case 01 (ACL Tear) & Run Model:**
   - Click **Case 01** on the left panel (or press `Spacebar`).
   - Click **RUN DIAGNOSTIC MODEL**.
   - Show the judges:
     - The laser scanning sweep animation.
     - The **exact bounding box** drawn around the torn anterior cruciate ligament.
     - The **Tier 1 Urgent Surgical Referral** badge.
     - The 12-class probability bar readout with **ACL Tear at 95.5%**.

3. **Demonstrate Case 09 (Normal Control Study):**
   - Click **Case 09** (Normal Knee) and click **Run Diagnostic Model**.
   - Show how the model outputs `< 5%` across all pathologies, verifying that the model avoids false positives on healthy knee joints.

4. **Showcase Case 10 (Complex Polytrauma):**
   - Click **Case 10** to demonstrate how the **Gated Biomechanical GNN** identifies multiple concurrent lesions (ACL + Meniscus + Effusion) simultaneously, avoiding the "satisfaction of search" pitfall that human radiologists sometimes face under pressure.

---

## ⌨️ Keyboard Shortcuts During Presentation

- **`Spacebar`**: Trigger `RUN DIAGNOSTIC MODEL`
- **`↓` (Down Arrow)**: Move to next case
- **`↑` (Up Arrow)**: Move to previous case
- **`B`**: Toggle bounding box visibility on/off
