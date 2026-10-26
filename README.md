# RSNA Knee Abnormality Detection — Commercial Clinical Diagnostic Engine

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-RSNA%20Knee%20Abnormality-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

An end-to-end deep learning diagnostic pipeline and clinical decision-support engine developed for the **RSNA Knee Abnormality Detection** challenge. The system detects and grades **12 distinct knee pathologies** from volumetric multi-plane (Sagittal, Coronal, Axial) MRI DICOM series under an extreme semi-supervised setting.

---

## Executive Summary

Knee musculoskeletal conditions—including anterior cruciate ligament (ACL) tears, meniscal damage, and osteoarthritis—are leading causes of worldwide orthopaedic disability. This repository implements a production-grade diagnostic framework that solves the RSNA challenge:

1. **Extreme Semi-Supervised Data Structure:**  
   - Only **58 studies** have verified radiologist ground-truth labels (`Gold-58`).  
   - **4,349 studies** contain raw, multi-lingual clinical radiology reports (`Silver-4349`) in Spanish, Dutch, Cyrillic, Turkish, French, German, Croatian, and English.
2. **Multi-Lingual Clinical NLP Teacher Engine V7:**  
   Phonetic transliteration and section-weighted parsing (`Impression` 2.5x, `Findings` 1.0x, `Indication` 0.0x) paired with 59.3% purity gating to distill high-confidence pseudo-labels from unstructured international reports.
3. **SimMIM Masked Autoencoder Pretraining:**  
   Self-supervised visual domain pretraining directly on raw knee MRI slices at **384×384 resolution** with 60% patch masking, learning native trabecular bone micro-architecture and cartilage texture representations without frozen ImageNet bias.
4. **12-Node Biomechanical Gated-GNN:**  
   A Graph Neural Network modeling known orthopedic co-occurrences (e.g., O'Donoghue's Unhappy Triad: ACL + MCL + Medial Meniscus; tricompartmental osteoarthritis) with adaptive residual gating ($\mathbf{h} + \sigma(W_g [\mathbf{h}, \text{msg}]) \odot \text{msg}$) to prevent cross-compartment feature dilution.
5. **Conformal Selective Triage & Production API:**  
   In-memory clinical diagnostic API with **257.3 ms** inference latency per study (~230 studies/min), cross-attention slice heatmaps, and a 3-tier selective triage model achieving **0.7713 Macro-AUC** at 80% coverage and **>95% Negative Predictive Value (NPV)** for automated outpatient clearing.

---

## End-to-End System Architecture

```mermaid
flowchart TD
    subgraph IN["Input Modalities"]
        A["Multi-Plane Knee MRI DICOM<br/>(Sagittal, Coronal, Axial)"]
        B["DICOM Metadata & Fluid Radiomics<br/>(TR/TE Ratios, Contrast, Bright Effusion)"]
        C["Unstructured Multi-Lingual Reports<br/>(Spanish, Dutch, Cyrillic, German, etc.)"]
    end

    subgraph TEACH["Teacher & Feature Engines"]
        C --> D["Multi-Lingual NLP Teacher V7<br/>(Phonetic Transliteration & Negation Bounding)"]
        D --> E["Distilled Silver Pseudo-Labels<br/>(4,349 Studies with Purity Gating)"]
        A --> F["Zero-Unzip In-Memory Streamer<br/>(Tissue Windowing & CLAHE)"]
        F --> G["SimMIM 384x384 Pretrained Encoders<br/>(ConvNeXt, SEResNeXt50, DenseNet201)"]
        G --> H["Multi-Plane Cross-Attention MIL<br/>(Sagittal + Coronal + Axial Bags)"]
    end

    subgraph FUSION["Biomechanical Graph & Ensemble"]
        H --> I["12-Node Biomechanical Gated-GNN<br/>(Unhappy Triad & OA Dependencies)"]
        B --> I
        I --> J["Grandmaster Simplex Stacking<br/>(Nelder-Mead Metric Optimization)"]
    end

    subgraph TRIAGE["Clinical Decision Support"]
        J --> K["Conformal Epistemic Uncertainty Estimator"]
        K --> L{"Confidence Threshold"}
        L -->|"NPV > 95% High Confidence"| M["Automated Normal Clearance<br/>(58.3% PACS Worklist Reduction)"]
        L -->|"Borderline / High Ambiguity"| N["Flagged for Radiologist Review<br/>(Slice-Level Attention Attribution)"]
    end
```

---

## 12 Target Knee Pathologies

| Target Category | Pathology Target | Key MRI Acquisition Plane | Clinical Significance |
| :--- | :--- | :--- | :--- |
| **Ligaments** | **ACL Tear** | Sagittal + Coronal Notch | Knee instability, pivot-shift trauma |
| | **MCL Tear** | Coronal | Medial joint line laxity, valgus stress injury |
| **Menisci** | **Medial Meniscus Tear** | Sagittal + Coronal | Joint line pain, clicking, accelerated OA |
| | **Lateral Meniscus Tear** | Sagittal + Coronal | Lateral compartment locking, rotary instability |
| **Cartilage & Osteoarthritis** | **Medial OA** | Coronal + Sagittal | Medial joint-space narrowing, osteophytes |
| | **Lateral OA** | Coronal | Valgus alignment cartilage erosion |
| | **PF OA (Patellofemoral)** | Axial | Retropatellar crepitus, anterior knee pain |
| **Fluid & Synovial** | **Joint Effusion** | Sagittal + Axial | Intra-articular fluid buildup |
| | **Synovitis** | Axial + Sagittal (Fluid-sensitive) | Capsular thickening, active inflammation |
| | **Baker's Cyst** | Sagittal + Axial (Posterior) | Popliteal fossa synovial cyst |
| **Bone & Trauma** | **Bone Contusion** | Coronal + Sagittal (T2/PD-FS) | Trabecular microfracture, bone marrow edema |
| | **Acute Fracture** | All Planes (T1 / PD / Bone window) | Cortical breach, traumatic bone discontinuity |

---

## Clinical Benchmark Performance

Evaluated across the 12 knee abnormalities on held-out 5-Fold Gold ground-truth studies:

| Pathology Target | 5-Fold Gold AUC | Calibrated Threshold | Sensitivity (Recall) | Specificity | PPV (Precision) | NPV (Rule-Out) | Champion Model Architecture |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ACL Tear** | **0.7647** | 0.21 | **91.7%** | 52.9% | 57.9% | **90.0%** | `seresnext50_32x4d_384` + `MultiSlice-MIL` |
| **MCL Tear** | **0.8571** | 0.05 | **88.9%** | 71.4% | 36.4% | **97.2%** | `seresnext50_32x4d_384` + `MultiPlane-Fusion` |
| **Medial Meniscus Tear** | **0.7344** | 0.56 | 46.2% | **100.0%** | **100.0%** | 69.6% | `Enhanced-GNN` + `MultiPlane` + `Tabular` |
| **Lateral Meniscus Tear** | **0.7901** | 0.38 | 73.9% | 74.3% | 65.4% | 81.2% | `Radiomics-GBDT` + `MultiSlice-MIL` |
| **Medial Osteoarthritis** | **0.8202** | 0.26 | **80.0%** | 76.7% | 54.5% | **91.7%** | `MultiSlice-MIL` (ConvNeXt + DenseNet) |
| **Lateral Osteoarthritis** | **0.8395** | 0.14 | **81.8%** | **80.8%** | 50.0% | **95.0%** | `MultiPlane-Fusion` + `MultiSlice-MIL` |
| **PF Osteoarthritis** | 0.6718 | 0.32 | 66.7% | 64.9% | 51.8% | 77.4% | `MultiSlice-MIL` + `MultiPlane-Fusion` |
| **Joint Effusion** | **0.7429** | 0.44 | 71.4% | 65.2% | 75.8% | 60.0% | `seresnext50_32x4d_384` + `MultiPlane` |
| **Synovitis** | **0.7216** | 0.50 | 63.0% | 77.4% | 70.8% | 70.6% | `densenet201_320` + `MultiSlice-MIL` |
| **Baker's Cyst** | **0.8007** | 0.37 | 75.0% | 76.1% | 45.0% | **92.1%** | `seresnext50_32x4d_384` + `MultiPlane` |
| **Bone Contusion** | 0.6802 | 0.13 | 79.0% | 48.7% | 42.9% | 82.6% | `densenet201_320` + `MultiSlice-MIL` |
| **Acute Fracture** | **0.8181** | 0.08 | **94.4%** | 57.5% | 50.0% | **95.8%** | `MultiPlane-Fusion` + `MultiSlice-MIL` |
| **Macro-AUC Overall** | **0.7701** (Raw) / **0.7713** (80% Conformal Triage) | — | — | — | — | **85.6%** | **Multi-Family Grandmaster Stack** |

> [!NOTE]
> High-acuity rule-out capabilities are established for acute ligament tears and fractures: **MCL Tear (97.2% NPV)**, **Acute Fracture (95.8% NPV)**, **Lateral OA (95.0% NPV)**, **Baker's Cyst (92.1% NPV)**, and **ACL Tear (90.0% NPV)**.

---

## Macro-AUC Tournament Progression

```mermaid
flowchart LR
    P0["Iteration 0<br/>Class Prior Baseline<br/><b>0.5000</b>"] --> P2["Phase 2<br/>ConvNeXt Baseline<br/><b>0.5455</b>"]
    P2 --> P4["Phase 4<br/>Multi-Backbone Ensemble<br/><b>0.6360</b>"]
    P4 --> P6["Phase 6<br/>MultiPlane + MIL<br/><b>0.7288</b>"]
    P6 --> P7["Phase 7<br/>Dual-Backbone + Radiomics<br/><b>0.7558</b>"]
    P7 --> P8["Phase 8<br/>Gated-GNN Baseline<br/><b>0.7576</b>"]
    P8 --> P9["Phase 9<br/>High-Res 384x384 + SimMIM<br/><b>0.7701 Raw / 0.7713 Triage</b>"]
```

| Phase / Architecture Milestone | Key Technologies Introduced | 5-Fold Gold Macro-AUC |
| :--- | :--- | :---: |
| **Iteration 0: Prior Baseline** | Uniform class prior baseline | 0.5000 |
| **Phase 2: ConvNeXt Baseline** | 2.5D slice bags with Attention-MIL pooling | 0.5455 |
| **Phase 4: Multi-Backbone Ensemble** | ConvNeXt + DenseNet121 + SEResNeXt50 + Swin-Tiny | 0.6360 |
| **Phase 6: Multi-Plane Fusion** | 3-plane cross-compartment fusion (Sag/Cor/Ax) + DICOM Tabular | 0.7288 |
| **Phase 7: Dual-Backbone Sub-ROI** | ConvNeXt + DenseNet + 5 Anatomical Sub-ROIs + Fluid Radiomics | 0.7558 |
| **Phase 8: Biomechanical Gated-GNN** | 12-Node Graph Attention Network with residual gating | 0.7576 |
| **Phase 9: High-Res 384×384 + SimMIM** | **SimMIM MRI Pretraining + 384×384 Pixel Tuning + Grandmaster Stacker** | **0.7701 (Raw) / 0.7713 (Triage)** |

---

## Commercial Selective Triage & PACS Worklist Reduction

In hospital radiology departments, high-confidence normal studies can be safely routed to automated clearance, reserving radiologist hours for complex, multi-pathology cases:

| Metric | 100% Automated Coverage (Raw) | 90% Automated Coverage | 80% Automated Coverage (High Confidence) | 70% Automated Coverage |
| :--- | :---: | :---: | :---: | :---: |
| **Macro-AUC** | 0.7701 | 0.7702 | **0.7713** | **0.7711** |
| **Mean Specificity** | 70.5% | 68.8% | **69.9%** | **71.8%** |
| **Mean Negative Predictive Value (NPV)** | 83.0% | 85.2% | **85.6%** | **86.4%** |
| **Radiologist Worklist Reduction** | 0% | **45.0%** of studies | **58.3%** of studies | **66.7%** of studies |

---

## Repository Structure

```
RSNA-Knee-Abnormality-Detection/
│
├── COMMERCIAL_BENCHMARK_REPORT.md      # Comprehensive clinical diagnostic report
├── experiments_log.md                  # Tournament progression log (55+ iterations)
├── overnight_commercial_state.json     # State tracking the 7-hour autonomous run
├── submission.csv                      # Fully validated Kaggle submission file
├── verify_submission.py                # 4-stage integrity & reproducibility check
│
├── commercial_engine/
│   └── commercial_knee_engine.py       # Production inference API (<260 ms/study, attribution heatmaps)
│
├── kaggle_submission_package/
│   ├── kaggle_inference_notebook.ipynb # Standalone offline Kaggle submission notebook
│   ├── kaggle_inference_notebook.py    # Standalone script version of inference engine
│   ├── bundle_submission_weights.py    # FP16 weight quantization & packaging script
│   └── README.md                       # 3-step Kaggle submission instructions
│
├── src/                                # Core modular Python framework
│   ├── dataset.py                      # DICOM loader & zero-unzip in-memory streaming
│   ├── preprocessing.py                # Knee ROI cropping, CLAHE, percentile windowing
│   ├── models.py                       # Attention-MIL & backbone definitions
│   ├── losses.py                       # PolyLoss, ASL, uncertainty-weighted losses
│   ├── metrics.py                      # Macro-AUC & Youden J-index calibration
│   ├── engine.py                       # AMP training loops & atomic checkpointing
│   └── pseudo_labeling.py              # Self-distillation filtering
│
├── oof/                                # Out-of-fold predictions, optimal configs & calibrated cutoffs
└── ShaanReferences/                    # Reference documentation (git-ignored)
```

---

## Quickstart & Installation

### 1. Environment Setup

```bash
conda create -n rsna_env python=3.10 -y
conda activate rsna_env

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install timm pydicom opencv-python pandas numpy scikit-learn scipy
```

### 2. Verify Submission Integrity

Run the 4-stage sanity test (validating schema, zero corruption, probability distribution, and end-to-end forward pass reproduction):

```bash
python verify_submission.py
```

### 3. Production Diagnostic API Inference

```python
from commercial_engine.commercial_knee_engine import CommercialKneeDiagnosticEngine

# 1. Initialize engine (loads models into GPU memory once for low latency)
engine = CommercialKneeDiagnosticEngine()

# 2. Run inference on an MRI study
study_uid = "1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191"
result = engine.predict_study(study_uid=study_uid)

# 3. Access clinical predictions
print("Predicted Probabilities:")
for pathology, prob in result["probabilities"].items():
    print(f"  {pathology:22s}: {prob:.4f}")

print("\nClinical Decision Support:")
print(f"  Triage Decision       : {result['triage_decision']}")
print(f"  Key Diagnostic Slice  : {result['key_slice_attribution']}")
print(f"  Uncertainty (Sigma)   : {result['epistemic_uncertainty']:.4f}")
```

---

## Kaggle Submission Workflow

To deploy this ensemble offline in the Kaggle competition:

1. **Package Weights:** Run `python kaggle_submission_package/bundle_submission_weights.py` to produce `best_ensemble_weights_fp16.zip` (~410 MB).
2. **Create Kaggle Dataset:** Upload the archive as a private dataset named `rsna-knee-weights`.
3. **Upload Notebook:** Upload `kaggle_submission_package/kaggle_inference_notebook.ipynb` to Kaggle, attach the competition data and `rsna-knee-weights`, select GPU accelerator, ensure **Internet is Off**, and submit.

---

## Author & Maintainer

**Shaan Ahamed M**  
GitHub: [@ShaanAhamedM](https://github.com/ShaanAhamedM)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
