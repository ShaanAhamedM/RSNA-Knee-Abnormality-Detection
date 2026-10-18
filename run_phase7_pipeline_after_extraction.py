"""
Phase 7 End-to-End Orchestrator:
Executes sequentially once feature extraction completes:
1. Upgraded 50-Feature Fluid-Sequence Radiomics GBDTs (train_phase7_radiomics_gbdt.py)
2. 5-Fold Dual-Backbone Multi-ROI Cross-Attention Transformer (train_phase7_dual_backbone_transformer.py)
3. 5-Family Grandmaster Grid Ensemble Optimizer + Level-2 Stacker (optimize_phase7_grandmaster_ensemble.py)
4. Phase 7 Test Submission & Dynamic Clinical Spread Verification (generate_phase7_submission.py)
5. Offline Kaggle Package Bundler (bundle_phase7_package.py)
6. Bulletproof 4/4 Verification Sanity Suite (verify_submission.py)
"""

import os
import sys
import time
import subprocess

PYTHON_EXE = r"D:\Miniconda3\envs\rsna_env\python.exe"
CWD = r"D:\MLprojs\KaggleCompetition\RSNA"

steps = [
    ("Step 1/6: Training Phase 7 Radiomics & Sequence GBDTs", "train_phase7_radiomics_gbdt.py", CWD),
    ("Step 2/6: Training Phase 7 Dual-Backbone Transformer", "train_phase7_dual_backbone_transformer.py", CWD),
    ("Step 3/6: Optimizing 5-Family Grandmaster Ensemble", "optimize_phase7_grandmaster_ensemble.py", CWD),
    ("Step 4/6: Generating Final Phase 7 Submission", "generate_phase7_submission.py", CWD),
    ("Step 5/6: Bundling Kaggle Offline Submission Package", os.path.join("kaggle_submission_package", "bundle_phase7_package.py"), os.path.join(CWD, "kaggle_submission_package")),
    ("Step 6/6: Running Bulletproof Verification Sanity Suite", "verify_submission.py", CWD)
]

print("=" * 85)
print("STARTING PHASE 7 POST-EXTRACTION AUTOMATED PIPELINE")
print("=" * 85)

for title, script, run_cwd in steps:
    print(f"\n{'='*85}\n>>> {title} <<<\n{'='*85}", flush=True)
    cmd = [PYTHON_EXE, "-u", script]
    t0 = time.time()
    res = subprocess.run(cmd, cwd=run_cwd)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[ERROR] {script} failed with exit code {res.returncode}! Aborting pipeline.", flush=True)
        sys.exit(res.returncode)
    print(f"[COMPLETED] {title} in {dt:.1f}s", flush=True)

print("\n" + "=" * 85)
print(">>> ALL PHASE 7 PIPELINE STEPS COMPLETED SUCCESSFULLY! <<<")
print("=" * 85)
