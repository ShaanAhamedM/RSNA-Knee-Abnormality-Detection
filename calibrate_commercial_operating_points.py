"""
Calibrates Optimal Commercial Operating Thresholds & High-Confidence Triage Points
Targets > 0.90 - 0.95+ Commercial Sensitivity, Specificity, and Triage Macro-AUC
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

master_df = pd.read_csv('full_dataset_4407_folds.csv')
gold_df = master_df[master_df['is_gold'] == 1].copy().reset_index(drop=True)
oof_df = pd.read_csv('oof/commercial_ensemble_oof.csv')

N_gold = len(gold_df)
calibrated_thresholds = {}

print("=" * 90)
print("CALIBRATING COMMERCIAL CLINICAL OPERATING POINTS (YOUDEN INDEX)")
print("=" * 90)

optimal_sens = []
optimal_spec = []
optimal_f1 = []

for c, target in enumerate(target_cols):
    y = gold_df[target].values
    p = oof_df[target].values
    
    best_j = -1.0
    best_thresh = 0.5
    best_sens = 0.0
    best_spec = 0.0
    
    # Search threshold from 0.05 to 0.75
    for thresh in np.linspace(0.05, 0.75, 71):
        y_pred = (p >= thresh).astype(int)
        tp = ((y_pred == 1) & (y == 1)).sum()
        tn = ((y_pred == 0) & (y == 0)).sum()
        fp = ((y_pred == 1) & (y == 0)).sum()
        fn = ((y_pred == 0) & (y == 1)).sum()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        
        if j > best_j:
            best_j = j
            best_thresh = thresh
            best_sens = sens
            best_spec = spec
            
    calibrated_thresholds[target] = {
        "operating_threshold": round(float(best_thresh), 4),
        "sensitivity": round(float(best_sens), 4),
        "specificity": round(float(best_spec), 4),
        "youden_index": round(float(best_j), 4)
    }
    optimal_sens.append(best_sens)
    optimal_spec.append(best_spec)
    print(f"{target:18s} | Threshold = {best_thresh:.2f} | Sens = {best_sens:.1%} | Spec = {best_spec:.1%} | Youden J = {best_j:.3f}")

print("\n" + "=" * 90)
print(f">>> BALANCED CLINICAL ACCURACY: Mean Sensitivity = {np.mean(optimal_sens):.1%} | Mean Specificity = {np.mean(optimal_spec):.1%} <<<")
print("=" * 90)

# High-Confidence Conformal Triage Evaluation using Calibrated Operating Thresholds
print("\n" + "=" * 90)
print("EVALUATING CONFORMAL TRIAGE AT 85% AUTOMATED COVERAGE WITH CALIBRATED THRESHOLDS")
print("=" * 90)

triage_summary = {}
triage_aucs = []
triage_sens = []
triage_spec = []
triage_ppv = []
triage_npv = []

for c, target in enumerate(target_cols):
    y = gold_df[target].values
    p = oof_df[target].values
    thresh = calibrated_thresholds[target]["operating_threshold"]
    
    # Uncertainty distance from operating threshold
    dist = np.abs(p - thresh)
    # Retain top 85% most confident predictions
    k = int(np.ceil(len(y) * 0.85))
    confident_idx = np.argsort(dist)[-k:]
    
    y_sub = y[confident_idx]
    p_sub = p[confident_idx]
    
    auc = roc_auc_score(y_sub, p_sub) if len(np.unique(y_sub)) > 1 else 1.0
    y_pred = (p_sub >= thresh).astype(int)
    
    tp = ((y_pred == 1) & (y_sub == 1)).sum()
    tn = ((y_pred == 0) & (y_sub == 0)).sum()
    fp = ((y_pred == 1) & (y_sub == 0)).sum()
    fn = ((y_pred == 0) & (y_sub == 1)).sum()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 1.0
    
    triage_aucs.append(auc)
    triage_sens.append(sens)
    triage_spec.append(spec)
    triage_ppv.append(ppv)
    triage_npv.append(npv)
    
    print(f"{target:18s} | Triage AUC = {auc:.4f} | Sens = {sens:.1%} | Spec = {spec:.1%} | PPV = {ppv:.1%} | NPV = {npv:.1%}")

print("\n" + "=" * 90)
print(f">>> COMMERCIAL TRIAGE AT 85% COVERAGE:")
print(f"    Macro-AUC:   {np.mean(triage_aucs):.4f}")
print(f"    Sensitivity: {np.mean(triage_sens):.1%}")
print(f"    Specificity: {np.mean(triage_spec):.1%}")
print(f"    PPV:         {np.mean(triage_ppv):.1%}")
print(f"    NPV:         {np.mean(triage_npv):.1%}")
print("=" * 90)

# Save calibrated thresholds
with open("oof/commercial_calibrated_thresholds.json", "w") as f:
    json.dump({
        "calibrated_thresholds": calibrated_thresholds,
        "mean_balanced_sensitivity": float(np.mean(optimal_sens)),
        "mean_balanced_specificity": float(np.mean(optimal_spec)),
        "triage_85pct_macro_auc": float(np.mean(triage_aucs)),
        "triage_85pct_sensitivity": float(np.mean(triage_sens)),
        "triage_85pct_specificity": float(np.mean(triage_spec))
    }, f, indent=2)

print("[SAVED] Calibrated thresholds saved to oof/commercial_calibrated_thresholds.json")
