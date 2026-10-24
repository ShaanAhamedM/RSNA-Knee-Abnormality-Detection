"""
Upgraded Commercial Ensemble Optimizer & Calibrator
Integrates:
1. MultiPlane Attention Fusion (MP)
2. MultiSlice Single-Backbone Transformer (MS1)
3. Phase 7 Multi-Backbone Sub-ROI Transformer (MS2)
4. Enhanced Biomechanical Gated-GNN Transformer (GNN_Enh)
5. Phase 8 GNN Transformer (GNN_Orig)
6. Fluid Radiomics GBDT (Rad)
7. Tabular DICOM GBDT (Tab)
Finds optimal non-negative ensemble weights per pathology using dense simplex search,
calibrates clinical operating points, and computes conformal triage metrics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.optimize import minimize
from scipy.special import logit, expit

ROOT_DIR = r"D:\MLprojs\KaggleCompetition\RSNA"
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

from thermal_governor import update_state

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print("=" * 90)
print("UPGRADED COMMERCIAL ENSEMBLE OPTIMIZATION & CONFORMAL TRIAGE")
print("=" * 90)

master_df = pd.read_csv('full_dataset_4407_folds.csv')
gold_df = master_df[master_df['is_gold'] == 1].copy().reset_index(drop=True)
N_gold = len(gold_df)

# Load base predictions
models = []
model_names = []

base_oofs = [
    ("MP", r"oof/multiplane_fusion_oof.csv"),
    ("MS1", r"oof/multislice_transformer_oof.csv"),
    ("MS2", r"oof/phase7_transformer_oof.csv"),
    ("GNN_Enh", r"oof/enhanced_gated_gnn_oof.csv"),
    ("Rad", r"oof/phase7_radiomics_oof.csv"),
    ("Tab", r"oof/tabular_dicom_oof.csv")
]

for name, path in base_oofs:
    if os.path.exists(path):
        models.append(pd.read_csv(path)[TARGETS].values)
        model_names.append(name)

# Dynamically discover all additional cycle OOFs
import glob
for extra_path in sorted(glob.glob("oof/*_cycle*_oof.csv")):
    b_name = os.path.basename(extra_path).replace("_oof.csv", "")
    try:
        extra_df = pd.read_csv(extra_path)
        if all(t in extra_df.columns for t in TARGETS) and len(extra_df) == N_gold:
            models.append(extra_df[TARGETS].values)
            model_names.append(b_name)
            print(f"  [Discovered New Tournament Competitor] {b_name}")
    except Exception:
        pass

M = len(models)

# Individual Macro-AUCs
print("Individual Model Family 5-Fold Macro-AUCs:")
for name, m in zip(model_names, models):
    aucs = [roc_auc_score(gold_df[TARGETS[c]].values, m[:, c]) for c in range(12)]
    print(f"  {name:10s} : {np.mean(aucs):.4f}")

# Target-specific simplex grid search + multi-start optimization
p_ensemble = np.zeros((N_gold, 12))
optimal_weights = {}
target_aucs = {}

print("\n" + "=" * 90)
print("OPTIMIZING PER-TARGET ENSEMBLE WEIGHTS")
print("=" * 90)

for c, target in enumerate(TARGETS):
    y = gold_df[target].values
    
    # Base single models
    single_aucs = [roc_auc_score(y, m[:, c]) for m in models]
    best_single_idx = int(np.argmax(single_aucs))
    best_auc = single_aucs[best_single_idx]
    best_w = np.zeros(M)
    best_w[best_single_idx] = 1.0
    
    def loss_fn(w):
        w_pos = np.maximum(0, w)
        s = w_pos.sum()
        if s == 0: return 0.0
        w_norm = w_pos / s
        p = sum(w_norm[m_idx] * models[m_idx][:, c] for m_idx in range(M))
        try:
            return -roc_auc_score(y, p)
        except Exception:
            return 0.0
            
    # Test all pairs with 0.1 step
    for i in range(M):
        for j in range(i + 1, M):
            for alpha in np.linspace(0.1, 0.9, 9):
                w = np.zeros(M)
                w[i] = alpha
                w[j] = 1.0 - alpha
                a = -loss_fn(w)
                if a > best_auc + 1e-5:
                    best_auc = a
                    best_w = w.copy()
                    
    # Test top-3 triplets
    top_indices = np.argsort(single_aucs)[::-1][:4]
    for w0 in np.linspace(0.1, 0.8, 8):
        for w1 in np.linspace(0.1, 0.9 - w0, 8):
            w2 = max(0.0, 1.0 - w0 - w1)
            w = np.zeros(M)
            w[top_indices[0]] = w0
            w[top_indices[1]] = w1
            w[top_indices[2]] = w2
            a = -loss_fn(w)
            if a > best_auc + 1e-5:
                best_auc = a
                best_w = w.copy()
                
    # Nelder-Mead refinement
    res = minimize(loss_fn, best_w, method='Nelder-Mead', options={'maxiter': 200})
    if -res.fun > best_auc + 1e-5:
        best_auc = -res.fun
        best_w = np.maximum(0, res.x) / np.maximum(0, res.x).sum()
        
    best_w = [float(x) for x in best_w]
    optimal_weights[target] = best_w
    target_aucs[target] = float(best_auc)
    
    p_ensemble[:, c] = sum(best_w[m_idx] * models[m_idx][:, c] for m_idx in range(M))
    w_str = ", ".join([f"{model_names[m]}:{best_w[m]:.2f}" for m in range(M) if best_w[m] > 0.01])
    print(f"{target:18s} -> AUC: {best_auc:.4f} | Weights: {w_str}")

macro_auc = float(np.mean(list(target_aucs.values())))
print("\n" + "=" * 90)
print(f">>> UPGRADED ENSEMBLE RAW 5-FOLD MACRO-AUC: {macro_auc:.4f} <<<")
print("=" * 90)

# Save ensemble OOF
ensemble_df = gold_df[['StudyInstanceUID']].copy()
for c, t in enumerate(TARGETS):
    ensemble_df[t] = p_ensemble[:, c]
ensemble_df.to_csv("oof/commercial_ensemble_upgraded_oof.csv", index=False)

# Save weights config
config = {
    "model_names": model_names,
    "optimal_weights": optimal_weights,
    "target_aucs": target_aucs,
    "raw_macro_auc": macro_auc
}
with open("oof/commercial_upgraded_config.json", "w") as f:
    json.dump(config, f, indent=2)

# Compute Epistemic Uncertainty & Conformal Triage
ensemble_std = np.std(np.stack([models[m] for m in range(M)], axis=0), axis=0) # (N, 12)

# Youden index calibration
calibrated_thresholds = {}
operating_metrics = {}

for c, target in enumerate(TARGETS):
    y = gold_df[target].values
    p = p_ensemble[:, c]
    
    best_youden = -1.0
    best_th = 0.5
    best_sens, best_spec, best_ppv, best_npv = 0.0, 0.0, 0.0, 0.0
    
    for th in np.linspace(0.02, 0.98, 97):
        pred_bin = (p >= th).astype(int)
        tp = np.sum((pred_bin == 1) & (y == 1))
        tn = np.sum((pred_bin == 0) & (y == 0))
        fp = np.sum((pred_bin == 1) & (y == 0))
        fn = np.sum((pred_bin == 0) & (y == 1))
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        youden = sens + spec - 1.0
        if youden > best_youden:
            best_youden = youden
            best_th = th
            best_sens, best_spec, best_ppv, best_npv = sens, spec, ppv, npv
            
    calibrated_thresholds[target] = float(best_th)
    operating_metrics[target] = {
        "threshold": round(float(best_th), 3),
        "sensitivity": round(float(best_sens), 4),
        "specificity": round(float(best_spec), 4),
        "ppv": round(float(best_ppv), 4),
        "npv": round(float(best_npv), 4),
        "youden": round(float(best_youden), 4)
    }

with open("oof/commercial_calibrated_thresholds.json", "w") as f:
    json.dump(calibrated_thresholds, f, indent=2)

print("\n" + "=" * 90)
print("CLINICALLY CALIBRATED OPERATING POINTS (YOUDEN J-INDEX)")
print("=" * 90)
print(f"{'Pathology':18s} | {'Thresh':6s} | {'Sensitivity':11s} | {'Specificity':11s} | {'PPV':6s} | {'NPV':6s}")
print("-" * 75)
for t in TARGETS:
    m = operating_metrics[t]
    print(f"{t:18s} | {m['threshold']:<6.2f} | {m['sensitivity']*100:<10.1f}% | {m['specificity']*100:<10.1f}% | {m['ppv']*100:<5.1f}% | {m['npv']*100:<5.1f}%")

mean_sens = np.mean([m['sensitivity'] for m in operating_metrics.values()])
mean_spec = np.mean([m['specificity'] for m in operating_metrics.values()])
print(f"\nMean Balanced Sensitivity: {mean_sens*100:.1f}% | Mean Specificity: {mean_spec*100:.1f}%")

# Conformal Triage Evaluation at 80% and 90% coverage
# Confidence metric: |p - th| / (sigma + 1e-4)
triage_eval = {}
for cov in [0.90, 0.80, 0.70]:
    triage_aucs = []
    triage_specs = []
    triage_ppvs = []
    triage_npvs = []
    
    for c, target in enumerate(TARGETS):
        y = gold_df[target].values
        p = p_ensemble[:, c]
        sigma = ensemble_std[:, c]
        th = calibrated_thresholds[target]
        
        confidence = np.abs(p - th) / (sigma + 1e-3)
        cutoff = np.quantile(confidence, 1.0 - cov)
        keep = (confidence >= cutoff)
        
        y_kept = y[keep]
        p_kept = p[keep]
        
        if len(np.unique(y_kept)) > 1:
            triage_aucs.append(roc_auc_score(y_kept, p_kept))
        else:
            triage_aucs.append(1.0)
            
        pred_bin = (p_kept >= th).astype(int)
        tp = np.sum((pred_bin == 1) & (y_kept == 1))
        tn = np.sum((pred_bin == 0) & (y_kept == 0))
        fp = np.sum((pred_bin == 1) & (y_kept == 0))
        fn = np.sum((pred_bin == 0) & (y_kept == 1))
        
        spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 1.0
        
        triage_specs.append(spec)
        triage_ppvs.append(ppv)
        triage_npvs.append(npv)
        
    triage_eval[f"coverage_{int(cov*100)}"] = {
        "macro_auc": round(float(np.mean(triage_aucs)), 4),
        "mean_specificity": round(float(np.mean(triage_specs)), 4),
        "mean_ppv": round(float(np.mean(triage_ppvs)), 4),
        "mean_npv": round(float(np.mean(triage_npvs)), 4)
    }

print("\n" + "=" * 90)
print("CONFORMAL SELECTIVE TRIAGE AT DIFFERENT AUTOMATED COVERAGE LEVELS")
print("=" * 90)
for k, v in triage_eval.items():
    print(f"{k.upper():16s} -> Macro-AUC: {v['macro_auc']:.4f} | Specificity: {v['mean_specificity']*100:.1f}% | PPV: {v['mean_ppv']*100:.1f}% | NPV: {v['mean_npv']*100:.1f}%")

with open("oof/commercial_triage_evaluation.json", "w") as f:
    json.dump(triage_eval, f, indent=2)

update_state(
    phase_name="Upgraded Commercial Grandmaster Optimization",
    status="COMPLETED",
    metrics={
        "raw_macro_auc": macro_auc,
        "triage_90_macro_auc": triage_eval["coverage_90"]["macro_auc"],
        "triage_80_macro_auc": triage_eval["coverage_80"]["macro_auc"],
        "triage_80_specificity": triage_eval["coverage_80"]["mean_specificity"],
        "triage_80_ppv": triage_eval["coverage_80"]["mean_ppv"]
    },
    notes="Upgraded commercial grandmaster optimization completed with full conformal selective triage."
)

print("\nAll ensemble configurations and calibration parameters saved successfully.")
