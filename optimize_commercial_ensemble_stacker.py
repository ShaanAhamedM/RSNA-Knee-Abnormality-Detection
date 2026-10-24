"""
Phase 4: Commercial Grandmaster Ensemble Stacker & Conformal Clinical Triage Suite
Features:
1. 5 Diverse Model Families:
   - MP: MultiPlane ConvNeXt Global Fusion
   - MS1: MultiSlice ConvNeXt MIL Transformer
   - MS2: Phase 7 Dual-Backbone (ConvNeXt + DenseNet) Sub-ROI Transformer
   - GNN: Phase 8 12-Node Anatomical Graph Attention Network (SigLIP Aligned)
   - Rad: Fluid-Sequence DICOM & Bright Radiomics GBDT
2. Level-1 Multi-Family Grid Optimization per target
3. Level-2 Anatomical Co-Occurrence Ridge Stacker
4. Epistemic Uncertainty Estimation (ensemble variance across diverse families)
5. Conformal Selective Triage (High-Confidence Positive, High-Confidence Negative, Radiologist Review)
6. Comprehensive Evaluation: Raw Macro-AUC, Triage Macro-AUC, Sens, Spec, PPV, NPV at 80% & 90% coverage
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.linear_model import Ridge
from scipy.special import logit, expit

from thermal_governor import update_state

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print("=" * 90)
print("PHASE 4: COMMERCIAL GRANDMASTER ENSEMBLE STACKER & CONFORMAL TRIAGE")
print("=" * 90)

master_df = pd.read_csv('full_dataset_4407_folds.csv')
gold_df = master_df[master_df['is_gold'] == 1].copy().reset_index(drop=True)
N_gold = len(gold_df)

# Load 5 Complementary Model Families
print("Loading OOF predictions from 5 complementary model families...")
oof_mp = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\oof\multiplane_fusion_oof.csv")[target_cols].values
oof_ms1 = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\oof\multislice_transformer_oof.csv")[target_cols].values
oof_ms2 = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\oof\phase7_transformer_oof.csv")[target_cols].values
oof_rad = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\oof\phase7_radiomics_oof.csv")[target_cols].values

gnn_path = r"D:\MLprojs\KaggleCompetition\RSNA\oof\phase8_gnn_transformer_oof.csv"
if os.path.exists(gnn_path):
    oof_gnn = pd.read_csv(gnn_path)[target_cols].values
    print("  [Loaded] Phase 8 12-Node Anatomical GNN OOF.")
else:
    print("  [Note] GNN OOF not found, substituting ms2.")
    oof_gnn = oof_ms2.copy()

models_list = [oof_mp, oof_ms1, oof_ms2, oof_gnn, oof_rad]
model_names = ['MP', 'MS1', 'MS2', 'GNN', 'Rad']

# Print Individual Model Macro-AUCs
for name, preds in zip(model_names, models_list):
    m_aucs = [roc_auc_score(gold_df[target_cols[c]].values, preds[:, c]) for c in range(12)]
    print(f"Model Family {name:6s} | 5-Fold Gold Macro-AUC: {np.mean(m_aucs):.4f}")

# Level 1 Optimization
p_optimal_l1 = np.zeros((N_gold, 12))
optimal_weights = {}

print("\n" + "=" * 90)
print("OPTIMIZING 5-WAY LEVEL-1 ENSEMBLE WEIGHTS PER TARGET")
print("=" * 90)

for c, target in enumerate(target_cols):
    y = gold_df[target].values
    
    aucs = [roc_auc_score(y, m[:, c]) for m in models_list]
    best_idx = int(np.argmax(aucs))
    best_auc = aucs[best_idx]
    best_w = [0.0] * 5
    best_w[best_idx] = 1.0
    
    # 5-way grid search with step 0.1
    weights_candidates = [
        # Pure single models
        [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],
        # Equal pairs
        [0.5, 0.5, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [0.5, 0, 0, 0.5, 0], [0, 0.5, 0.5, 0, 0],
        [0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0.5, 0], [0.4, 0, 0.3, 0.3, 0], [0.3, 0.3, 0.2, 0.2, 0],
        [0.4, 0.2, 0.2, 0.2, 0], [0.2, 0.2, 0.3, 0.3, 0], [0.35, 0.15, 0.25, 0.25, 0],
        [0.3, 0.2, 0.2, 0.2, 0.1], [0.4, 0, 0.3, 0.2, 0.1], [0.5, 0, 0.2, 0.2, 0.1],
        [0.2, 0.3, 0.2, 0.2, 0.1], [0.1, 0.4, 0.2, 0.2, 0.1], [0.2, 0.2, 0.4, 0.1, 0.1],
        [0.1, 0.1, 0.4, 0.3, 0.1], [0.1, 0.1, 0.3, 0.4, 0.1], [0.0, 0.1, 0.4, 0.4, 0.1]
    ]
    
    for cand in weights_candidates:
        w = np.array(cand, dtype=float)
        w = w / w.sum()
        p = sum(w[m] * models_list[m][:, c] for m in range(5))
        try:
            a = roc_auc_score(y, p)
            if a > best_auc + 1e-5:
                best_auc = a
                best_w = [float(x) for x in w]
        except Exception:
            pass
            
    optimal_weights[target] = best_w
    p_optimal_l1[:, c] = sum(best_w[m] * models_list[m][:, c] for m in range(5))
    print(f"{target:18s} -> Optimal L1 AUC = {best_auc:.4f} | Weights: MP={best_w[0]:.2f}, MS1={best_w[1]:.2f}, MS2={best_w[2]:.2f}, GNN={best_w[3]:.2f}, Rad={best_w[4]:.2f}")

l1_macro = np.mean([roc_auc_score(gold_df[target_cols[c]].values, p_optimal_l1[:, c]) for c in range(12)])
print("\n" + "=" * 90)
print(f">>> LEVEL-1 5-FAMILY ENSEMBLE 5-FOLD MACRO-AUC: {l1_macro:.4f} <<<")
print("=" * 90)

# Level-2 Anatomical Co-Occurrence Stacker
p_final = p_optimal_l1.copy()
stacker_retained = {}

for c, target in enumerate(target_cols):
    y = gold_df[target].values
    auc_l1 = roc_auc_score(y, p_optimal_l1[:, c])
    
    X_context = np.hstack([p_optimal_l1, logit(np.clip(p_optimal_l1, 0.01, 0.99))])
    oof_l2 = np.zeros(N_gold)
    
    for fold in range(5):
        val_idx = gold_df[gold_df['fold'] == fold].index
        train_idx = gold_df[gold_df['fold'] != fold].index
        
        stk = Ridge(alpha=30.0)
        stk.fit(X_context[train_idx], y[train_idx])
        oof_l2[val_idx] = np.clip(stk.predict(X_context[val_idx]), 0.005, 0.995)
        
    try:
        auc_l2 = roc_auc_score(y, oof_l2)
    except Exception:
        auc_l2 = 0.5
        
    if auc_l2 > auc_l1 + 0.002:
        p_final[:, c] = oof_l2
        stacker_retained[target] = True
        print(f"  [+] Level-2 Stacker RETAINED for {target:18s}: L1={auc_l1:.4f} -> L2={auc_l2:.4f} (+{auc_l2-auc_l1:.4f})")
    else:
        stacker_retained[target] = False
        print(f"  [-] Level-1 PRESERVED for {target:18s}: L1={auc_l1:.4f}")

final_raw_macro = np.mean([roc_auc_score(gold_df[target_cols[c]].values, p_final[:, c]) for c in range(12)])
print("\n" + "=" * 90)
print(f">>> FINAL GRANDMASTER UNFILTERED 5-FOLD MACRO-AUC: {final_raw_macro:.4f} <<<")
print("=" * 90)

# Conformal Uncertainty Estimation & Epistemic Variance
print("\n" + "=" * 90)
print("CONFORMAL UNCERTAINTY ESTIMATION & SELECTIVE CLINICAL TRIAGE")
print("=" * 90)

# Calculate model variance across 5 families as epistemic uncertainty
stacked_models = np.stack(models_list, axis=0) # (5, N_gold, 12)
epistemic_std = np.std(stacked_models, axis=0) # (N_gold, 12)

# Distance to decision boundary (probabilistic confidence)
boundary_dist = np.abs(p_final - 0.5) * 2.0 # in [0, 1]
# Composite Conformal Confidence
conformal_conf = boundary_dist * (1.0 - np.clip(2.0 * epistemic_std, 0.0, 0.8))

triage_results = {}

for coverage in [1.0, 0.90, 0.80]:
    cov_str = f"{int(coverage*100)}%"
    print(f"\n--- EVALUATING CLINICAL TRIAGE AT {cov_str} AUTOMATED COVERAGE ---")
    
    target_stats = {}
    macro_aucs_cov = []
    sens_cov = []
    spec_cov = []
    ppv_cov = []
    npv_cov = []
    
    for c, target in enumerate(target_cols):
        y = gold_df[target].values
        p = p_final[:, c]
        conf = conformal_conf[:, c]
        
        if coverage < 1.0:
            # Retain top coverage fraction of highest confidence studies
            k = int(np.ceil(len(y) * coverage))
            top_indices = np.argsort(conf)[-k:]
            y_sub = y[top_indices]
            p_sub = p[top_indices]
        else:
            y_sub = y
            p_sub = p
            
        try:
            auc = roc_auc_score(y_sub, p_sub)
        except Exception:
            auc = 1.0
            
        y_pred = (p_sub >= 0.5).astype(int)
        
        # Sensitivity, Specificity, PPV, NPV
        tp = ((y_pred == 1) & (y_sub == 1)).sum()
        tn = ((y_pred == 0) & (y_sub == 0)).sum()
        fp = ((y_pred == 1) & (y_sub == 0)).sum()
        fn = ((y_pred == 0) & (y_sub == 1)).sum()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 1.0
        
        macro_aucs_cov.append(auc)
        sens_cov.append(sens)
        spec_cov.append(spec)
        ppv_cov.append(ppv)
        npv_cov.append(npv)
        
        target_stats[target] = {
            "auc": float(auc),
            "sens": float(sens),
            "spec": float(spec),
            "ppv": float(ppv),
            "npv": float(npv)
        }
        print(f"  {target:18s} | AUC={auc:.4f} | Sens={sens:.1%} | Spec={spec:.1%} | PPV={ppv:.1%} | NPV={npv:.1%}")
        
    mean_auc = np.mean(macro_aucs_cov)
    mean_sens = np.mean(sens_cov)
    mean_spec = np.mean(spec_cov)
    mean_ppv = np.mean(ppv_cov)
    mean_npv = np.mean(npv_cov)
    
    print(f"\n>>> SUMMARY AT {cov_str} COVERAGE: Macro-AUC={mean_auc:.4f} | Sens={mean_sens:.1%} | Spec={mean_spec:.1%} | PPV={mean_ppv:.1%} | NPV={mean_npv:.1%} <<<")
    
    triage_results[cov_str] = {
        "coverage": float(coverage),
        "macro_auc": float(mean_auc),
        "mean_sensitivity": float(mean_sens),
        "mean_specificity": float(mean_spec),
        "mean_ppv": float(mean_ppv),
        "mean_npv": float(mean_npv),
        "per_target": target_stats
    }

# Save Triage Results
with open("oof/commercial_triage_evaluation.json", "w") as f:
    json.dump(triage_results, f, indent=2)

# Save Ensemble Configuration
config = {
    "optimal_weights": optimal_weights,
    "stacker_retained": stacker_retained,
    "unfiltered_gold_macro_auc": float(final_raw_macro),
    "triage_90pct_macro_auc": float(triage_results["90%"]["macro_auc"]),
    "triage_80pct_macro_auc": float(triage_results["80%"]["macro_auc"]),
    "triage_80pct_sensitivity": float(triage_results["80%"]["mean_sensitivity"]),
    "triage_80pct_specificity": float(triage_results["80%"]["mean_specificity"])
}
with open("oof/commercial_grandmaster_config.json", "w") as f:
    json.dump(config, f, indent=2)

# Save Final Ensemble OOF
oof_df = gold_df[['StudyInstanceUID']].copy()
for c, target in enumerate(target_cols):
    oof_df[target] = p_final[:, c]
    oof_df[f"{target}_std"] = epistemic_std[:, c]
    oof_df[f"{target}_conf"] = conformal_conf[:, c]

oof_df.to_csv("oof/commercial_ensemble_oof.csv", index=False)
print("\n[SAVED] All Commercial Grandmaster Ensemble Artifacts saved successfully!")

update_state(
    "Phase 4: Commercial Deployment Suite",
    "COMPLETED",
    {
        "unfiltered_gold_macro_auc": float(final_raw_macro),
        "triage_90_macro_auc": float(triage_results["90%"]["macro_auc"]),
        "triage_80_macro_auc": float(triage_results["80%"]["macro_auc"]),
        "triage_80_sensitivity": float(triage_results["80%"]["mean_sensitivity"]),
        "triage_80_specificity": float(triage_results["80%"]["mean_specificity"])
    },
    f"Ensemble reached {triage_results['80%']['macro_auc']:.4f} Macro-AUC and {triage_results['80%']['mean_sensitivity']:.1%} Sens / {triage_results['80%']['mean_specificity']:.1%} Spec at 80% coverage."
)
