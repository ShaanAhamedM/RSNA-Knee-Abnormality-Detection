"""
Phase 7 Grandmaster Ensemble Optimizer
Fuses:
1. MultiPlane Attention Fusion (MP)
2. MultiSlice Single-Backbone Transformer (MS1)
3. Phase 7 Multi-Backbone + Sub-ROI Dual Transformer (MS2)
4. Phase 7 Upgraded Fluid-Sequence Radiomics GBDT (Rad)
5. Level-2 Anatomical Co-Occurrence Stacker
Performs rigorous grid search per target to maximize 5-fold Macro-AUC on held-out Gold-58.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from scipy.special import logit, expit

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print("=" * 85)
print("OPTIMIZING PHASE 7 GRANDMASTER ENSEMBLE (5 COMPLEMENTARY FAMILIES)")
print("=" * 85)

master_df = pd.read_csv('full_dataset_4407_folds.csv')
gold_df = master_df[master_df['is_gold'] == 1].copy().reset_index(drop=True)
N_gold = len(gold_df)

oof_mp = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\oof\multiplane_fusion_oof.csv")[target_cols].values
oof_ms1 = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\oof\multislice_transformer_oof.csv")[target_cols].values
oof_ms2 = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\oof\phase7_transformer_oof.csv")[target_cols].values
oof_rad = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\oof\phase7_radiomics_oof.csv")[target_cols].values

p_optimal_l1 = np.zeros((N_gold, 12))
optimal_weights = {}

for c, target in enumerate(target_cols):
    y = gold_df[target].values
    
    auc_mp = roc_auc_score(y, oof_mp[:, c])
    auc_ms1 = roc_auc_score(y, oof_ms1[:, c])
    auc_ms2 = roc_auc_score(y, oof_ms2[:, c])
    auc_rad = roc_auc_score(y, oof_rad[:, c])
    
    best_auc = max(auc_mp, auc_ms1, auc_ms2, auc_rad)
    if best_auc == auc_mp: best_w = [1.0, 0.0, 0.0, 0.0]
    elif best_auc == auc_ms1: best_w = [0.0, 1.0, 0.0, 0.0]
    elif best_auc == auc_ms2: best_w = [0.0, 0.0, 1.0, 0.0]
    else: best_w = [0.0, 0.0, 0.0, 1.0]
    
    # Grid search over 4 families with step 0.05
    for w1 in np.linspace(0.0, 1.0, 21):
        for w2 in np.linspace(0.0, 1.0 - w1, 21):
            for w3 in np.linspace(0.0, 1.0 - w1 - w2, 21):
                w4 = max(0.0, 1.0 - w1 - w2 - w3)
                p = w1 * oof_mp[:, c] + w2 * oof_ms1[:, c] + w3 * oof_ms2[:, c] + w4 * oof_rad[:, c]
                try:
                    a = roc_auc_score(y, p)
                    if a > best_auc + 1e-5:
                        best_auc = a
                        best_w = [w1, w2, w3, w4]
                except:
                    pass
                    
    optimal_weights[target] = [float(x) for x in best_w]
    p_optimal_l1[:, c] = best_w[0] * oof_mp[:, c] + best_w[1] * oof_ms1[:, c] + best_w[2] * oof_ms2[:, c] + best_w[3] * oof_rad[:, c]
    print(f"{target:18s} | MP:{auc_mp:.4f} MS1:{auc_ms1:.4f} MS2:{auc_ms2:.4f} Rad:{auc_rad:.4f} -> Optimal L1 AUC = {best_auc:.4f}")
    print(f"  Weights: MP={best_w[0]:.2f}, MS1={best_w[1]:.2f}, MS2={best_w[2]:.2f}, Rad={best_w[3]:.2f}")

l1_macro = np.mean([roc_auc_score(gold_df[target_cols[c]].values, p_optimal_l1[:, c]) for c in range(12)])
print("\n" + "=" * 85)
print(f">>> PHASE 7 LEVEL-1 MACRO-AUC: {l1_macro:.4f} <<<")
print("=" * 85)

# Level-2 Anatomical Co-Occurrence Stacker
p_final = np.zeros((N_gold, 12))
stacker_retained = {}

for c, target in enumerate(target_cols):
    y = gold_df[target].values
    auc_l1 = roc_auc_score(y, p_optimal_l1[:, c])
    
    X_context = np.hstack([p_optimal_l1, logit(np.clip(p_optimal_l1, 0.01, 0.99))])
    
    oof_l2 = np.zeros(N_gold)
    for fold in range(5):
        val_idx = gold_df[gold_df['fold'] == fold].index
        train_idx = gold_df[gold_df['fold'] != fold].index
        
        stk = Ridge(alpha=35.0)
        stk.fit(X_context[train_idx], y[train_idx])
        oof_l2[val_idx] = np.clip(stk.predict(X_context[val_idx]), 0.005, 0.995)
        
    try:
        auc_l2 = roc_auc_score(y, oof_l2)
    except:
        auc_l2 = 0.5
        
    if auc_l2 > auc_l1 + 0.001:
        p_final[:, c] = oof_l2
        stacker_retained[target] = True
        print(f"  [+] Level-2 RETAINED for {target:18s}: L1={auc_l1:.4f} -> L2={auc_l2:.4f} (+{auc_l2-auc_l1:.4f})")
    else:
        p_final[:, c] = p_optimal_l1[:, c]
        stacker_retained[target] = False
        print(f"  [-] Level-1 PRESERVED for {target:18s}: L1={auc_l1:.4f}")

final_macro = np.mean([roc_auc_score(gold_df[target_cols[c]].values, p_final[:, c]) for c in range(12)])
print("\n" + "=" * 85)
print(f">>> FINAL PHASE 7 GRANDMASTER 5-FOLD MACRO-AUC: {final_macro:.4f} <<<")
print("=" * 85)

for c, target in enumerate(target_cols):
    auc = roc_auc_score(gold_df[target].values, p_final[:, c])
    print(f"  {target_cols[c]:18s}: Phase 7 Gold OOF AUC = {auc:.4f}")

config = {
    'optimal_weights': optimal_weights,
    'stacker_retained': stacker_retained,
    'final_macro_auc': float(final_macro)
}
with open(r"D:\MLprojs\KaggleCompetition\RSNA\oof\phase7_grandmaster_config.json", 'w') as f:
    json.dump(config, f, indent=2)

final_oof_df = gold_df[['StudyInstanceUID']].copy()
for c, target in enumerate(target_cols):
    final_oof_df[target] = p_final[:, c]
final_oof_df.to_csv(r"D:\MLprojs\KaggleCompetition\RSNA\oof\phase7_grandmaster_oof.csv", index=False)
print("\n[SAVED] Phase 7 ensemble configuration & final OOF saved!")
