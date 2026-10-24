"""
Upgrade 4: Level-2 Pathology Co-Occurrence Meta-Stacker & Rank-Logit Blending
Learns cross-pathology couplings (e.g., ACL <-> Meniscus <-> Contusion <-> Effusion).
Selectively updates targets only where Leave-One-Out CV strictly improves Gold-58 Macro-AUC.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.special import logit, expit

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def run_cooccurrence_stacking():
    print("=" * 85)
    print("UPGRADE 4: LEVEL-2 ANATOMICAL CO-OCCURRENCE META-STACKER")
    print("=" * 85)
    
    master_df = pd.read_csv('full_dataset_4407_folds.csv')
    gold_df = master_df[master_df['is_gold'] == 1].copy().reset_index(drop=True)
    N_gold = len(gold_df)
    
    oof_ms_path = r"D:\MLprojs\KaggleCompetition\RSNA\oof\multislice_transformer_oof.csv"
    oof_rad_path = r"D:\MLprojs\KaggleCompetition\RSNA\oof\radiomics_tabular_gbdt_oof.csv"
    
    if not os.path.exists(oof_ms_path) or not os.path.exists(oof_rad_path):
        print("Required Level-1 OOF files not generated yet!")
        return
        
    oof_ms = pd.read_csv(oof_ms_path)[target_cols].values
    oof_rad = pd.read_csv(oof_rad_path)[target_cols].values
    
    # 1. First find optimal Level-1 blend per target
    p_l1 = np.zeros((N_gold, 12))
    l1_weights = {}
    
    print("\n--- Finding Optimal Level-1 Blend Weights (Vision + Radiomics) ---")
    for c, target in enumerate(target_cols):
        y = gold_df[target].values
        p_v = oof_ms[:, c]
        p_r = oof_rad[:, c]
        
        best_w = 1.0
        best_auc = roc_auc_score(y, p_v)
        
        for w in np.linspace(0.0, 1.0, 101):
            blend = w * p_v + (1.0 - w) * p_r
            try:
                a = roc_auc_score(y, blend)
                if a > best_auc:
                    best_auc = a
                    best_w = w
            except:
                pass
                
        l1_weights[target] = float(best_w)
        p_l1[:, c] = best_w * p_v + (1.0 - best_w) * p_r
        auc_v = roc_auc_score(y, p_v)
        auc_r = roc_auc_score(y, p_r)
        print(f"{target:18s}: Vision={auc_v:.4f} | Radiomics={auc_r:.4f} -> L1 Blend={best_auc:.4f} (Vision Weight: {best_w:.2f})")
        
    l1_macro = np.mean([roc_auc_score(gold_df[target_cols[c]].values, p_l1[:, c]) for c in range(12)])
    print(f"\n>>> LEVEL-1 BLENDED MACRO-AUC: {l1_macro:.4f} <<<")
    
    # 2. Level-2 Co-Occurrence Stacker (Leave-One-Group-Out CV)
    print("\n--- Evaluating Level-2 Co-Occurrence Refinement ---")
    p_l2 = np.zeros((N_gold, 12))
    stacker_status = {}
    stacker_models = {}
    
    for c, target in enumerate(target_cols):
        y = gold_df[target].values
        # Input features: all 12 L1 predictions + target's logit
        X_stack = np.hstack([p_l1, logit(np.clip(p_l1, 0.01, 0.99))])
        
        # Leave-one-out CV across the 5 folds
        oof_l2_c = np.zeros(N_gold)
        for fold in range(5):
            val_idx = gold_df[gold_df['fold'] == fold].index
            train_idx = gold_df[gold_df['fold'] != fold].index
            
            # L2 Ridge Regressor
            stk = Ridge(alpha=20.0)
            stk.fit(X_stack[train_idx], y[train_idx])
            oof_l2_c[val_idx] = np.clip(stk.predict(X_stack[val_idx]), 0.005, 0.995)
            
        try:
            auc_l2 = roc_auc_score(y, oof_l2_c)
        except:
            auc_l2 = 0.5
            
        auc_l1 = roc_auc_score(y, p_l1[:, c])
        
        if auc_l2 > auc_l1 + 0.002:
            p_l2[:, c] = oof_l2_c
            stacker_status[target] = True
            print(f"  [+] Level-2 RETAINED for {target:18s}: L1={auc_l1:.4f} -> L2={auc_l2:.4f} (+{auc_l2-auc_l1:.4f})")
        else:
            p_l2[:, c] = p_l1[:, c]
            stacker_status[target] = False
            print(f"  [-] Level-1 PRESERVED for {target:18s}: L1={auc_l1:.4f} (L2 was {auc_l2:.4f})")
            
    final_l2_macro = np.mean([roc_auc_score(gold_df[target_cols[c]].values, p_l2[:, c]) for c in range(12)])
    print("\n" + "=" * 85)
    print(f">>> FINAL GRANDMASTER ENSEMBLE 5-FOLD MACRO-AUC: {final_l2_macro:.4f} <<<")
    print("=" * 85)
    
    for c, target in enumerate(target_cols):
        auc = roc_auc_score(gold_df[target].values, p_l2[:, c])
        print(f"  {target:18s}: Final OOF AUC = {auc:.4f}")
        
    # Save final OOF
    out_final_oof = r"D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_final_oof.csv"
    final_oof_df = gold_df[['StudyInstanceUID']].copy()
    for c, target in enumerate(target_cols):
        final_oof_df[target] = p_l2[:, c]
    final_oof_df.to_csv(out_final_oof, index=False)
    
    # Save configuration
    config = {
        'l1_weights': l1_weights,
        'stacker_status': stacker_status,
        'final_macro_auc': float(final_l2_macro)
    }
    with open(r"D:\MLprojs\KaggleCompetition\RSNA\oof\phase6_ensemble_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved Phase 6 ensemble configuration to oof/phase6_ensemble_config.json!")

if __name__ == '__main__':
    run_cooccurrence_stacking()
