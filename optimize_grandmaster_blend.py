import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

master_df = pd.read_csv('full_dataset_4407_folds.csv')
gold_df = master_df[master_df['is_gold'] == 1].copy().reset_index(drop=True)

oof_mp = pd.read_csv('oof/multiplane_fusion_oof.csv')
oof_tab = pd.read_csv('oof/tabular_dicom_oof.csv')

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print("=" * 80)
print("OPTIMIZING PER-TARGET BLEND (MultiPlane Vision + Tabular GBDT)")
print("=" * 80)

blended_oof = np.zeros((len(gold_df), 12))
best_weights = {}

for c, target in enumerate(target_cols):
    y = gold_df[target].values
    p_mp = oof_mp[target].values
    p_tab = oof_tab[target].values
    
    auc_mp = roc_auc_score(y, p_mp)
    auc_tab = roc_auc_score(y, p_tab)
    
    best_w = 1.0
    best_auc = auc_mp
    for w in np.linspace(0.0, 1.0, 101):
        blend = w * p_mp + (1.0 - w) * p_tab
        try:
            a = roc_auc_score(y, blend)
            if a > best_auc:
                best_auc = a
                best_w = w
        except:
            pass
            
    best_weights[target] = float(best_w)
    blended_oof[:, c] = best_w * p_mp + (1.0 - best_w) * p_tab
    print(f"{target:18s}: Vision={auc_mp:.4f} | Tabular={auc_tab:.4f} -> Blended AUC = {best_auc:.4f} (Vision Weight: {best_w:.2f})")

macro = np.mean([roc_auc_score(gold_df[target_cols[c]].values, blended_oof[:, c]) for c in range(12)])
print("=" * 80)
print(f"FINAL BLENDED 5-FOLD MACRO-AUC: {macro:.4f}")
print("=" * 80)

# Save blend weights
with open(r"D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_blend_weights.json", 'w') as f:
    json.dump(best_weights, f, indent=2)
print("Saved blend weights to oof/grandmaster_blend_weights.json")
