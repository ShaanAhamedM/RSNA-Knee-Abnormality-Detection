"""
Generate Phase 6 Grandmaster Submission & Inspect Dynamic Spread
Fuses:
1. 5-Fold MultiPlaneAttentionFusion
2. 5-Fold MultiSliceTargetQueryTransformer
3. 5-Fold Tabular DICOM GBDT
4. 5-Fold Tabular Radiomics GBDT
Using optimal per-target grid weights yielding 0.7492 Macro-AUC.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
from catboost import CatBoostRegressor

from train_multiplane_fusion_model import MultiPlaneAttentionFusion
from train_multislice_mil_transformer import MultiSliceTargetQueryTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Phase 6 Inference Device: {device}")

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# 1. Load feature stats & configurations
with open('checkpoints/phase6_norm_stats.json') as f:
    norm_stats = json.load(f)
dicom_cols = norm_stats['dicom_cols']
rad_cols = norm_stats['rad_cols']
dicom_means = norm_stats['dicom_means']
dicom_stds = norm_stats['dicom_stds']
rad_means = norm_stats['rad_means']
rad_stds = norm_stats['rad_stds']

with open(r"D:\MLprojs\KaggleCompetition\RSNA\oof\phase6_grandmaster_config.json") as f:
    config = json.load(f)
optimal_weights = config['optimal_weights']

# 2. Load cached embeddings, tokens, and tabular features
print("Loading embeddings and token databases...")
mp_emb = torch.load('multiplane_embeddings_convnext.pt', map_location='cpu')
ms_tokens = torch.load('multislice_tokens_convnext.pt', map_location='cpu')

rad_df = pd.read_csv('dicom_and_radiomics_features.csv')
rad_df['ratio_sag_total'] = rad_df['n_sagittal_slices'] / (rad_df['n_total_slices'] + 1e-5)
rad_df['ratio_cor_total'] = rad_df['n_coronal_slices'] / (rad_df['n_total_slices'] + 1e-5)
rad_df['ratio_ax_total'] = rad_df['n_axial_slices'] / (rad_df['n_total_slices'] + 1e-5)
rad_df['tr_te_ratio'] = rad_df['mean_tr'] / (rad_df['mean_te'] + 1e-5)
rad_df = pd.get_dummies(rad_df, columns=['manufacturer'], drop_first=True)

# Ensure all columns exist in rad_df
for c in rad_cols:
    if c not in rad_df.columns:
        rad_df[c] = 0.0
for c in dicom_cols:
    if c not in rad_df.columns:
        rad_df[c] = 0.0

rad_df[rad_cols] = rad_df[rad_cols].fillna(0.0)

# Build normalized lookup dictionaries
tab_dicom_norm = {}
tab_rad_norm = {}
tab_dicom_raw = {}
tab_rad_raw = {}

for idx, row in rad_df.iterrows():
    uid = row['StudyInstanceUID']
    
    # Normalized for neural networks
    v_dicom_norm = [(row[c] - dicom_means[c]) / (dicom_stds[c] + 1e-5) for c in dicom_cols]
    v_rad_norm = [(row[c] - rad_means[c]) / (rad_stds[c] + 1e-5) for c in rad_cols]
    tab_dicom_norm[uid] = np.array(v_dicom_norm, dtype=np.float32)
    tab_rad_norm[uid] = np.array(v_rad_norm, dtype=np.float32)
    
    # Raw for GBDT
    tab_dicom_raw[uid] = np.array([row[c] for c in dicom_cols], dtype=np.float32)
    tab_rad_raw[uid] = np.array([row[c] for c in rad_cols], dtype=np.float32)

# 3. Load Vision Models
print("Loading 5-Fold Vision Models...")
mp_models = []
for fold in range(5):
    m = MultiPlaneAttentionFusion(embed_dim=768, num_tab_features=len(dicom_cols), num_classes=12, hidden_dim=256, dropout=0.0).to(device)
    m.load_state_dict(torch.load(f"checkpoints/multiplane_fusion_fold{fold}_best.pt", map_location=device))
    m.eval()
    mp_models.append(m)

ms_models = []
for fold in range(5):
    m = MultiSliceTargetQueryTransformer(embed_dim=768, num_tab_features=len(rad_cols), num_classes=12, hidden_dim=256, dropout=0.0).to(device)
    m.load_state_dict(torch.load(f"checkpoints/multislice_transformer_fold{fold}_best.pt", map_location=device))
    m.eval()
    ms_models.append(m)

# 4. Load GBDT Models
print("Loading 5-Fold GBDT Checkpoints...")
dicom_lgb = {}
dicom_ridge = {}
for c in range(12):
    dicom_lgb[c] = [lgb.Booster(model_file=f'checkpoints/tabular_dicom/lgb_fold{f}_target{c}.txt') for f in range(5)]
    dicom_ridge[c] = [joblib.load(f'checkpoints/tabular_dicom/ridge_fold{f}_target{c}.joblib') for f in range(5)]

rad_lgb = {}
rad_cat = {}
rad_ridge = {}
for c in range(12):
    rad_lgb[c] = [lgb.Booster(model_file=f'checkpoints/tabular_radiomics/lgb_fold{f}_target{c}.txt') for f in range(5)]
    rad_ridge[c] = [joblib.load(f'checkpoints/tabular_radiomics/ridge_fold{f}_target{c}.joblib') for f in range(5)]
    cb_folds = []
    for f in range(5):
        cb = CatBoostRegressor()
        cb.load_model(f'checkpoints/tabular_radiomics/cat_fold{f}_target{c}.cbm')
        cb_folds.append(cb)
    rad_cat[c] = cb_folds

print("All models successfully loaded into memory!")

# 5. Core Batch Prediction Function
def predict_study_batch(uids: list) -> np.ndarray:
    B = len(uids)
    
    # 5A. MultiPlane inference
    mp_planes = []
    mp_tabs = []
    for uid in uids:
        e = mp_emb.get(uid, torch.zeros(3, 768))
        if e.dtype == torch.float16: e = e.to(torch.float32)
        mp_planes.append(e)
        mp_tabs.append(tab_dicom_norm.get(uid, np.zeros(len(dicom_cols), dtype=np.float32)))
        
    x_mp_p = torch.stack(mp_planes, dim=0).to(device)
    x_mp_t = torch.tensor(np.array(mp_tabs), dtype=torch.float32, device=device)
    
    mp_probs = []
    with torch.no_grad():
        for m in mp_models:
            mp_probs.append(torch.sigmoid(m(x_mp_p, x_mp_t)).cpu().numpy())
    p_mp = np.mean(mp_probs, axis=0) # (B, 12)
    
    # 5B. MultiSlice inference
    ms_tokens_list = []
    ms_planes_list = []
    ms_depths_list = []
    ms_tabs = []
    for uid in uids:
        entry = ms_tokens.get(uid, None)
        if entry is not None:
            t = entry['tokens']
            pl = entry['planes']
            dp = entry['depths']
            if len(t) > 16:
                t = t[:16]
                pl = pl[:16]
                dp = dp[:16]
            elif len(t) < 16:
                pad = 16 - len(t)
                t = torch.cat([t, torch.zeros((pad, 768), dtype=t.dtype)], dim=0)
                pl = torch.cat([pl, torch.zeros(pad, dtype=pl.dtype)], dim=0)
                dp = torch.cat([dp, torch.zeros(pad, dtype=dp.dtype)], dim=0)
        else:
            t = torch.zeros((16, 768), dtype=torch.float32)
            pl = torch.zeros(16, dtype=torch.long)
            dp = torch.zeros(16, dtype=torch.float32)
            
        if t.dtype == torch.float16: t = t.to(torch.float32)
        ms_tokens_list.append(t)
        ms_planes_list.append(pl.long())
        ms_depths_list.append(dp.float())
        ms_tabs.append(tab_rad_norm.get(uid, np.zeros(len(rad_cols), dtype=np.float32)))
        
    x_ms_tok = torch.stack(ms_tokens_list, dim=0).to(device)
    x_ms_pl = torch.stack(ms_planes_list, dim=0).to(device)
    x_ms_dp = torch.stack(ms_depths_list, dim=0).to(device)
    x_ms_t = torch.tensor(np.array(ms_tabs), dtype=torch.float32, device=device)
    
    ms_probs = []
    with torch.no_grad():
        for m in ms_models:
            ms_probs.append(torch.sigmoid(m(x_ms_tok, x_ms_pl, x_ms_dp, x_ms_t)).cpu().numpy())
    p_ms = np.mean(ms_probs, axis=0) # (B, 12)
    
    # 5C. GBDT Inference
    X_raw_dicom = np.array([tab_dicom_raw.get(uid, np.zeros(len(dicom_cols), dtype=np.float32)) for uid in uids])
    X_raw_rad = np.array([tab_rad_raw.get(uid, np.zeros(len(rad_cols), dtype=np.float32)) for uid in uids])
    
    p_tab = np.zeros((B, 12))
    p_rad = np.zeros((B, 12))
    
    for c in range(12):
        # DICOM 5-fold blend
        preds_lgb = np.mean([gbm.predict(X_raw_dicom) for gbm in dicom_lgb[c]], axis=0)
        preds_rd = np.mean([np.clip(rd.predict(X_raw_dicom), 0.01, 0.99) for rd in dicom_ridge[c]], axis=0)
        p_tab[:, c] = 0.65 * preds_lgb + 0.35 * preds_rd
        
        # Radiomics 5-fold blend
        r_lgb = np.mean([gbm.predict(X_raw_rad) for gbm in rad_lgb[c]], axis=0)
        r_cb = np.mean([np.clip(cb.predict(X_raw_rad), 0.01, 0.99) for cb in rad_cat[c]], axis=0)
        r_rd = np.mean([np.clip(rd.predict(X_raw_rad), 0.01, 0.99) for rd in rad_ridge[c]], axis=0)
        p_rad[:, c] = 0.45 * r_lgb + 0.35 * r_cb + 0.20 * r_rd
        
    # 5D. Per-Target Optimal Ensemble Fusion
    p_final = np.zeros((B, 12))
    for c, target in enumerate(target_cols):
        w = optimal_weights[target]
        p_final[:, c] = w[0] * p_mp[:, c] + w[1] * p_ms[:, c] + w[2] * p_tab[:, c] + w[3] * p_rad[:, c]
        
    return p_final

# 6. Generate Submission for Public Test Studies
test_df = pd.read_csv('test.csv')
test_uids = test_df['StudyInstanceUID'].tolist()
print(f"\nInferring on {len(test_uids)} public test studies...")

test_preds = predict_study_batch(test_uids)
sub_df = pd.DataFrame({'StudyInstanceUID': test_uids})
for c, target in enumerate(target_cols):
    sub_df[target] = test_preds[:, c]

sub_path = r"D:\MLprojs\KaggleCompetition\RSNA\submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"[VERIFIED] Final Phase 6 submission saved to {sub_path}!")
print("\n--- Predictions on 3 Public Test Studies ---")
print(sub_df.to_string())

# 7. Evaluate Clinical Spread on 20 Diverse Patients
print("\n" + "=" * 80)
print("EVALUATING CLINICAL SPREAD ACROSS 20 DIVERSE PATIENTS")
print("=" * 80)

diverse_uids = list(ms_tokens.keys())[200:220]
diverse_preds = predict_study_batch(diverse_uids)

print(f"{'Target Pathology':18s} | {'Mean':6s} | {'Std':6s} | {'Min':6s} | {'Max':6s} | {'Spread (Max-Min)':16s}")
print("-" * 75)
for c, target in enumerate(target_cols):
    p = diverse_preds[:, c]
    m_val, s_val = np.mean(p), np.std(p)
    min_val, max_val = np.min(p), np.max(p)
    spread = max_val - min_val
    print(f"{target:18s} | {m_val:.4f} | {s_val:.4f} | {min_val:.4f} | {max_val:.4f} | {spread:.4f}")
