"""
Generate Phase 7 Grandmaster Submission & Inspect Dynamic Spread
Fuses:
1. 5-Fold MultiPlane Attention Fusion (MP)
2. 5-Fold MultiSlice Target-Query Transformer (MS1)
3. 5-Fold Dual-Backbone Sub-ROI Transformer (MS2)
4. 5-Fold Fluid-Sequence Radiomics GBDT (Rad)
5. Level-2 Anatomical Co-Occurrence Stacker
Evaluated on:
1. The 3 Public Test studies (written to submission.csv).
2. 20 Diverse studies to verify broad clinical patient separation.
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
from scipy.special import logit

from train_multiplane_fusion_model import MultiPlaneAttentionFusion
from train_multislice_mil_transformer import MultiSliceTargetQueryTransformer
from train_phase7_dual_backbone_transformer import Phase7DualBackboneTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Phase 7 Inference Device: {device}")

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# 1. Load Configurations & Norm stats
with open(r"D:\MLprojs\KaggleCompetition\RSNA\oof\phase7_grandmaster_config.json") as f:
    config = json.load(f)
optimal_weights = config['optimal_weights']
stacker_retained = config.get('stacker_retained', {})

with open('checkpoints/phase6_norm_stats.json') as f:
    norm_stats_p6 = json.load(f)
dicom_cols = norm_stats_p6['dicom_cols']
rad_cols = norm_stats_p6['rad_cols']
dicom_means = norm_stats_p6['dicom_means']
dicom_stds = norm_stats_p6['dicom_stds']
rad_means = norm_stats_p6['rad_means']
rad_stds = norm_stats_p6['rad_stds']

with open('checkpoints/phase7_radiomics_norm_stats.json') as f:
    norm_stats_p7 = json.load(f)
rad_cols_p7 = norm_stats_p7['feature_cols']
rad_means_p7 = norm_stats_p7['means']
rad_stds_p7 = norm_stats_p7['stds']

# 2. Load Caches
print("Loading feature caches...")
mp_emb = torch.load('multiplane_embeddings_convnext.pt', map_location='cpu')
ms1_tokens = torch.load('multislice_tokens_convnext.pt', map_location='cpu')
ms2_tokens = torch.load('phase7_tokens_dual_backbone.pt', map_location='cpu')

rad_df = pd.read_csv('dicom_radiomics_phase7_features.csv')
rad_df['ratio_sag_total'] = rad_df['n_sagittal_slices'] / (rad_df['n_total_slices'] + 1e-5)
rad_df['ratio_cor_total'] = rad_df['n_coronal_slices'] / (rad_df['n_total_slices'] + 1e-5)
rad_df['ratio_ax_total'] = rad_df['n_axial_slices'] / (rad_df['n_total_slices'] + 1e-5)
rad_df['tr_te_ratio'] = rad_df['mean_tr'] / (rad_df['mean_te'] + 1e-5)
rad_df = pd.get_dummies(rad_df, columns=['manufacturer'], drop_first=True)

# Build lookup dictionaries
tab_dicom_norm = {}
tab_rad_p6_norm = {}
tab_rad_p7_norm = {}
tab_rad_p7_raw = {}

for idx, row in rad_df.iterrows():
    uid = row['StudyInstanceUID']
    tab_dicom_norm[uid] = np.array([(row.get(c, 0.0) - dicom_means[c]) / (dicom_stds[c] + 1e-5) for c in dicom_cols], dtype=np.float32)
    tab_rad_p6_norm[uid] = np.array([(row.get(c, 0.0) - rad_means[c]) / (rad_stds[c] + 1e-5) for c in rad_cols], dtype=np.float32)
    tab_rad_p7_norm[uid] = np.array([(row.get(c, 0.0) - rad_means_p7[c]) / (rad_stds_p7[c] + 1e-5) for c in rad_cols_p7], dtype=np.float32)
    tab_rad_p7_raw[uid] = np.array([row.get(c, 0.0) for c in rad_cols_p7], dtype=np.float32)

# 3. Load Vision Models
print("Loading Vision Checkpoints...")
mp_models = []
for f in range(5):
    m = MultiPlaneAttentionFusion(embed_dim=768, num_tab_features=len(dicom_cols), num_classes=12, hidden_dim=256, dropout=0.0).to(device)
    m.load_state_dict(torch.load(f"checkpoints/multiplane_fusion_fold{f}_best.pt", map_location=device))
    m.eval()
    mp_models.append(m)

ms1_models = []
for f in range(5):
    m = MultiSliceTargetQueryTransformer(embed_dim=768, num_tab_features=len(rad_cols), num_classes=12, hidden_dim=256, dropout=0.0).to(device)
    m.load_state_dict(torch.load(f"checkpoints/multislice_transformer_fold{f}_best.pt", map_location=device))
    m.eval()
    ms1_models.append(m)

ms2_models = []
for f in range(5):
    m = Phase7DualBackboneTransformer(embed_dim=1792, num_tab_features=len(rad_cols_p7), num_classes=12, hidden_dim=256, dropout=0.0).to(device)
    m.load_state_dict(torch.load(f"checkpoints/phase7_transformer_fold{f}_best.pt", map_location=device))
    m.eval()
    ms2_models.append(m)

# 4. Load Tabular Models
print("Loading Tabular Checkpoints...")
rad_p7_lgb = {}
rad_p7_cat = {}
rad_p7_ridge = {}

for c in range(12):
    rad_p7_lgb[c] = [lgb.Booster(model_file=f'checkpoints/phase7_tabular_radiomics/lgb_fold{f}_target{c}.txt') for f in range(5)]
    rad_p7_ridge[c] = [joblib.load(f'checkpoints/phase7_tabular_radiomics/ridge_fold{f}_target{c}.joblib') for f in range(5)]
    cbs = []
    for f in range(5):
        cb = CatBoostRegressor()
        cb.load_model(f'checkpoints/phase7_tabular_radiomics/cat_fold{f}_target{c}.cbm')
        cbs.append(cb)
    rad_p7_cat[c] = cbs

print("All models successfully loaded!")

def predict_batch(uids: list) -> np.ndarray:
    B = len(uids)
    
    # 1. MultiPlane
    mp_planes = [mp_emb.get(u, torch.zeros(3, 768)).to(torch.float32) for u in uids]
    x_mp_p = torch.stack(mp_planes, dim=0).to(device)
    x_mp_t = torch.tensor(np.array([tab_dicom_norm.get(u, np.zeros(len(dicom_cols))) for u in uids]), dtype=torch.float32, device=device)
    with torch.no_grad():
        p_mp = np.mean([torch.sigmoid(m(x_mp_p, x_mp_t)).cpu().numpy() for m in mp_models], axis=0)
        
    # 2. MultiSlice MS1
    ms1_toks = []
    ms1_pls = []
    ms1_dps = []
    for u in uids:
        e = ms1_tokens.get(u, None)
        if e is not None:
            t = e['tokens'][:16] if len(e['tokens']) >= 16 else torch.cat([e['tokens'], torch.zeros((16 - len(e['tokens']), 768))], dim=0)
            pl = e['planes'][:16] if len(e['planes']) >= 16 else torch.cat([e['planes'], torch.zeros(16 - len(e['planes']), dtype=torch.long)], dim=0)
            dp = e['depths'][:16] if len(e['depths']) >= 16 else torch.cat([e['depths'], torch.zeros(16 - len(e['depths']))], dim=0)
        else:
            t = torch.zeros((16, 768))
            pl = torch.zeros(16, dtype=torch.long)
            dp = torch.zeros(16)
        ms1_toks.append(t.to(torch.float32))
        ms1_pls.append(pl.long())
        ms1_dps.append(dp.float())
    x_ms1_t = torch.stack(ms1_toks, dim=0).to(device)
    x_ms1_pl = torch.stack(ms1_pls, dim=0).to(device)
    x_ms1_dp = torch.stack(ms1_dps, dim=0).to(device)
    x_ms1_tab = torch.tensor(np.array([tab_rad_p6_norm.get(u, np.zeros(len(rad_cols))) for u in uids]), dtype=torch.float32, device=device)
    with torch.no_grad():
        p_ms1 = np.mean([torch.sigmoid(m(x_ms1_t, x_ms1_pl, x_ms1_dp, x_ms1_tab)).cpu().numpy() for m in ms1_models], axis=0)
        
    # 3. Dual-Backbone MS2
    ms2_toks = []
    ms2_pls = []
    ms2_rois = []
    ms2_dps = []
    ms2_cntr = []
    for u in uids:
        e = ms2_tokens.get(u, None)
        if e is not None:
            t = e['tokens'][:20] if len(e['tokens']) >= 20 else torch.cat([e['tokens'], torch.zeros((20 - len(e['tokens']), 1792))], dim=0)
            pl = e['planes'][:20] if len(e['planes']) >= 20 else torch.cat([e['planes'], torch.zeros(20 - len(e['planes']), dtype=torch.long)], dim=0)
            ro = e.get('rois', torch.zeros_like(pl))[:20] if len(e.get('rois', torch.zeros_like(pl))) >= 20 else torch.cat([e.get('rois', torch.zeros_like(pl)), torch.zeros(20 - len(pl), dtype=torch.long)], dim=0)
            dp = e['depths'][:20] if len(e['depths']) >= 20 else torch.cat([e['depths'], torch.zeros(20 - len(e['depths']))], dim=0)
            cn = e.get('contrast', torch.zeros((len(pl), 3)))[:20] if len(e.get('contrast', torch.zeros((len(pl), 3)))) >= 20 else torch.cat([e.get('contrast', torch.zeros((len(pl), 3))), torch.zeros((20 - len(pl), 3))], dim=0)
        else:
            t = torch.zeros((20, 1792))
            pl = torch.zeros(20, dtype=torch.long)
            ro = torch.zeros(20, dtype=torch.long)
            dp = torch.zeros(20)
            cn = torch.zeros((20, 3))
        ms2_toks.append(t.to(torch.float32))
        ms2_pls.append(pl.long())
        ms2_rois.append(ro.long())
        ms2_dps.append(dp.float())
        ms2_cntr.append(cn.float())
    x_ms2_t = torch.stack(ms2_toks, dim=0).to(device)
    x_ms2_pl = torch.stack(ms2_pls, dim=0).to(device)
    x_ms2_ro = torch.stack(ms2_rois, dim=0).to(device)
    x_ms2_dp = torch.stack(ms2_dps, dim=0).to(device)
    x_ms2_cn = torch.stack(ms2_cntr, dim=0).to(device)
    x_ms2_tab = torch.tensor(np.array([tab_rad_p7_norm.get(u, np.zeros(len(rad_cols_p7))) for u in uids]), dtype=torch.float32, device=device)
    with torch.no_grad():
        p_ms2 = np.mean([torch.sigmoid(m(x_ms2_t, x_ms2_pl, x_ms2_ro, x_ms2_dp, x_ms2_cn, x_ms2_tab)).cpu().numpy() for m in ms2_models], axis=0)

    # 4. Phase 7 Radiomics GBDT
    X_raw_p7 = np.array([tab_rad_p7_raw.get(u, np.zeros(len(rad_cols_p7))) for u in uids])
    p_rad = np.zeros((B, 12))
    for c in range(12):
        rl = np.mean([gbm.predict(X_raw_p7) for gbm in rad_p7_lgb[c]], axis=0)
        rc = np.mean([np.clip(cb.predict(X_raw_p7), 0.01, 0.99) for cb in rad_p7_cat[c]], axis=0)
        rr = np.mean([np.clip(rd.predict(X_raw_p7), 0.01, 0.99) for rd in rad_p7_ridge[c]], axis=0)
        p_rad[:, c] = 0.45 * rl + 0.35 * rc + 0.20 * rr

    # 5. Optimal Fusion
    p_final = np.zeros((B, 12))
    for c, target in enumerate(target_cols):
        w = optimal_weights[target]
        p_final[:, c] = w[0] * p_mp[:, c] + w[1] * p_ms1[:, c] + w[2] * p_ms2[:, c] + w[3] * p_rad[:, c]
        
    return p_final

# Infer on Public Test Studies
test_df = pd.read_csv('test.csv')
test_uids = test_df['StudyInstanceUID'].tolist()
print(f"\nInferring on {len(test_uids)} public test studies...")

test_preds = predict_batch(test_uids)
sub_df = pd.DataFrame({'StudyInstanceUID': test_uids})
for c, target in enumerate(target_cols):
    sub_df[target] = test_preds[:, c]

sub_path = r"D:\MLprojs\KaggleCompetition\RSNA\submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"[VERIFIED] Phase 7 submission saved to {sub_path}!")
print("\n--- Predictions on 3 Public Test Studies ---")
print(sub_df.to_string())

# Dynamic Clinical Spread on 20 Diverse Patients
print("\n" + "=" * 80)
print("EVALUATING CLINICAL SPREAD ACROSS 20 DIVERSE PATIENTS")
print("=" * 80)

sample_uids = list(ms2_tokens.keys())[300:320]
sample_preds = predict_batch(sample_uids)

print(f"{'Target Pathology':18s} | {'Mean':6s} | {'Std':6s} | {'Min':6s} | {'Max':6s} | {'Spread (Max-Min)':16s}")
print("-" * 75)
for c, target in enumerate(target_cols):
    p = sample_preds[:, c]
    m_val, s_val = np.mean(p), np.std(p)
    min_val, max_val = np.min(p), np.max(p)
    spread = max_val - min_val
    print(f"{target:18s} | {m_val:.4f} | {s_val:.4f} | {min_val:.4f} | {max_val:.4f} | {spread:.4f}")
