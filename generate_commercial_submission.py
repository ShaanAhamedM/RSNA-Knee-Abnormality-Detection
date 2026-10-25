"""
Generate Commercial Grandmaster Submission
Fuses 5 Complementary Model Families:
1. MultiPlane Attention Fusion (MP)
2. MultiSlice Single-Backbone Transformer (MS1)
3. Phase 7 Dual-Backbone Sub-ROI Transformer (MS2)
4. Phase 8 12-Node Anatomical Co-Occurrence GNN (SigLIP Aligned)
5. Fluid-Sequence DICOM & Bright Radiomics GBDT (Rad)
Writes verified submission to:
- RSNA root submission.csv
- kaggle_submission_package/submission.csv
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

ROOT_DIR = r"D:\MLprojs\KaggleCompetition\RSNA"
os.chdir(ROOT_DIR)

from train_multiplane_fusion_model import MultiPlaneAttentionFusion
from train_multislice_mil_transformer import MultiSliceTargetQueryTransformer
from train_phase7_dual_backbone_transformer import Phase7DualBackboneTransformer
from train_commercial_gnn_transformer import CommercialAnatomicalGNNTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Commercial Submission Inference Device: {device}")

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# 1. Load Configurations
with open(r"D:\MLprojs\KaggleCompetition\RSNA\oof\commercial_grandmaster_config.json") as f:
    config = json.load(f)
optimal_weights = config['optimal_weights']
stacker_retained = config.get('stacker_retained', {})

with open('checkpoints/phase7_radiomics_norm_stats.json') as f:
    norm_stats_p7 = json.load(f)
rad_cols_p7 = norm_stats_p7['feature_cols']
rad_means_p7 = norm_stats_p7['means']
rad_stds_p7 = norm_stats_p7['stds']

with open('checkpoints/phase6_norm_stats.json') as f:
    norm_stats_p6 = json.load(f)
dicom_cols = norm_stats_p6['dicom_cols']
rad_cols_p6 = norm_stats_p6['rad_cols']
dicom_means = norm_stats_p6['dicom_means']
dicom_stds = norm_stats_p6['dicom_stds']
rad_means_p6 = norm_stats_p6['rad_means']
rad_stds_p6 = norm_stats_p6['rad_stds']

# 2. Load Feature Caches
print("Loading feature caches...")
mp_emb = torch.load('multiplane_embeddings_convnext.pt', map_location='cpu', weights_only=False)
ms1_tokens = torch.load('multislice_tokens_convnext.pt', map_location='cpu', weights_only=False)
ms2_tokens = torch.load('phase7_tokens_dual_backbone.pt', map_location='cpu', weights_only=False)

rad_df = pd.read_csv('dicom_radiomics_phase7_features.csv')
rad_df['ratio_sag_total'] = rad_df['n_sagittal_slices'] / (rad_df['n_total_slices'] + 1e-5)
rad_df['ratio_cor_total'] = rad_df['n_coronal_slices'] / (rad_df['n_total_slices'] + 1e-5)
rad_df['ratio_ax_total'] = rad_df['n_axial_slices'] / (rad_df['n_total_slices'] + 1e-5)
rad_df['tr_te_ratio'] = rad_df['mean_tr'] / (rad_df['mean_te'] + 1e-5)
rad_df = pd.get_dummies(rad_df, columns=['manufacturer'], drop_first=True)

gnn_feature_cols = [c for c in rad_cols_p7 if c != 'sample_weight']

tab_rad_p7_norm = {}
tab_rad_p7_raw = {}
tab_dicom_norm = {}
tab_rad_p6_norm = {}
tab_gnn_norm = {}

for idx, row in rad_df.iterrows():
    uid = row['StudyInstanceUID']
    tab_rad_p7_norm[uid] = np.array([(row.get(c, 0.0) - rad_means_p7[c]) / (rad_stds_p7[c] + 1e-5) for c in rad_cols_p7], dtype=np.float32)
    tab_rad_p7_raw[uid] = np.array([row.get(c, 0.0) for c in rad_cols_p7], dtype=np.float32)
    tab_dicom_norm[uid] = np.array([(row.get(c, 0.0) - dicom_means[c]) / (dicom_stds[c] + 1e-5) for c in dicom_cols], dtype=np.float32)
    tab_rad_p6_norm[uid] = np.array([(row.get(c, 0.0) - rad_means_p6[c]) / (rad_stds_p6[c] + 1e-5) for c in rad_cols_p6], dtype=np.float32)
    tab_gnn_norm[uid] = np.array([(row.get(c, 0.0) - rad_means_p7[c]) / (rad_stds_p7[c] + 1e-5) for c in gnn_feature_cols], dtype=np.float32)

# Load test studies
test_df = pd.read_csv('test.csv')
test_uids = test_df['StudyInstanceUID'].unique()
print(f"Test Studies to predict: {len(test_uids)}")

# Load Vision Models
mp_models = []
for f in range(5):
    m = MultiPlaneAttentionFusion(embed_dim=768, num_tab_features=len(dicom_cols), num_classes=12, hidden_dim=256, dropout=0.0).to(device)
    m.load_state_dict(torch.load(f"checkpoints/multiplane_fusion_fold{f}_best.pt", map_location=device))
    m.eval()
    mp_models.append(m)

ms1_models = []
for f in range(5):
    m = MultiSliceTargetQueryTransformer(embed_dim=768, num_tab_features=len(rad_cols_p6), num_classes=12, hidden_dim=256, dropout=0.0).to(device)
    m.load_state_dict(torch.load(f"checkpoints/multislice_transformer_fold{f}_best.pt", map_location=device))
    m.eval()
    ms1_models.append(m)

ms2_models = []
for f in range(5):
    m = Phase7DualBackboneTransformer(embed_dim=1792, num_tab_features=len(rad_cols_p7), num_classes=12, hidden_dim=256, dropout=0.0).to(device)
    m.load_state_dict(torch.load(f"checkpoints/phase7_transformer_fold{f}_best.pt", map_location=device))
    m.eval()
    ms2_models.append(m)

gnn_feature_cols = [c for c in rad_cols_p7 if c != 'sample_weight']

gnn_models = []
for f in range(5):
    m = CommercialAnatomicalGNNTransformer(visual_dim=1792, tab_dim=len(gnn_feature_cols), hidden_dim=256, num_classes=12).to(device)
    m.load_state_dict(torch.load(f"checkpoints/phase8_gnn_fold{f}_best.pt", map_location=device))
    m.eval()
    gnn_models.append(m)

# Load Tabular Models
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

# Predict Test Studies
test_preds = {}

for uid in test_uids:
    tb_d = torch.tensor(tab_dicom_norm[uid]).unsqueeze(0).to(device)
    tb_r6 = torch.tensor(tab_rad_p6_norm[uid]).unsqueeze(0).to(device)
    tb_r7 = torch.tensor(tab_rad_p7_norm[uid]).unsqueeze(0).to(device)
    
    # 1. MP
    emb = mp_emb[uid].unsqueeze(0).to(device).float()
    p_mp = torch.sigmoid(torch.stack([m(emb, tb_d) for m in mp_models]).mean(dim=0)).detach().cpu().numpy()[0]
    
    # 2. MS1
    t1 = ms1_tokens[uid]['tokens'].unsqueeze(0).to(device).float()
    pl1 = ms1_tokens[uid]['planes'].unsqueeze(0).to(device).long()
    dp1 = ms1_tokens[uid]['depths'].unsqueeze(0).to(device).float()
    p_ms1 = torch.sigmoid(torch.stack([m(t1, pl1, dp1, tb_r6) for m in ms1_models]).mean(dim=0)).detach().cpu().numpy()[0]
    
    # 3. MS2
    t2 = ms2_tokens[uid]['tokens'].unsqueeze(0).to(device).float()
    pl2 = ms2_tokens[uid]['planes'].unsqueeze(0).to(device).long()
    ro2 = ms2_tokens[uid]['rois'].unsqueeze(0).to(device).long()
    dp2 = ms2_tokens[uid]['depths'].unsqueeze(0).to(device).float()
    ct2 = ms2_tokens[uid]['contrast'].unsqueeze(0).to(device).float()
    p_ms2 = torch.sigmoid(torch.stack([m(t2, pl2, ro2, dp2, ct2, tb_r7) for m in ms2_models]).mean(dim=0)).detach().cpu().numpy()[0]
    
    # 4. GNN
    tb_gnn = torch.tensor(tab_gnn_norm[uid]).unsqueeze(0).to(device)
    dp2_sq = dp2.unsqueeze(-1)
    p_gnn = torch.sigmoid(torch.stack([m(t2, pl2, ro2, dp2_sq, ct2, tb_gnn) for m in gnn_models]).mean(dim=0)).detach().cpu().numpy()[0]
    
    # 5. Rad GBDT
    x_raw = tab_rad_p7_raw[uid].reshape(1, -1)
    x_norm = tab_rad_p7_norm[uid].reshape(1, -1)
    p_rad = np.zeros(12)
    for c in range(12):
        p_l = np.mean([mod.predict(x_raw)[0] for mod in rad_p7_lgb[c]])
        p_c = np.mean([mod.predict(x_raw)[0] for mod in rad_p7_cat[c]])
        p_r = np.mean([mod.predict(x_norm)[0] for mod in rad_p7_ridge[c]])
        p_rad[c] = np.clip(0.45 * p_l + 0.35 * p_c + 0.20 * p_r, 0.01, 0.99)
        
    # Blend using optimal weights
    p_study = np.zeros(12)
    for c, target in enumerate(target_cols):
        w = optimal_weights[target]
        p_study[c] = w[0] * p_mp[c] + w[1] * p_ms1[c] + w[2] * p_ms2[c] + w[3] * p_gnn[c] + w[4] * p_rad[c]
        
    test_preds[uid] = np.clip(p_study, 0.001, 0.999)

# Build submission dataframe
rows = []
for uid in test_uids:
    p = test_preds[uid]
    row = {'StudyInstanceUID': uid}
    for c, target in enumerate(target_cols):
        row[target] = float(p[c])
    rows.append(row)

sub_df = pd.DataFrame(rows)
print("\nCommercial Ensemble Submission Preview:")
print(sub_df)

# Write to root submission.csv and kaggle_submission_package/submission.csv
sub_df.to_csv("submission.csv", index=False)
sub_df.to_csv("kaggle_submission_package/submission.csv", index=False)
print("\n[SAVED] submission.csv updated in both root and kaggle_submission_package/")
