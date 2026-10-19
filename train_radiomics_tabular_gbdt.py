"""
Upgrade 3: Upgraded DICOM & Radiomics GBDT Ensemble (LightGBM + CatBoost + Ridge)
Trained on 45 tabular & radiomic features across all 4,407 studies.
Evaluated on honest 5-fold cross-validation on held-out Gold-58 studies.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print("=" * 85)
print("UPGRADE 3: TRAINING DICOM + RADIOMICS GBDT (45 FEATURES)")
print("=" * 85)

master_df = pd.read_csv('full_dataset_4407_folds.csv')
rad_tab_path = 'dicom_and_radiomics_features.csv'

if not os.path.exists(rad_tab_path):
    print(f"File {rad_tab_path} not ready yet!")
    exit(0)

rad_tab_df = pd.read_csv(rad_tab_path)
df = pd.merge(master_df, rad_tab_df, on='StudyInstanceUID', how='left')

# Feature engineering
df['ratio_sag_total'] = df['n_sagittal_slices'] / (df['n_total_slices'] + 1e-5)
df['ratio_cor_total'] = df['n_coronal_slices'] / (df['n_total_slices'] + 1e-5)
df['ratio_ax_total'] = df['n_axial_slices'] / (df['n_total_slices'] + 1e-5)
df['tr_te_ratio'] = df['mean_tr'] / (df['mean_te'] + 1e-5)
df = pd.get_dummies(df, columns=['manufacturer'], drop_first=True)

exclude = ['StudyInstanceUID', 'fold', 'is_gold', 'Report', 'has_report'] + target_cols
feature_cols = [c for c in df.columns if c not in exclude]

# Fill missing
df[feature_cols] = df[feature_cols].fillna(0.0)

print(f"Total features: {len(feature_cols)} (18 DICOM + 27 Radiomic features)")

gold_df = df[df['is_gold'] == 1].copy().reset_index(drop=True)
report_df = df[df['is_gold'] == 0].copy().reset_index(drop=True)

oof_rad_preds = np.zeros((len(gold_df), 12))
models_saved = {}

for c, target in enumerate(target_cols):
    oof_target = np.zeros(len(gold_df))
    
    for fold in range(5):
        val_gold = gold_df[gold_df['fold'] == fold]
        val_idx = gold_df[gold_df['fold'] == fold].index
        
        train_gold = gold_df[gold_df['fold'] != fold]
        train_report = report_df[report_df['fold'] != fold]
        train_combined = pd.concat([train_gold, train_report], ignore_index=True)
        
        X_tr = train_combined[feature_cols].values
        y_tr = train_combined[target].values
        w_tr = np.where(train_combined['is_gold'] == 1, 5.0, 1.0)
        
        X_va = val_gold[feature_cols].values
        y_va = val_gold[target].values
        
        # 1. LightGBM
        params = {
            'objective': 'xentropy',
            'metric': 'auc',
            'learning_rate': 0.03,
            'num_leaves': 20,
            'max_depth': 5,
            'feature_fraction': 0.8,
            'min_data_in_leaf': 25,
            'verbose': -1,
            'random_state': 42 + fold
        }
        y_tr_c = np.clip(y_tr, 0.001, 0.999)
        dtrain = lgb.Dataset(X_tr, label=y_tr_c, weight=w_tr)
        gbm = lgb.train(params, dtrain, num_boost_round=100)
        p_lgb = gbm.predict(X_va)
        
        # 2. CatBoost
        cb = CatBoostRegressor(
            iterations=120,
            learning_rate=0.04,
            depth=4,
            verbose=0,
            random_seed=42 + fold
        )
        cb.fit(X_tr, y_tr, sample_weight=w_tr)
        p_cb = np.clip(cb.predict(X_va), 0.01, 0.99)
        
        # 3. Ridge
        rd = Ridge(alpha=15.0)
        rd.fit(X_tr, y_tr, sample_weight=w_tr)
        p_rd = np.clip(rd.predict(X_va), 0.01, 0.99)
        
        # Blend
        p_blend = 0.45 * p_lgb + 0.35 * p_cb + 0.20 * p_rd
        oof_target[val_idx] = p_blend
        
    oof_rad_preds[:, c] = oof_target
    auc = roc_auc_score(gold_df[target].values, oof_target)
    print(f"Target {c:2d} ({target:18s}): 5-Fold OOF AUC = {auc:.4f}")

final_rad_macro = np.mean([roc_auc_score(gold_df[target_cols[c]].values, oof_rad_preds[:, c]) for c in range(12)])
print("\n" + "=" * 85)
print(f">>> DICOM + RADIOMICS GBDT 5-FOLD MACRO-AUC: {final_rad_macro:.4f} <<<")
print("=" * 85)

# Save OOF
oof_df = gold_df[['StudyInstanceUID']].copy()
for c, target in enumerate(target_cols):
    oof_df[target] = oof_rad_preds[:, c]
oof_df.to_csv(r"D:\MLprojs\KaggleCompetition\RSNA\oof\radiomics_tabular_gbdt_oof.csv", index=False)
print("Saved radiomics GBDT OOF to oof/radiomics_tabular_gbdt_oof.csv!")
