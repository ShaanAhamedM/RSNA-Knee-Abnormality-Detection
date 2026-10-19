"""
Stage 2: 5-Fold LightGBM & Ridge Tabular DICOM Metadata Model
Trained on 4,407 studies (4,349 report studies + in-fold gold studies).
Evaluated strictly on held-out Gold-58 validation studies.
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.metrics import roc_auc_score
from scipy.special import expit, logit

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print("=" * 80)
print("TRAINING 5-FOLD TABULAR GBDT ON DICOM METADATA (4,407 STUDIES)")
print("=" * 80)

master_df = pd.read_csv('full_dataset_4407_folds.csv')
dicom_df = pd.read_csv('dicom_tabular_features.csv')

# Merge
df = pd.merge(master_df, dicom_df, on='StudyInstanceUID', how='left')

# Feature engineering
df['ratio_sag_total'] = df['n_sagittal_slices'] / (df['n_total_slices'] + 1e-5)
df['ratio_cor_total'] = df['n_coronal_slices'] / (df['n_total_slices'] + 1e-5)
df['ratio_ax_total'] = df['n_axial_slices'] / (df['n_total_slices'] + 1e-5)
df['tr_te_ratio'] = df['mean_tr'] / (df['mean_te'] + 1e-5)

# One-hot encode manufacturer
df = pd.get_dummies(df, columns=['manufacturer'], drop_first=True)

feature_cols = [
    'n_series', 'n_total_slices', 'n_sagittal_slices', 'n_coronal_slices', 'n_axial_slices',
    'ratio_sag_total', 'ratio_cor_total', 'ratio_ax_total',
    'patient_sex', 'field_strength', 'mean_tr', 'mean_te', 'pixel_spacing', 'tr_te_ratio'
] + [c for c in df.columns if c.startswith('manufacturer_')]

print(f"Engineered {len(feature_cols)} tabular features:\n{feature_cols}")

gold_df = df[df['is_gold'] == 1].copy().reset_index(drop=True)
report_df = df[df['is_gold'] == 0].copy().reset_index(drop=True)

oof_tabular_preds = np.zeros((len(gold_df), 12))
test_tabular_preds = None

models_per_target = {}

print("\n--- Running 5-Fold Tabular GBDT Training & Evaluation on Gold 58 ---")
for c, target in enumerate(target_cols):
    oof_target = np.zeros(len(gold_df))
    models_per_target[target] = []
    
    for fold in range(5):
        val_gold = gold_df[gold_df['fold'] == fold]
        val_idx = gold_df[gold_df['fold'] == fold].index
        
        train_gold = gold_df[gold_df['fold'] != fold]
        train_report = report_df[report_df['fold'] != fold]
        
        # Combine train
        train_combined = pd.concat([train_gold, train_report], ignore_index=True)
        
        X_tr = train_combined[feature_cols].values
        y_tr = train_combined[target].values
        # Gold upweighting
        w_tr = np.where(train_combined['is_gold'] == 1, 5.0, 1.0)
        
        X_va = val_gold[feature_cols].values
        y_va = val_gold[target].values
        
        # LightGBM Regressor with cross-entropy objective
        params = {
            'objective': 'xentropy',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 15,
            'max_depth': 4,
            'feature_fraction': 0.8,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'random_state': 42 + fold
        }
        
        y_tr_clipped = np.clip(y_tr, 0.001, 0.999)
        dtrain = lgb.Dataset(X_tr, label=y_tr_clipped, weight=w_tr)
        gbm = lgb.train(params, dtrain, num_boost_round=80)
        
        # Ridge model
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr, y_tr, sample_weight=w_tr)
        
        p_lgb = gbm.predict(X_va)
        p_ridge = np.clip(ridge.predict(X_va), 0.01, 0.99)
        
        p_va = 0.6 * p_lgb + 0.4 * p_ridge
        oof_target[val_idx] = p_va
        models_per_target[target].append((gbm, ridge))
        
    oof_tabular_preds[:, c] = oof_target
    auc = roc_auc_score(gold_df[target].values, oof_target)
    print(f"Target {c:2d} ({target:18s}): 5-Fold OOF AUC = {auc:.4f}")

macro_auc = np.mean([roc_auc_score(gold_df[target_cols[c]].values, oof_tabular_preds[:, c]) for c in range(12)])
print("\n" + "=" * 80)
print(f">>> TABULAR DICOM GBDT 5-FOLD MACRO-AUC: {macro_auc:.4f} <<<")
print("=" * 80)

# Save tabular OOF
oof_df = gold_df[['StudyInstanceUID']].copy()
for c, target in enumerate(target_cols):
    oof_df[target] = oof_tabular_preds[:, c]
oof_df.to_csv(r"D:\MLprojs\KaggleCompetition\RSNA\oof\tabular_dicom_oof.csv", index=False)
print("Saved tabular OOF predictions to oof/tabular_dicom_oof.csv!")
