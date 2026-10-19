"""
Train and Save 5-Fold Tabular DICOM & Radiomics GBDT Checkpoints
Saves native LightGBM, CatBoost, and Ridge models so they can be bundled into
the offline Kaggle submission package and used in final submission pipeline.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print("=" * 80)
print("SAVING 5-FOLD TABULAR & RADIOMICS GBDT CHECKPOINTS")
print("=" * 80)

master_df = pd.read_csv('full_dataset_4407_folds.csv')
rad_tab_df = pd.read_csv('dicom_and_radiomics_features.csv')
df = pd.merge(master_df, rad_tab_df, on='StudyInstanceUID', how='left')

# Feature engineering
df['ratio_sag_total'] = df['n_sagittal_slices'] / (df['n_total_slices'] + 1e-5)
df['ratio_cor_total'] = df['n_coronal_slices'] / (df['n_total_slices'] + 1e-5)
df['ratio_ax_total'] = df['n_axial_slices'] / (df['n_total_slices'] + 1e-5)
df['tr_te_ratio'] = df['mean_tr'] / (df['mean_te'] + 1e-5)
df = pd.get_dummies(df, columns=['manufacturer'], drop_first=True)

# Define DICOM features (18) and All Features (45)
dicom_cols = [
    'n_series', 'n_total_slices', 'n_sagittal_slices', 'n_coronal_slices', 'n_axial_slices',
    'ratio_sag_total', 'ratio_cor_total', 'ratio_ax_total',
    'patient_sex', 'field_strength', 'mean_tr', 'mean_te', 'pixel_spacing', 'tr_te_ratio'
] + [c for c in df.columns if c.startswith('manufacturer_')]

exclude = ['StudyInstanceUID', 'fold', 'is_gold', 'Report', 'has_report'] + target_cols
all_rad_cols = [c for c in df.columns if c not in exclude]

df[all_rad_cols] = df[all_rad_cols].fillna(0.0)

gold_df = df[df['is_gold'] == 1].copy().reset_index(drop=True)
report_df = df[df['is_gold'] == 0].copy().reset_index(drop=True)

os.makedirs('checkpoints/tabular_dicom', exist_ok=True)
os.makedirs('checkpoints/tabular_radiomics', exist_ok=True)

# 1. Train and save Tabular DICOM models (18 features)
print("\n--- Training & Saving Tabular DICOM 5-Fold Models ---")
oof_dicom = np.zeros((len(gold_df), 12))

for c, target in enumerate(target_cols):
    for fold in range(5):
        val_idx = gold_df[gold_df['fold'] == fold].index
        val_gold = gold_df.iloc[val_idx]
        train_gold = gold_df[gold_df['fold'] != fold]
        train_report = report_df[report_df['fold'] != fold]
        train_comb = pd.concat([train_gold, train_report], ignore_index=True)
        
        X_tr = train_comb[dicom_cols].values
        y_tr = train_comb[target].values
        w_tr = np.where(train_comb['is_gold'] == 1, 5.0, 1.0)
        X_va = val_gold[dicom_cols].values
        
        # LightGBM
        params = {
            'objective': 'xentropy',
            'metric': 'auc',
            'learning_rate': 0.03,
            'num_leaves': 15,
            'max_depth': 4,
            'feature_fraction': 0.8,
            'min_data_in_leaf': 30,
            'verbose': -1,
            'random_state': 42 + fold
        }
        dtrain = lgb.Dataset(X_tr, label=np.clip(y_tr, 0.001, 0.999), weight=w_tr)
        gbm = lgb.train(params, dtrain, num_boost_round=80)
        gbm.save_model(f'checkpoints/tabular_dicom/lgb_fold{fold}_target{c}.txt')
        
        # Ridge
        rd = Ridge(alpha=20.0)
        rd.fit(X_tr, y_tr, sample_weight=w_tr)
        joblib.dump(rd, f'checkpoints/tabular_dicom/ridge_fold{fold}_target{c}.joblib')
        
        p = 0.65 * gbm.predict(X_va) + 0.35 * np.clip(rd.predict(X_va), 0.01, 0.99)
        oof_dicom[val_idx, c] = p

# 2. Train and save Tabular Radiomics models (45 features)
print("\n--- Training & Saving Tabular Radiomics 5-Fold Models ---")
oof_rad = np.zeros((len(gold_df), 12))

for c, target in enumerate(target_cols):
    for fold in range(5):
        val_idx = gold_df[gold_df['fold'] == fold].index
        val_gold = gold_df.iloc[val_idx]
        train_gold = gold_df[gold_df['fold'] != fold]
        train_report = report_df[report_df['fold'] != fold]
        train_comb = pd.concat([train_gold, train_report], ignore_index=True)
        
        X_tr = train_comb[all_rad_cols].values
        y_tr = train_comb[target].values
        w_tr = np.where(train_comb['is_gold'] == 1, 5.0, 1.0)
        X_va = val_gold[all_rad_cols].values
        
        # LightGBM
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
        dtrain = lgb.Dataset(X_tr, label=np.clip(y_tr, 0.001, 0.999), weight=w_tr)
        gbm = lgb.train(params, dtrain, num_boost_round=100)
        gbm.save_model(f'checkpoints/tabular_radiomics/lgb_fold{fold}_target{c}.txt')
        
        # CatBoost
        cb = CatBoostRegressor(
            iterations=120,
            learning_rate=0.04,
            depth=4,
            verbose=0,
            random_seed=42 + fold
        )
        cb.fit(X_tr, y_tr, sample_weight=w_tr)
        cb.save_model(f'checkpoints/tabular_radiomics/cat_fold{fold}_target{c}.cbm')
        
        # Ridge
        rd = Ridge(alpha=15.0)
        rd.fit(X_tr, y_tr, sample_weight=w_tr)
        joblib.dump(rd, f'checkpoints/tabular_radiomics/ridge_fold{fold}_target{c}.joblib')
        
        p = 0.45 * gbm.predict(X_va) + 0.35 * np.clip(cb.predict(X_va), 0.01, 0.99) + 0.20 * np.clip(rd.predict(X_va), 0.01, 0.99)
        oof_rad[val_idx, c] = p

meta = {
    'dicom_cols': dicom_cols,
    'all_rad_cols': all_rad_cols,
    'target_cols': target_cols
}
with open('checkpoints/tabular_models_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\nAll 5-fold GBDT and Ridge models saved successfully to checkpoints/!")
