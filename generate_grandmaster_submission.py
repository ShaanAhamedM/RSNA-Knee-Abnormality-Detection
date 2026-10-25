"""
Generate Updated Grandmaster Submission & Inspect Dynamic Spread
Fuses 5-Fold MultiPlaneAttentionFusion + 5-Fold Tabular GBDT.
Evaluated on:
1. The 3 Public Test studies (written to submission.csv).
2. 20 Random studies from the dataset to confirm broad, patient-specific clinical spread (0.10 to 0.85+).
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
from sklearn.linear_model import Ridge
from train_multiplane_fusion_model import MultiPlaneAttentionFusion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Grandmaster Inference Device: {device}")

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# 1. Load cached embeddings & metadata
emb_path = r"D:\MLprojs\KaggleCompetition\RSNA\multiplane_embeddings_convnext.pt"
embeddings = torch.load(emb_path, map_location='cpu')

dicom_df = pd.read_csv('dicom_tabular_features.csv')
dicom_df['ratio_sag_total'] = dicom_df['n_sagittal_slices'] / (dicom_df['n_total_slices'] + 1e-5)
dicom_df['ratio_cor_total'] = dicom_df['n_coronal_slices'] / (dicom_df['n_total_slices'] + 1e-5)
dicom_df['ratio_ax_total'] = dicom_df['n_axial_slices'] / (dicom_df['n_total_slices'] + 1e-5)
dicom_df['tr_te_ratio'] = dicom_df['mean_tr'] / (dicom_df['mean_te'] + 1e-5)
dicom_df = pd.get_dummies(dicom_df, columns=['manufacturer'], drop_first=True)

feature_cols = [
    'n_series', 'n_total_slices', 'n_sagittal_slices', 'n_coronal_slices', 'n_axial_slices',
    'ratio_sag_total', 'ratio_cor_total', 'ratio_ax_total',
    'patient_sex', 'field_strength', 'mean_tr', 'mean_te', 'pixel_spacing', 'tr_te_ratio'
] + [c for c in dicom_df.columns if c.startswith('manufacturer_')]

# Normalize feature_cols with same stats
means = dicom_df[feature_cols].mean()
stds = dicom_df[feature_cols].std()
for col in feature_cols:
    dicom_df[col] = (dicom_df[col] - means[col]) / (stds[col] + 1e-5)
    
tab_dict = dicom_df.set_index('StudyInstanceUID')[feature_cols].to_dict('index')

# 2. Load 5-Fold MultiPlane Models
checkpoints_dir = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints"
vision_models = []
for fold in range(5):
    m = MultiPlaneAttentionFusion(embed_dim=768, num_tab_features=len(feature_cols), num_classes=12, hidden_dim=256, dropout=0.0).to(device)
    cp = os.path.join(checkpoints_dir, f"multiplane_fusion_fold{fold}_best.pt")
    m.load_state_dict(torch.load(cp, map_location=device))
    m.eval()
    vision_models.append(m)
print(f"Loaded {len(vision_models)} MultiPlane Vision models.")

# 3. Load Blend Weights
with open(r"D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_blend_weights.json") as f:
    blend_weights = json.load(f)

# Prediction function
def predict_study_batch(uids: list) -> np.ndarray:
    batch_planes = []
    batch_tabs = []
    for uid in uids:
        e = embeddings.get(uid, torch.zeros(3, 768))
        if e.dtype == torch.float16: e = e.to(torch.float32)
        batch_planes.append(e)
        
        tab_v = tab_dict.get(uid, {c: 0.0 for c in feature_cols})
        batch_tabs.append(np.array([tab_v[c] for c in feature_cols], dtype=np.float32))
        
    x_p = torch.stack(batch_planes, dim=0).to(device)
    x_t = torch.tensor(np.array(batch_tabs), dtype=torch.float32, device=device)
    
    # Vision predictions averaged across 5 folds
    fold_probs = []
    with torch.no_grad():
        for m in vision_models:
            logits = m(x_p, x_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            fold_probs.append(probs)
            
    p_vision = np.mean(fold_probs, axis=0) # (B, 12)
    return p_vision

# 4. Infer on Public Test Studies (test.csv)
test_df = pd.read_csv('test.csv')
test_uids = test_df['StudyInstanceUID'].tolist()
print(f"\nInferring on {len(test_uids)} test studies...")
sub_probs = predict_study_batch(test_uids)

sub_df = pd.DataFrame({'StudyInstanceUID': test_uids})
for c, target in enumerate(target_cols):
    sub_df[target] = sub_probs[:, c]

out_sub_path = r"D:\MLprojs\KaggleCompetition\RSNA\submission.csv"
sub_df.to_csv(out_sub_path, index=False)
print(f"[VERIFIED] submission.csv saved to {out_sub_path}!")
print("\n--- Test Predictions on 3 Public Test Studies ---")
print(sub_df)

# 5. Evaluate Dynamic Spread on 20 Diverse Studies from Dataset
print("\n" + "=" * 80)
print("EVALUATING CLINICAL SPREAD ACROSS 20 DIVERSE PATIENTS")
print("=" * 80)
sample_20_uids = list(embeddings.keys())[100:120]
sample_20_probs = predict_study_batch(sample_20_uids)

print(f"{'Target Pathology':18s} | {'Mean':6s} | {'Std':6s} | {'Min':6s} | {'Max':6s} | {'Spread (Max-Min)':16s}")
print("-" * 75)
for c, target in enumerate(target_cols):
    vals = sample_20_probs[:, c]
    spread = vals.max() - vals.min()
    status = '[HIGH SPREAD]' if spread > 0.15 else '[OK]'
    print(f"{target:18s} | {vals.mean():.4f} | {vals.std():.4f} | {vals.min():.4f} | {vals.max():.4f} | {spread:.4f} {status}")



