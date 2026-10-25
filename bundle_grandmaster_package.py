"""
Bundle Grandmaster Weights & Metadata into best_ensemble_weights_fp16.zip
For Kaggle Dataset upload as 'rsna-knee-weights'
"""

import os
import json
import zipfile
import pandas as pd
import torch

print("=" * 80)
print("BUNDLING GRANDMASTER WEIGHTS & PARAMETERS FOR KAGGLE")
print("=" * 80)

# Save normalization statistics for DICOM tabular features
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

norm_stats = {
    'feature_cols': feature_cols,
    'means': {c: float(dicom_df[c].mean()) for c in feature_cols},
    'stds': {c: float(dicom_df[c].std()) for c in feature_cols}
}

norm_path = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints\grandmaster_metadata_norm_stats.json"
with open(norm_path, 'w') as f:
    json.dump(norm_stats, f, indent=2)
print(f"Saved tabular normalization stats to {norm_path}!")

out_zip = r"D:\MLprojs\KaggleCompetition\RSNA\kaggle_submission_package\best_ensemble_weights_fp16.zip"
with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    # 1. Multiplane fusion models
    for fold in range(5):
        fn = f"multiplane_fusion_fold{fold}_best.pt"
        fp = os.path.join(r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints", fn)
        z.write(fp, arcname=fn)
        print(f"Added {fn}")
        
    # 2. Offline backbone
    bb_path = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints\convnext_small_backbone_in1.pt"
    z.write(bb_path, arcname="convnext_small_backbone_in1.pt")
    print("Added convnext_small_backbone_in1.pt")
    
    # 3. Parameters and stats
    z.write(norm_path, arcname="grandmaster_metadata_norm_stats.json")
    print("Added grandmaster_metadata_norm_stats.json")
    
    weights_path = r"D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_blend_weights.json"
    z.write(weights_path, arcname="grandmaster_blend_weights.json")
    print("Added grandmaster_blend_weights.json")

print(f"\n[SUCCESS] Bundled archive saved to {out_zip}!")
print(f"Archive size: {os.path.getsize(out_zip) / (1024*1024):.1f} MB")
