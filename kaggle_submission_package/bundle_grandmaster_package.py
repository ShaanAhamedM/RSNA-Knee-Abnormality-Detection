"""
Bundle Phase 6 Grandmaster Kaggle Ensemble Package
Packs all 4 model families into best_ensemble_weights_fp16.zip:
1. ConvNeXt-Small backbone
2. 5-Fold MultiPlane Attention Fusion models
3. 5-Fold MultiSlice Target-Query Transformer models
4. 5-Fold Tabular DICOM models (LightGBM + Ridge)
5. 5-Fold Tabular Radiomics models (LightGBM + CatBoost + Ridge)
6. Normalization statistics & optimal per-target ensemble configuration
"""

import os
import glob
import json
import zipfile
import torch

checkpoints_dir = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints"
oof_dir = r"D:\MLprojs\KaggleCompetition\RSNA\oof"
output_dir = r"D:\MLprojs\KaggleCompetition\RSNA\kaggle_submission_package"
zip_path = os.path.join(output_dir, "best_ensemble_weights_fp16.zip")

print("=" * 80)
print("BUNDLING PHASE 6 GRANDMASTER KAGGLE ENSEMBLE PACKAGE")
print("=" * 80)

# Build zip archive
temp_dir = os.path.join(output_dir, "temp_fp16")
os.makedirs(temp_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    # 1. Add Configuration & Norm stats
    config_path = os.path.join(oof_dir, "phase6_grandmaster_config.json")
    norm_stats_path = os.path.join(checkpoints_dir, "phase6_norm_stats.json")
    
    assert os.path.exists(config_path), f"Config {config_path} missing!"
    assert os.path.exists(norm_stats_path), f"Norm stats {norm_stats_path} missing!"
    
    z.write(config_path, arcname="phase6_grandmaster_config.json")
    z.write(norm_stats_path, arcname="phase6_norm_stats.json")
    print("Added JSON configurations.")

    # 2. Add Backbone
    bb_path = os.path.join(checkpoints_dir, "convnext_small_backbone_in1.pt")
    if os.path.exists(bb_path):
        print("Packaging backbone...")
        z.write(bb_path, arcname="convnext_small_backbone_in1.pt")

    # 3. Add 5-Fold MultiPlane models
    print("Packaging MultiPlane checkpoints...")
    for f in range(5):
        cp = os.path.join(checkpoints_dir, f"multiplane_fusion_fold{f}_best.pt")
        z.write(cp, arcname=f"multiplane_fusion_fold{f}_best.pt")

    # 4. Add 5-Fold MultiSlice Transformer models
    print("Packaging MultiSlice Transformer checkpoints...")
    for f in range(5):
        cp = os.path.join(checkpoints_dir, f"multislice_transformer_fold{f}_best.pt")
        z.write(cp, arcname=f"multislice_transformer_fold{f}_best.pt")

    # 5. Add Tabular DICOM models
    print("Packaging Tabular DICOM models...")
    dicom_files = glob.glob(os.path.join(checkpoints_dir, "tabular_dicom", "*"))
    for df in dicom_files:
        arc = os.path.join("tabular_dicom", os.path.basename(df))
        z.write(df, arcname=arc)

    # 6. Add Tabular Radiomics models
    print("Packaging Tabular Radiomics models...")
    rad_files = glob.glob(os.path.join(checkpoints_dir, "tabular_radiomics", "*"))
    for rf in rad_files:
        arc = os.path.join("tabular_radiomics", os.path.basename(rf))
        z.write(rf, arcname=arc)

final_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print("\n" + "=" * 80)
print(f"[SUCCESS] Kaggle Dataset Archive Created: {zip_path}")
print(f"Total Archive Size: {final_size_mb:.1f} MB")
print("=" * 80)
