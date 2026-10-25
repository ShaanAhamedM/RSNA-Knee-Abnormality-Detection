"""
Bundle Phase 7 Grandmaster Kaggle Ensemble Package
Packs all model families, dual backbones, sub-ROI transformers, and tabular checkpoints into best_ensemble_weights_fp16.zip.
"""

import os
import glob
import json
import zipfile

checkpoints_dir = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints"
oof_dir = r"D:\MLprojs\KaggleCompetition\RSNA\oof"
output_dir = r"D:\MLprojs\KaggleCompetition\RSNA\kaggle_submission_package"
zip_path = os.path.join(output_dir, "best_ensemble_weights_fp16.zip")

print("=" * 80)
print("BUNDLING PHASE 7 GRANDMASTER KAGGLE ENSEMBLE PACKAGE")
print("=" * 80)

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    config_path_p7 = os.path.join(oof_dir, "phase7_grandmaster_config.json")
    config_path_p6 = os.path.join(oof_dir, "phase6_grandmaster_config.json")
    norm_stats_p6 = os.path.join(checkpoints_dir, "phase6_norm_stats.json")
    norm_stats_p7 = os.path.join(checkpoints_dir, "phase7_radiomics_norm_stats.json")
    
    if os.path.exists(config_path_p7):
        z.write(config_path_p7, arcname="phase7_grandmaster_config.json")
    if os.path.exists(config_path_p6):
        z.write(config_path_p6, arcname="phase6_grandmaster_config.json")
    if os.path.exists(norm_stats_p6):
        z.write(norm_stats_p6, arcname="phase6_norm_stats.json")
    if os.path.exists(norm_stats_p7):
        z.write(norm_stats_p7, arcname="phase7_radiomics_norm_stats.json")
    print("Added JSON configurations.")

    # 2. Add Both Backbones
    bb_c = os.path.join(checkpoints_dir, "convnext_small_backbone_in1.pt")
    bb_d = os.path.join(checkpoints_dir, "densenet121_backbone_in1.pt")
    if os.path.exists(bb_c):
        print("Packaging ConvNeXt backbone...")
        z.write(bb_c, arcname="convnext_small_backbone_in1.pt")
    if os.path.exists(bb_d):
        print("Packaging DenseNet121 backbone...")
        z.write(bb_d, arcname="densenet121_backbone_in1.pt")

    # 3. Add 5-Fold MultiPlane models
    print("Packaging MultiPlane checkpoints...")
    for f in range(5):
        cp = os.path.join(checkpoints_dir, f"multiplane_fusion_fold{f}_best.pt")
        if os.path.exists(cp):
            z.write(cp, arcname=f"multiplane_fusion_fold{f}_best.pt")

    # 4. Add 5-Fold MS1 Transformer models
    print("Packaging MS1 Transformer checkpoints...")
    for f in range(5):
        cp = os.path.join(checkpoints_dir, f"multislice_transformer_fold{f}_best.pt")
        if os.path.exists(cp):
            z.write(cp, arcname=f"multislice_transformer_fold{f}_best.pt")

    # 5. Add 5-Fold Phase 7 Dual-Backbone Transformer models
    print("Packaging Phase 7 Dual-Backbone Transformer checkpoints...")
    for f in range(5):
        cp = os.path.join(checkpoints_dir, f"phase7_transformer_fold{f}_best.pt")
        if os.path.exists(cp):
            z.write(cp, arcname=f"phase7_transformer_fold{f}_best.pt")

    # 6. Add Phase 7 Tabular Radiomics models
    print("Packaging Phase 7 Tabular Radiomics models...")
    rad_files = glob.glob(os.path.join(checkpoints_dir, "phase7_tabular_radiomics", "*"))
    for rf in rad_files:
        arc = os.path.join("phase7_tabular_radiomics", os.path.basename(rf))
        z.write(rf, arcname=arc)

final_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print("\n" + "=" * 80)
print(f"[SUCCESS] Kaggle Dataset Archive Created: {zip_path}")
print(f"Total Archive Size: {final_size_mb:.1f} MB")
print("=" * 80)
