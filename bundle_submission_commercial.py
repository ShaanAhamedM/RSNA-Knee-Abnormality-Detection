"""
Bundles Commercial Grandmaster Kaggle Ensemble Package
Packs all backbones, SigLIP weights, GNN checkpoints, and configuration into best_ensemble_weights_fp16.zip.
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
print("BUNDLING COMMERCIAL GRANDMASTER KAGGLE ENSEMBLE PACKAGE")
print("=" * 80)

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    # 1. Configs and Normalization Stats
    configs_to_add = [
        (os.path.join(oof_dir, "commercial_upgraded_config.json"), "commercial_upgraded_config.json"),
        (os.path.join(oof_dir, "commercial_grandmaster_config.json"), "commercial_grandmaster_config.json"),
        (os.path.join(oof_dir, "commercial_triage_evaluation.json"), "commercial_triage_evaluation.json"),
        (os.path.join(oof_dir, "commercial_calibrated_thresholds.json"), "commercial_calibrated_thresholds.json"),
        (os.path.join(oof_dir, "phase7_grandmaster_config.json"), "phase7_grandmaster_config.json"),
        (os.path.join(checkpoints_dir, "phase7_radiomics_norm_stats.json"), "phase7_radiomics_norm_stats.json"),
        (os.path.join(checkpoints_dir, "phase6_norm_stats.json"), "phase6_norm_stats.json")
    ]
    for src, arc in configs_to_add:
        if os.path.exists(src):
            z.write(src, arcname=arc)
            print(f"Added config: {arc}")

    # 2. Add Backbones
    bb_c = os.path.join(checkpoints_dir, "convnext_small_backbone_in1.pt")
    bb_d = os.path.join(checkpoints_dir, "densenet121_backbone_in1.pt")
    siglip = os.path.join(checkpoints_dir, "biomedical_siglip_visual_encoder_best.pt")
    
    if os.path.exists(bb_c):
        print("Packaging ConvNeXt backbone...")
        z.write(bb_c, arcname="convnext_small_backbone_in1.pt")
    if os.path.exists(bb_d):
        print("Packaging DenseNet121 backbone...")
        z.write(bb_d, arcname="densenet121_backbone_in1.pt")
    if os.path.exists(siglip):
        print("Packaging Pretrained Biomedical SigLIP Visual Encoder...")
        z.write(siglip, arcname="biomedical_siglip_visual_encoder_best.pt")

    # 3. Add 5-Fold Models
    for prefix in ["enhanced_gated_gnn", "phase8_gnn", "phase7_transformer", "multislice_transformer", "multiplane_fusion"]:
        for f in range(5):
            cp = os.path.join(checkpoints_dir, f"{prefix}_fold{f}_best.pt")
            if os.path.exists(cp):
                z.write(cp, arcname=f"{prefix}_fold{f}_best.pt")
        print(f"Packaging {prefix} checkpoints...")

    # 4. Add Tabular Radiomics models
    rad_files = glob.glob(os.path.join(checkpoints_dir, "phase7_tabular_radiomics", "*"))
    for rf in rad_files:
        arc = os.path.join("phase7_tabular_radiomics", os.path.basename(rf))
        z.write(rf, arcname=arc)
    print("Packaging Tabular Radiomics GBDT models...")

final_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print("\n" + "=" * 80)
print(f"[SUCCESS] Kaggle Dataset Archive Created: {zip_path}")
print(f"Total Archive Size: {final_size_mb:.1f} MB")
print("=" * 80)
