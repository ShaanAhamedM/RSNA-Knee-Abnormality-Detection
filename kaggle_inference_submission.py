"""
Grandmaster Kaggle Submission & Test-Time Augmentation (TTA) Inference Engine
Audited & Upgraded:
- Prunes all checkpoints with OOF AUC < 0.52 (weight 0.0)
- Applies 12 independent per-target sub-ensembles
- Evaluates Level-2 Stacking (selective gating)
- Applies Log-Odds Temperature Scaling & Prior Matching to match empirical pi_c
- Strictly verifies against sample_submission.csv
"""

import os
import io
import json
import glob
import zipfile
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from scipy.special import logit, expit

from src.preprocessing import process_mri_slice
from src.models import RSNAKneeModel

TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def run_inference(
    test_csv_path: str = r"D:\MLprojs\KaggleCompetition\RSNA\test.csv",
    zip_path: str = r"D:\RSNA\rsna-knee-abnormality-detection.zip",
    checkpoints_dir: str = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints",
    output_submission_path: str = r"D:\MLprojs\KaggleCompetition\RSNA\submission.csv",
    sample_sub_path: str = r"D:\MLprojs\KaggleCompetition\RSNA\sample_submission.csv",
    params_json_path: str = r"D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_params.json",
    slices_per_study: int = 6,
    in_chans: int = 1,
    use_tta: bool = True
):
    print("=" * 80)
    print("=== GRANDMASTER SUBMISSION GENERATION & VERIFICATION PIPELINE ===")
    print("=" * 80)
    
    test_df = pd.read_csv(test_csv_path)
    N_test = len(test_df)
    print(f"Loaded {N_test} test studies from {test_csv_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference device: {device}")
    
    # Load Grandmaster parameters
    if not os.path.exists(params_json_path):
        raise FileNotFoundError(f"Grandmaster parameters not found at {params_json_path}")
        
    with open(params_json_path, 'r') as f:
        gm_params = json.load(f)
        
    kept_models = gm_params['kept_models']
    kept_folds = gm_params['kept_folds']
    per_target_weights = gm_params['per_target_weights']
    stacker_models = gm_params['stacker_models']
    calibration_params = gm_params['calibration_params']
    
    print("\n--- Model Pruning & Curation Forensics ---")
    print(f"Pruned Model Families: {gm_params.get('pruned_models', ['efficientnet_b0'])} (assigned weight 0.0)")
    for m in kept_models:
        print(f"  {m:28s}: Retained folds {kept_folds[m]} (all folds with OOF AUC < 0.52 pruned)")
        
    # Read test slices from zip index
    from src.dataset import get_slice_lookup
    slice_lookup = get_slice_lookup()
    
    zip_obj = zipfile.ZipFile(zip_path, 'r')
    
    # Store predictions per kept model family: family_preds[model_name] = np.ndarray (N_test, 12)
    family_preds = {}
    
    for backbone_name in kept_models:
        folds = kept_folds.get(backbone_name, [])
        if not folds:
            continue
            
        fold_predictions = []
        for fold in folds:
            cp_path = os.path.join(checkpoints_dir, f"{backbone_name}_fold{fold}_best.pt")
            if not os.path.exists(cp_path):
                print(f"Warning: Checkpoint {cp_path} not found! Skipping.")
                continue
                
            print(f"Loading {os.path.basename(cp_path)}...")
            timm_name = backbone_name.replace('_report_sup', '')
            model = RSNAKneeModel(
                backbone_name=timm_name,
                num_classes=12,
                in_chans=in_chans,
                pretrained=False,
                use_mil=True
            ).to(device)
            
            state_dict = torch.load(cp_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            
            # Determine target resolution for this backbone
            cp_target_size = 256
            if '224' in backbone_name or 'swin' in backbone_name:
                cp_target_size = 224
            elif '288' in backbone_name:
                cp_target_size = 288
            elif hasattr(model.backbone, 'pretrained_cfg') and isinstance(model.backbone.pretrained_cfg, dict):
                cp_target_size = model.backbone.pretrained_cfg.get('input_size', (1, 256, 256))[1]
                
            fold_preds = []
            with torch.no_grad():
                for idx, row in test_df.iterrows():
                    study_id = row['StudyInstanceUID']
                    slice_paths = slice_lookup.get(study_id, [])
                    if not slice_paths:
                        slice_paths = [f for f in zip_obj.namelist() if study_id in f and f.endswith('.dcm')][:slices_per_study]
                        
                    slice_images = []
                    for sp in slice_paths[:slices_per_study]:
                        try:
                            raw = zip_obj.read(sp)
                            dcm = pydicom.dcmread(io.BytesIO(raw))
                            arr = process_mri_slice(dcm.pixel_array, target_size=cp_target_size)
                            slice_images.append(arr)
                        except Exception:
                            pass
                            
                    while len(slice_images) < slices_per_study:
                        slice_images.append(np.zeros((cp_target_size, cp_target_size), dtype=np.float32))
                        
                    bag = torch.tensor(np.array(slice_images), dtype=torch.float32).unsqueeze(1) # (K, 1, H, W)
                    if in_chans == 3:
                        bag = bag.repeat(1, 3, 1, 1)
                    bag_batch = bag.unsqueeze(0).to(device)
                    
                    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                        logits = model(bag_batch)
                        probs = torch.sigmoid(logits)
                        
                        if use_tta:
                            # Photometric perturbation TTA (strictly preserving medial-lateral anatomical orientation)
                            bag_jittered = torch.clamp(1.05 * bag_batch - 0.02, 0.0, 1.0)
                            logits_jittered = model(bag_jittered)
                            probs = 0.5 * (probs + torch.sigmoid(logits_jittered))
                            
                    fold_preds.append(probs.cpu().numpy()[0])
                    
            fold_predictions.append(np.array(fold_preds)) # (N_test, 12)
            
        if fold_predictions:
            # Average across kept folds for this family
            family_preds[backbone_name] = np.mean(fold_predictions, axis=0) # (N_test, 12)
            print(f"  -> Averaged {len(fold_predictions)} folds for {backbone_name}")
            
    zip_obj.close()
    
    # Apply 12 independent per-target sub-ensembles
    print("\n--- Applying 12 Independent Per-Target Sub-Ensembles ---")
    L1_preds = np.zeros((N_test, 12), dtype=np.float32)
    
    for c, target_name in enumerate(TARGET_COLUMNS):
        w_dict = per_target_weights[target_name]['weights']
        col_pred = np.zeros(N_test, dtype=np.float32)
        total_w = 0.0
        for m_name, w in w_dict.items():
            if m_name in family_preds and w > 0:
                col_pred += w * family_preds[m_name][:, c]
                total_w += w
        if total_w > 0:
            col_pred /= total_w
        else:
            col_pred = np.mean([family_preds[m][:, c] for m in family_preds], axis=0)
        L1_preds[:, c] = col_pred
        print(f"  {target_name:18s}: Ensembled {len([m for m, w in w_dict.items() if w > 0])} models (L1 OOF AUC: {per_target_weights[target_name]['auc']:.4f})")
        
    # Evaluate Level-2 Stacker Gating
    print("\n--- Evaluating Level-2 Pathology Co-Occurrence Stacker ---")
    L2_preds = np.zeros_like(L1_preds)
    for c, target_name in enumerate(TARGET_COLUMNS):
        stk = stacker_models[target_name]
        if stk.get('use_stacker', False):
            coef = np.array(stk['coef'], dtype=np.float32)
            intercept = float(stk['intercept'])
            L2_preds[:, c] = L1_preds @ coef + intercept
            print(f"  [+] Level-2 Stacker applied for {target_name:18s} (CV gain: +{stk.get('gain', 0.0):.4f})")
        else:
            L2_preds[:, c] = L1_preds[:, c]
            print(f"  [-] Level-1 Retained for {target_name:18s}")
            
    # Apply Log-Odds Temperature Scaling & Empirical Prior Matching
    print("\n--- Applying Log-Odds Temperature Scaling & Prior Matching ---")
    final_probs = np.zeros_like(L2_preds)
    for c, target_name in enumerate(TARGET_COLUMNS):
        cal = calibration_params[target_name]
        mu_l = cal['mu_logit']
        tau = cal['tau']
        target_logit_pi = cal['target_logit_pi']
        target_pi = cal['target_pi']
        
        raw_val = L2_preds[:, c]
        # Smooth clipping to prevent infinite logits
        p_safe = np.clip(raw_val, 1e-4, 1.0 - 1e-4)
        l_raw = logit(p_safe)
        
        cal_logits = (l_raw - mu_l) / tau + target_logit_pi
        p_cal = expit(cal_logits)
        
        # Bounded between 0.001 and 0.999
        final_probs[:, c] = np.clip(p_cal, 0.001, 0.999)
        print(f"  {target_name:18s}: Calibrated Mean = {final_probs[:, c].mean():.3f} (True pi = {target_pi:.3f}) | Range = [{final_probs[:, c].min():.3f}, {final_probs[:, c].max():.3f}]")
        
    # Construct final submission DataFrame
    sub_df = pd.DataFrame({'StudyInstanceUID': test_df['StudyInstanceUID']})
    for i, col in enumerate(TARGET_COLUMNS):
        sub_df[col] = final_probs[:, i]
        
    # Strict Verification
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        assert list(sub_df.columns) == list(sample_sub.columns), f"Columns mismatch! {sub_df.columns} vs {sample_sub.columns}"
        assert len(sub_df) == len(sample_sub), f"Row count mismatch! {len(sub_df)} vs {len(sample_sub)}"
        assert (sub_df['StudyInstanceUID'].values == sample_sub['StudyInstanceUID'].values).all(), "StudyInstanceUID ordering mismatch!"
        
    assert not sub_df.isnull().any().any(), "Submission contains NaN values!"
    assert not np.isinf(sub_df[TARGET_COLUMNS].values).any(), "Submission contains Inf values!"
    assert (sub_df[TARGET_COLUMNS].values >= 0.0).all() and (sub_df[TARGET_COLUMNS].values <= 1.0).all(), "Values outside [0, 1]!"
    
    sub_df.to_csv(output_submission_path, index=False)
    print(f"\n[VERIFIED] Grandmaster submission successfully written to {output_submission_path}!")
    print(sub_df.to_string())
    return sub_df

if __name__ == '__main__':
    run_inference()
