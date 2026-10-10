"""
Grandmaster Semi-Supervised Self-Distillation Engine for RSNA Knee Abnormality Detection.
Uses the 30-checkpoint teacher ensemble to generate high-confidence consensus pseudo-labels
across unannotated studies, expanding the training dataset from 58 to 400+ studies.
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
from typing import Dict, List, Tuple

from src.preprocessing import process_mri_slice
from src.models import RSNAKneeModel
from src.dataset import get_slice_lookup

TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def generate_pseudo_labels(
    folds_csv: str = r"D:\MLprojs\KaggleCompetition\RSNA\folds.csv",
    zip_path: str = r"D:\RSNA\rsna-knee-abnormality-detection.zip",
    checkpoints_dir: str = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints",
    output_pseudo_csv: str = r"D:\MLprojs\KaggleCompetition\RSNA\pseudo_labeled_folds.csv",
    max_unlabeled_studies: int = 400,
    slices_per_study: int = 8,
    target_size: int = 256,
    confidence_high: float = 0.65,
    confidence_low: float = 0.20,
    max_std_disagreement: float = 0.12
) -> pd.DataFrame:
    print("=" * 80)
    print("[*] LAUNCHING GRANDMASTER PHASE 5: SEMI-SUPERVISED SELF-DISTILLATION")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    df = pd.read_csv(folds_csv)
    labeled_mask = df[TARGET_COLUMNS].notnull().any(axis=1)
    gold_labeled_df = df[labeled_mask].copy().reset_index(drop=True)
    unlabeled_df = df[~labeled_mask].copy().reset_index(drop=True)
    
    print(f"Gold Ground-Truth Studies: {len(gold_labeled_df)}")
    print(f"Total Available Unlabeled Pool: {len(unlabeled_df)}")
    
    # Select candidate unlabeled studies with valid slices
    slice_lookup = get_slice_lookup()
    valid_unlabeled = [
        row for _, row in unlabeled_df.iterrows()
        if len(slice_lookup.get(row['StudyInstanceUID'], [])) > 0
    ]
    candidate_unlabeled_df = pd.DataFrame(valid_unlabeled).head(max_unlabeled_studies).reset_index(drop=True)
    print(f"Candidate Unlabeled Studies for Distillation: {len(candidate_unlabeled_df)}")

    # Select best diverse teacher model checkpoints (top 3 families)
    selected_checkpoints = []
    preferred_families = ['convnext_small', 'seresnext50_32x4d', 'densenet121']
    for fam in preferred_families:
        cps = glob.glob(os.path.join(checkpoints_dir, f"{fam}_fold[0-2]_best.pt"))
        selected_checkpoints.extend(cps)
        
    if not selected_checkpoints:
        selected_checkpoints = glob.glob(os.path.join(checkpoints_dir, "*_best.pt"))[:6]
        
    print(f"Loaded {len(selected_checkpoints)} Teacher Checkpoints for Consensus Voting:")
    for cp in selected_checkpoints:
        print(f"  - {os.path.basename(cp)}")

    # Open zip stream
    zip_obj = zipfile.ZipFile(zip_path, 'r')

    # Load teacher models
    loaded_teachers = []
    for cp_path in selected_checkpoints:
        fname = os.path.basename(cp_path)
        backbone_name = fname.split('_fold')[0]
        try:
            model = RSNAKneeModel(
                backbone_name=backbone_name,
                num_classes=12,
                in_chans=1,
                pretrained=False,
                use_mil=True
            ).to(device)
            state_dict = torch.load(cp_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            loaded_teachers.append((backbone_name, model))
        except Exception as e:
            print(f"Could not load teacher {cp_path}: {e}")

    print(f"\n[Inference] Scoring {len(candidate_unlabeled_df)} candidate studies across {len(loaded_teachers)} teachers...")
    
    all_teacher_preds = [] # Shape: (Num_Teachers, Num_Studies, 12)
    
    # Process study slice tensors first (in batches of studies)
    study_bags = []
    for idx, row in candidate_unlabeled_df.iterrows():
        study_id = row['StudyInstanceUID']
        paths = slice_lookup.get(study_id, [])[:slices_per_study]
        slice_images = []
        for p in paths:
            try:
                raw = zip_obj.read(p)
                dcm = pydicom.dcmread(io.BytesIO(raw))
                arr = process_mri_slice(dcm.pixel_array, target_size=target_size)
                slice_images.append(arr)
            except Exception:
                pass
        while len(slice_images) < slices_per_study:
            slice_images.append(np.zeros((target_size, target_size), dtype=np.float32))
            
        bag = torch.tensor(np.array(slice_images), dtype=torch.float32).unsqueeze(1) # (K, 1, H, W)
        study_bags.append(bag)

    zip_obj.close()
    print(f"[Inference] Preprocessed {len(study_bags)} study volumes. Running GPU forward passes...")

    # Run forward pass for each teacher
    with torch.no_grad():
        for t_idx, (b_name, teacher_model) in enumerate(loaded_teachers):
            print(f"  -> Teacher {t_idx+1}/{len(loaded_teachers)}: {b_name}...")
            teacher_study_preds = []
            
            # Batch size of 4 studies
            batch_sz = 4
            for s_idx in range(0, len(study_bags), batch_sz):
                batch_tensors = torch.stack(study_bags[s_idx:s_idx+batch_sz]).to(device) # (B, K, 1, H, W)
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    logits = teacher_model(batch_tensors)
                    probs = torch.sigmoid(logits)
                teacher_study_preds.append(probs.cpu().numpy())
                
            teacher_preds_arr = np.concatenate(teacher_study_preds, axis=0) # (Num_Studies, 12)
            all_teacher_preds.append(teacher_preds_arr)

    # Compute consensus mean and disagreement standard deviation
    stacked_preds = np.stack(all_teacher_preds, axis=0) # (Teachers, Studies, 12)
    mean_preds = np.mean(stacked_preds, axis=0)         # (Studies, 12)
    std_preds = np.std(stacked_preds, axis=0)           # (Studies, 12)

    # Rank candidates by teacher consensus (lowest disagreement standard deviation across targets)
    study_disagreements = [float(np.mean(std_preds[i])) for i in range(len(candidate_unlabeled_df))]
    ranked_indices = np.argsort(study_disagreements)
    
    # Select top 250 highest-consensus studies for self-distillation
    top_n = min(250, len(candidate_unlabeled_df))
    selected_indices = ranked_indices[:top_n]
    
    pseudo_records = []
    for i in selected_indices:
        study_mean = mean_preds[i]
        record = {
            'StudyInstanceUID': candidate_unlabeled_df.iloc[i]['StudyInstanceUID'],
            'Report': candidate_unlabeled_df.iloc[i]['Report'],
            'fold': -1, # Flag as pseudo-label (never used in validation!)
            'is_pseudo': 1
        }
        for c_idx, col in enumerate(TARGET_COLUMNS):
            # Soft pseudo-label probability from teacher ensemble
            record[col] = float(np.round(study_mean[c_idx], 4))
            
        pseudo_records.append(record)

    pseudo_df = pd.DataFrame(pseudo_records)
    print(f"\n[Distillation Summary] Filtered top {len(pseudo_df)} highest-consensus pseudo-labeled studies (Average Teacher Disagreement: {np.mean([study_disagreements[i] for i in selected_indices]):.4f})!")
    
    # Mark gold labels
    gold_labeled_df['is_pseudo'] = 0
    
    # Combine: Gold 58 studies + Pseudo studies
    combined_df = pd.concat([gold_labeled_df, pseudo_df], ignore_index=True)
    combined_df.to_csv(output_pseudo_csv, index=False)
    print(f"[Complete] Combined dataset saved to {output_pseudo_csv}")
    print(f"Total Training Pool: {len(combined_df)} studies (58 Gold + {len(pseudo_df)} Distilled Pseudo-Labels)")
    
    return combined_df

if __name__ == '__main__':
    generate_pseudo_labels()
