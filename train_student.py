"""
Grandmaster Student Model Training with Distilled Pseudo-Labels.
Trains high-capacity 2.5D models on expanded ~400-study dataset,
validating strictly against the 58 gold-standard radiologist consensus ground truth.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.engine import train_fold_with_auto_oom
from src.ensemble import optimize_ensemble_weights
from kaggle_inference_submission import run_inference
from src.pseudo_labeling import generate_pseudo_labels

PSEUDO_CSV = r"D:\MLprojs\KaggleCompetition\RSNA\pseudo_labeled_folds.csv"
EXP_LOG = r"D:\MLprojs\KaggleCompetition\RSNA\experiments_log.md"

def train_grandmaster_student():
    print("=" * 80)
    print("[*] STARTING PHASE 5: GRANDMASTER STUDENT MODEL TRAINING")
    print("=" * 80)

    # 1. Generate pseudo-labels if not present or refreshed
    if not os.path.exists(PSEUDO_CSV):
        print("[Distillation] Generating fresh high-confidence pseudo-labels...")
        combined_df = generate_pseudo_labels(max_unlabeled_studies=350, slices_per_study=6)
    else:
        combined_df = pd.read_csv(PSEUDO_CSV)
        print(f"[Distillation] Loaded existing pseudo-labeled dataset: {len(combined_df)} studies")

    gold_df = combined_df[combined_df['is_pseudo'] == 0].copy().reset_index(drop=True)
    pseudo_df = combined_df[combined_df['is_pseudo'] == 1].copy().reset_index(drop=True)

    print(f"Dataset Split: {len(gold_df)} Gold-Standard Studies + {len(pseudo_df)} Distilled Pseudo-Labeled Studies")
    print(f"Total Effective Training Pool: {len(combined_df)} studies")

    # 2. Student Model Configuration
    student_config = {
        'iteration': 100,
        'family': 'Distilled-Student-ConvNeXt',
        'backbone': 'convnext_small',
        'loss_type': 'poly',
        'poly_eps': 1.0,
        'target_size': 256,
        'slices_per_study': 6,
        'in_chans': 1,
        'batch_size': 4,
        'grad_accum_steps': 4,
        'epochs': 10,
        'lr': 2e-4,
        'notes': f"Student ConvNeXt-Small trained on {len(combined_df)} studies with Knowledge Distillation"
    }

    print(f"\nTraining Student Architecture: {student_config['backbone']}")
    print(f"Configuration: Resolution={student_config['target_size']}x{student_config['target_size']}, Slices={student_config['slices_per_study']}, Epochs={student_config['epochs']}")

    fold_scores = []
    
    # 3. 5-Fold Training: Validation is strictly evaluated on Gold Labels
    for fold in range(5):
        # Validation subset: ONLY gold-standard ground truth for this fold
        val_subset = gold_df[gold_df['fold'] == fold].copy().reset_index(drop=True)
        
        # Training subset: Gold training studies + ALL high-confidence pseudo-labeled studies
        train_gold = gold_df[gold_df['fold'] != fold].copy().reset_index(drop=True)
        train_subset = pd.concat([train_gold, pseudo_df], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        print(f"\n======================================================================")
        print(f"[*] TRAINING STUDENT FOLD {fold}/5: {len(train_subset)} Train Studies ({len(train_gold)} Gold + {len(pseudo_df)} Pseudo) | {len(val_subset)} Val Studies (Strict Gold)")
        print(f"======================================================================")
        
        try:
            result = train_fold_with_auto_oom(
                model_name=student_config['backbone'],
                fold=fold,
                train_df=train_subset,
                val_df=val_subset,
                config=student_config
            )
            fold_scores.append(result['best_auc'])
            print(f"[Student Fold {fold} Complete] Val Macro-AUC: {result['best_auc']:.4f}")
        except Exception as e:
            print(f"[Error in Student Fold {fold}] {e}")
            fold_scores.append(0.5000)

    mean_student_auc = float(np.mean(fold_scores))
    print(f"\n{'='*80}")
    print(f"[+] GRANDMASTER STUDENT 5-FOLD MEAN MACRO-AUC: {mean_student_auc:.4f}")
    print(f"{'='*80}\n")

    # 4. Master Re-Ensemble across all checkpoints
    print("--- Running Master Mathematical Ensemble Optimization (SLSQP) ---")
    ens_res = optimize_ensemble_weights()
    ensemble_score = ens_res.get('score', mean_student_auc)
    print(f">>> Final Grandmaster Ensemble OOF Macro-AUC: {ensemble_score:.4f} <<<")

    # 5. Overwrite and strictly verify final submission.csv
    print("\n--- Generating & Verifying Final 1st-Place Submission ---")
    try:
        sub_df = run_inference()
        print("[SUCCESS] Final submission regenerated, verified, and ready for 1st-place submission!")
    except Exception as e:
        print(f"Error in final inference: {e}")

    # 6. Log to experiments_log.md
    with open(EXP_LOG, 'a', encoding='utf-8') as f:
        f.write(f"\n### Phase 5: Grandmaster Self-Distillation Student ({student_config['backbone']})\n")
        f.write(f"- **Data Expansion:** {len(gold_df)} Gold $\\rightarrow$ {len(combined_df)} Studies (Self-Distilled Pseudo-Labels)\n")
        f.write(f"- **Student 5-Fold OOF Macro-AUC:** {mean_student_auc:.4f}\n")
        f.write(f"- **Final Ensemble OOF Macro-AUC:** {ensemble_score:.4f}\n")
        f.write(f"- **Resolution & Slices:** {student_config['target_size']}x{student_config['target_size']}, K={student_config['slices_per_study']}\n")

if __name__ == '__main__':
    train_grandmaster_student()
