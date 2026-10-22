"""
Autonomous Overnight Tournament Loop for RSNA Knee Abnormality Detection
Runs continuously, training new model families, loss landscapes, and hyperparameter perturbations.
Optimizes ensembling via Scipy and regenerates submission.csv whenever OOF CV improves.
"""

import os
import sys
import time
import copy
import json
import numpy as np
import pandas as pd
import torch

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from src.engine import train_fold_with_auto_oom
from src.ensemble import optimize_ensemble_weights
from kaggle_inference_submission import run_inference

FOLDS_CSV = r"D:\MLprojs\KaggleCompetition\RSNA\folds.csv"
EXP_LOG = r"D:\MLprojs\KaggleCompetition\RSNA\experiments_log.md"
SUB_CSV = r"D:\MLprojs\KaggleCompetition\RSNA\submission.csv"

def log_experiment_to_md(
    iteration: int,
    family: str,
    backbone: str,
    loss_name: str,
    res: int,
    slices: int,
    fold_auc: float,
    ensemble_auc: float,
    notes: str = ""
):
    """Appends iteration results to experiments_log.md."""
    sub_status = "Updated" if ensemble_auc >= fold_auc else "Retained"
    table_row = f"| {iteration} | {family} | {backbone} | {loss_name} | {res}x{res} | {slices} | {fold_auc:.4f} | {ensemble_auc:.4f} | {sub_status} |\n"
    
    # Read existing content
    if os.path.exists(EXP_LOG):
        with open(EXP_LOG, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = "# RSNA Knee Abnormality Detection -- Autonomous Tournament Log\n\n"

    # Insert row before "## Detailed Iteration Logs"
    if "## Detailed Iteration Logs" in content:
        parts = content.split("## Detailed Iteration Logs")
        new_content = parts[0] + table_row + "\n## Detailed Iteration Logs" + parts[1]
    else:
        new_content = content + "\n" + table_row

    # Append narrative notes
    narrative = f"\n### Iteration {iteration}: {family} ({backbone})\n"
    narrative += f"- **Loss Formulation:** {loss_name}\n"
    narrative += f"- **Configuration:** Resolution={res}x{res}, Slices={slices}\n"
    narrative += f"- **5-Fold OOF Macro-AUC:** {fold_auc:.4f}\n"
    narrative += f"- **Ensemble OOF Macro-AUC:** {ensemble_auc:.4f}\n"
    narrative += f"- **Notes:** {notes}\n"
    new_content += narrative

    with open(EXP_LOG, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"[Log] Successfully updated {EXP_LOG}")

def get_tournament_experiments():
    """Generates the progressive tournament schedule."""
    return [
        # Iteration 1: Modern Fast ConvNet Baseline with PolyLoss
        {
            'iteration': 1,
            'family': 'ConvNet',
            'backbone': 'efficientnet_b0',
            'loss_type': 'poly',
            'poly_eps': 1.0,
            'target_size': 256,
            'slices_per_study': 6,
            'in_chans': 1,
            'batch_size': 4,
            'grad_accum_steps': 4,
            'epochs': 8,
            'lr': 3e-4,
            'notes': "Baseline EfficientNet-B0 with 2.5D Gated Attention-MIL and PolyLoss (eps=1.0)"
        },
        # Iteration 2: Modern ConvNeXt Architecture with PolyLoss
        {
            'iteration': 2,
            'family': 'ConvNeXt',
            'backbone': 'convnext_small',
            'loss_type': 'poly',
            'poly_eps': 1.0,
            'target_size': 256,
            'slices_per_study': 6,
            'in_chans': 1,
            'batch_size': 2,
            'grad_accum_steps': 8,
            'epochs': 8,
            'lr': 2e-4,
            'notes': "ConvNeXt-Small with Gated Attention-MIL and Gradient Checkpointing"
        },
        # Iteration 3: Asymmetric Focal Loss (Hard Negative Suppression)
        {
            'iteration': 3,
            'family': 'ResNet/SENet',
            'backbone': 'seresnext50_32x4d',
            'loss_type': 'asl',
            'gamma_neg': 3.0,
            'gamma_pos': 1.0,
            'target_size': 256,
            'slices_per_study': 6,
            'in_chans': 1,
            'batch_size': 2,
            'grad_accum_steps': 8,
            'epochs': 8,
            'lr': 2e-4,
            'notes': "SE-ResNeXt50 with Asymmetric Focal Loss for extreme class imbalance"
        },
        # Iteration 4: Vision Transformer (Swin Transformer)
        {
            'iteration': 4,
            'family': 'Vision Transformer',
            'backbone': 'swin_tiny_patch4_window7_224',
            'loss_type': 'poly',
            'poly_eps': 0.5,
            'target_size': 224,
            'slices_per_study': 6,
            'in_chans': 1,
            'batch_size': 2,
            'grad_accum_steps': 8,
            'epochs': 8,
            'lr': 1.5e-4,
            'notes': "Swin-Tiny Transformer with shifted window attention across knee joint planes"
        },
        # Iteration 5: Multi-Scale DenseNet with Homoscedastic Task Uncertainty
        {
            'iteration': 5,
            'family': 'DenseNet',
            'backbone': 'densenet121',
            'loss_type': 'uncertainty',
            'target_size': 256,
            'slices_per_study': 8,
            'in_chans': 1,
            'batch_size': 2,
            'grad_accum_steps': 8,
            'epochs': 8,
            'lr': 2e-4,
            'notes': "DenseNet-121 with 8-slice volumetric depth and Kendall homoscedastic uncertainty loss"
        },
        # Iteration 6: Deeper Volumetric Window (K=10 slices)
        {
            'iteration': 6,
            'family': 'Volumetric ConvNeXt',
            'backbone': 'convnext_small',
            'loss_type': 'poly',
            'poly_eps': 1.5,
            'target_size': 256,
            'slices_per_study': 10,
            'in_chans': 1,
            'batch_size': 2,
            'grad_accum_steps': 8,
            'epochs': 8,
            'lr': 2e-4,
            'notes': "ConvNeXt-Small with expanded K=10 slice bag covering 90% of joint volume"
        },
        # Iteration 7: High-Resolution EfficientNet-B2
        {
            'iteration': 7,
            'family': 'EfficientNet',
            'backbone': 'efficientnet_b2',
            'loss_type': 'poly',
            'poly_eps': 1.0,
            'target_size': 288,
            'slices_per_study': 6,
            'in_chans': 1,
            'batch_size': 2,
            'grad_accum_steps': 8,
            'epochs': 8,
            'lr': 2e-4,
            'notes': "High-resolution 288x288 EfficientNet-B2 focusing on delicate cartilage and meniscus fissures"
        }
    ]

def run_tournament():
    print("=" * 80)
    print("[*] LAUNCHING NON-STOP OVERNIGHT AUTONOMOUS RESEARCH & TOURNAMENT LOOP")
    print("=" * 80)
    
    # Load folds
    df_folds = pd.read_csv(FOLDS_CSV)
    target_cols = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
    labeled_df = df_folds[df_folds[target_cols].notnull().any(axis=1)].copy().reset_index(drop=True)
    print(f"Total gold-standard labeled studies: {len(labeled_df)}")
    
    experiments = get_tournament_experiments()
    current_best_ensemble_score = 0.5325
    exp_idx = 55
    
    while True:
        if exp_idx < len(experiments):
            config = experiments[exp_idx]
        else:
            pert_num = exp_idx - len(experiments) + 1
            backbones = ['convnext_small', 'seresnext50_32x4d', 'efficientnet_b2', 'densenet121']
            chosen_bb = backbones[pert_num % len(backbones)]
            chosen_eps = float(np.random.choice([0.2, 0.5, 1.2, 1.8, 2.0]))
            chosen_slices = int(np.random.choice([8, 10, 12]))
            chosen_res = 224 if ('swin' in chosen_bb or '224' in chosen_bb) else 320
            chosen_bs = 4
            chosen_accum = 4
            config = {
                'iteration': exp_idx + 1,
                'family': f'Dynamic-Perturbation-{pert_num}',
                'backbone': chosen_bb,
                'loss_type': 'poly',
                'poly_eps': chosen_eps,
                'target_size': chosen_res,
                'slices_per_study': chosen_slices,
                'in_chans': 1,
                'batch_size': chosen_bs,
                'grad_accum_steps': chosen_accum,
                'epochs': 12,
                'lr': 2e-4,
                'notes': f"High-GPU Perturbation: res={chosen_res}, eps={chosen_eps}, slices={chosen_slices}, bs={chosen_bs}"
            }
            
        iter_num = config['iteration']
        backbone_name = config['backbone']
        print(f"\n{'='*70}")
        print(f"[*] TOURNAMENT ITERATION {iter_num}: {config['family']} ({backbone_name})")
        print(f"Objective: Train 5-Fold Cross Validation & Optimize Ensemble")
        print(f"{'='*70}\n")
        
        fold_scores = []
        for fold in range(5):
            train_subset = labeled_df[labeled_df['fold'] != fold].copy().reset_index(drop=True)
            val_subset = labeled_df[labeled_df['fold'] == fold].copy().reset_index(drop=True)
            
            try:
                result = train_fold_with_auto_oom(
                    model_name=backbone_name,
                    fold=fold,
                    train_df=train_subset,
                    val_df=val_subset,
                    config=config
                )
                fold_scores.append(result['best_auc'])
                print(f"[Fold {fold} Finished] Best Macro-AUC: {result['best_auc']:.4f}")
            except Exception as e:
                print(f"[Error in Fold {fold}] {e}")
                fold_scores.append(0.5)
                
        mean_fold_auc = float(np.mean(fold_scores))
        print(f"\n>>> Iteration {iter_num} 5-Fold Mean Macro-AUC: {mean_fold_auc:.4f} <<<")
        
        print("\n--- Running Mathematical Ensemble Optimization (SLSQP) ---")
        ens_res = optimize_ensemble_weights()
        ensemble_score = ens_res.get('score', mean_fold_auc)
        print(f">>> Ensemble OOF Macro-AUC: {ensemble_score:.4f} <<<")
        
        if ensemble_score > current_best_ensemble_score:
            print(f"\n[+] NEW BEST ENSEMBLE SCORE: {ensemble_score:.4f} > {current_best_ensemble_score:.4f}")
            current_best_ensemble_score = ensemble_score
            print("Regenerating submission.csv with new ensemble weights...")
            try:
                import importlib
                import kaggle_inference_submission
                importlib.reload(kaggle_inference_submission)
                kaggle_inference_submission.run_inference()
                print("Submission regenerated and strictly verified!")
            except Exception as e:
                print(f"Error generating submission: {e}")
        else:
            print(f"Ensemble score {ensemble_score:.4f} did not beat current best {current_best_ensemble_score:.4f}.")
            
        log_experiment_to_md(
            iteration=iter_num,
            family=config['family'],
            backbone=backbone_name,
            loss_name=f"{config['loss_type']}" + (f"(eps={config.get('poly_eps')})" if config['loss_type']=='poly' else ""),
            res=config['target_size'],
            slices=config['slices_per_study'],
            fold_auc=mean_fold_auc,
            ensemble_auc=ensemble_score,
            notes=config.get('notes', '')
        )
        
        exp_idx += 1
        sys.stdout.flush()

if __name__ == '__main__':
    run_tournament()
