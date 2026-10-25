import glob
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

targets = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

models = [
    'convnext_small', 'seresnext50_32x4d', 'densenet121', 
    'efficientnet_b2', 'swin_tiny_patch4_window7_224', 'efficientnet_b0'
]

true_list = [np.load(f'oof/true_fold{f}.npy') for f in range(5)]
true_all = np.vstack(true_list)  # (58, 12)

print("=" * 80)
print("1. EMPIRICAL PREVALENCE (PI_C) IN GOLD STANDARD (58 STUDIES)")
print("=" * 80)
pi_c = true_all.mean(axis=0)
for i, name in enumerate(targets):
    pos_count = int(true_all[:, i].sum())
    print(f"{name:20s}: {pos_count:2d}/58 positive ({pi_c[i]*100:5.1f}%)")

print("\n" + "=" * 80)
print("2. 5-FOLD CONCATENATED OOF MACRO-AUC PER MODEL FAMILY")
print("=" * 80)
model_oofs = {}
for m in models:
    pred_list = [np.load(f'oof/{m}_fold{f}_oof.npy') for f in range(5)]
    pred_all = np.vstack(pred_list).astype(np.float32)
    model_oofs[m] = pred_all
    
    col_aucs = []
    for c in range(12):
        try:
            score = roc_auc_score(true_all[:, c], pred_all[:, c])
        except Exception:
            score = 0.5
        col_aucs.append(score)
    macro_auc = np.mean(col_aucs)
    print(f"{m:30s}: Macro-AUC = {macro_auc:.4f}")
    col_details = " | ".join([f"{targets[c][:3]}:{col_aucs[c]:.2f}" for c in range(12)])
    print(f"   Targets: {col_details}")

print("\n" + "=" * 80)
print("3. INDIVIDUAL EVALUATION OF ALL 30 FOLD CHECKPOINTS")
print("=" * 80)
checkpoint_scores = {}
pruned_checkpoints = []
for m in models:
    for f in range(5):
        y_true = true_list[f]
        y_pred = np.load(f'oof/{m}_fold{f}_oof.npy').astype(np.float32)
        aucs = []
        for c in range(12):
            try:
                if len(np.unique(y_true[:, c])) > 1:
                    aucs.append(roc_auc_score(y_true[:, c], y_pred[:, c]))
                else:
                    aucs.append(0.5)
            except Exception:
                aucs.append(0.5)
        ckpt_auc = float(np.mean(aucs))
        ckpt_name = f"{m}_fold{f}"
        checkpoint_scores[ckpt_name] = ckpt_auc
        status = "KEEP" if ckpt_auc >= 0.52 else "PRUNE (<0.52)"
        if ckpt_auc < 0.52:
            pruned_checkpoints.append(ckpt_name)
        print(f"{ckpt_name:32s}: Fold OOF AUC = {ckpt_auc:.4f}  [{status}]")

print(f"\nPrune Summary: {len(pruned_checkpoints)} / 30 checkpoints have OOF AUC < 0.52")
print("Pruned list:", pruned_checkpoints)
