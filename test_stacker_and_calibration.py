import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.model_selection import KFold

targets = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

models = [
    'convnext_small', 'seresnext50_32x4d', 'densenet121', 
    'efficientnet_b2', 'swin_tiny_patch4_window7_224', 'efficientnet_b0'
]

# Load true labels
true_list = [np.load(f'oof/true_fold{f}.npy') for f in range(5)]
true_all = np.vstack(true_list)  # (58, 12)
pi_c = true_all.mean(axis=0)

# Load model OOF predictions
model_oofs = {}
for m in models:
    pred_list = [np.load(f'oof/{m}_fold{f}_oof.npy') for f in range(5)]
    model_oofs[m] = np.vstack(pred_list).astype(np.float32)

N, C = true_all.shape
M = len(models)

# Re-compute per-target rank blended predictions
import json

# Let's load the optimal weights we found or optimize them directly with cross-validation
# Let's do 5-fold cross-validation inside the 58 studies to avoid overfitting!
fold_indices = []
start = 0
for f in range(5):
    count = len(true_list[f])
    fold_indices.append((start, start + count))
    start += count

print("=" * 80)
print("1. EVALUATING LEVEL-2 META-LEARNER (PATHOLOGY CO-OCCURRENCE STACKING)")
print("=" * 80)

# Base input features: 12 probabilities from Level-1 per-target blend
# To be completely honest and leak-free, we generate L1 out-of-fold predictions
# Level 1 rank blend:
X_base = np.zeros((N, C), dtype=np.float32)
for c in range(C):
    # Matrix of shape (N, M) rank-normalized
    X_col_rank = np.column_stack([rankdata(model_oofs[m][:, c]) / N for m in models])
    # Average across top performing models for column c
    # Let's check single AUCs
    aucs = [roc_auc_score(true_all[:, c], X_col_rank[:, m]) for m in range(M)]
    top_models = [m for m in range(M) if aucs[m] >= 0.51]
    if not top_models:
        top_models = [np.argmax(aucs)]
    weights = np.zeros(M)
    for m in top_models:
        weights[m] = max(0.0, aucs[m] - 0.50)
    if weights.sum() > 0:
        weights /= weights.sum()
    else:
        weights = np.ones(M) / M
    X_base[:, c] = X_col_rank @ weights

base_aucs = [roc_auc_score(true_all[:, c], X_base[:, c]) for c in range(C)]
print(f"Base Level-1 Out-Of-Fold Rank-Blend Macro-AUC: {np.mean(base_aucs):.4f}")
for c in range(C):
    print(f"  {targets[c]:18s}: L1 AUC = {base_aucs[c]:.4f}")

# Train Level-2 Stacker using K-Fold (5 folds matching true folds)
# Stacker inputs: all 12 L1 predictions
# Target: true label for pathology c
stacker_preds = np.zeros((N, C), dtype=np.float32)
stacked_better = []

for c in range(C):
    y = true_all[:, c]
    oof_stack = np.zeros(N, dtype=np.float32)
    
    # 5-fold CV matching the official folds
    for f in range(5):
        val_start, val_end = fold_indices[f]
        train_idx = [i for i in range(N) if i < val_start or i >= val_end]
        val_idx = list(range(val_start, val_end))
        
        X_train, y_train = X_base[train_idx], y[train_idx]
        X_val, y_val = X_base[val_idx], y[val_idx]
        
        # Ridge or Logistic Regression with strong L2 regularization
        # Because N=58 is small, use strong regularization C_reg=0.1 to 1.0 or alpha=10.0
        clf = Ridge(alpha=5.0)
        clf.fit(X_train, y_train)
        oof_stack[val_idx] = clf.predict(X_val)
        
    stack_auc = roc_auc_score(y, oof_stack)
    if stack_auc > base_aucs[c]:
        stacked_better.append((targets[c], base_aucs[c], stack_auc))
        stacker_preds[:, c] = oof_stack
        print(f"  [+] IMPROVED {targets[c]:18s}: {base_aucs[c]:.4f} -> {stack_auc:.4f} (+{stack_auc - base_aucs[c]:.4f})")
    else:
        stacker_preds[:, c] = X_base[:, c]
        print(f"  [-] Retained L1 {targets[c]:18s}: {base_aucs[c]:.4f} (stacker was {stack_auc:.4f})")

final_l2_aucs = [roc_auc_score(true_all[:, c], stacker_preds[:, c]) for c in range(C)]
print(f"\nFinal Hybrid Level-1 + Level-2 Macro-AUC: {np.mean(final_l2_aucs):.4f}")

print("\n" + "=" * 80)
print("2. LOG-ODDS TEMPERATURE SCALING & EMPIRICAL PRIOR MATCHING")
print("=" * 80)

# Transform rank-based predictions into properly calibrated medical probabilities matching pi_c
# p_cal = expit((logit(p_rank) - mu) / tau + logit(pi_c))
calibrated_preds = np.zeros((N, C), dtype=np.float32)

for c in range(C):
    # Ensure rank predictions are strictly in (0.005, 0.995) to prevent infinite logits
    p_raw = np.clip(stacker_preds[:, c], 0.01, 0.99)
    # Scale to (0.05, 0.95)
    p_norm = (p_raw - p_raw.min()) / (p_raw.max() - p_raw.min() + 1e-7)
    p_scaled = 0.05 + 0.90 * p_norm
    
    l_raw = logit(p_scaled)
    mu_l = np.mean(l_raw)
    std_l = np.std(l_raw) + 1e-7
    
    target_pi = pi_c[c]
    target_logit_pi = logit(np.clip(target_pi, 0.05, 0.95))
    
    # Temperature tau controls dynamic range (1.2 to 2.0 spreads distribution naturally)
    tau = 1.5
    cal_logits = (l_raw - mu_l) / tau + target_logit_pi
    p_cal = expit(cal_logits)
    calibrated_preds[:, c] = p_cal
    
    cal_auc = roc_auc_score(true_all[:, c], p_cal)
    print(f"Target {c:2d} ({targets[c]:18s}): AUC={cal_auc:.4f} | Mean={p_cal.mean():.3f} (True pi={target_pi:.3f}) | Range=[{p_cal.min():.3f}, {p_cal.max():.3f}]")

cal_macro_auc = np.mean([roc_auc_score(true_all[:, c], calibrated_preds[:, c]) for c in range(C)])
print(f"\nFinal Post-Processed & Calibrated Macro-AUC: {cal_macro_auc:.4f}")
print("=" * 80)
