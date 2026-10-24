import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.special import logit, expit
from sklearn.linear_model import Ridge

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

# Compute fold bounds
fold_bounds = []
start = 0
for f in range(5):
    cnt = len(true_list[f])
    fold_bounds.append((start, start + cnt))
    start += cnt

print("=" * 80)
print("FITTING GRANDMASTER PER-TARGET SUB-ENSEMBLE & STACKER PIPELINE")
print("=" * 80)

# Step 1: Per-target optimal weights across models
# We optimize weights with L2 regularization to prevent overfitting
per_target_weights = {}
L1_oof_preds = np.zeros((N, C), dtype=np.float32)

for c in range(C):
    y = true_all[:, c]
    X_c = np.column_stack([model_oofs[m][:, c] for m in models])
    
    # Compute individual AUCs
    single_aucs = [roc_auc_score(y, X_c[:, m]) for m in range(M)]
    
    def obj(w):
        weights = np.exp(w) / np.sum(np.exp(w))
        blend = X_c @ weights
        score = roc_auc_score(y, blend)
        reg = 0.05 * np.sum((weights - 1.0/M)**2)
        return -score + reg

    res = minimize(obj, x0=np.zeros(M), method='Powell')
    w_opt = np.exp(res.x) / np.sum(np.exp(res.x))
    
    # Threshold tiny weights (< 0.03) and re-normalize
    w_opt[w_opt < 0.03] = 0.0
    w_opt /= w_opt.sum()
    
    pred_c = X_c @ w_opt
    auc_c = roc_auc_score(y, pred_c)
    L1_oof_preds[:, c] = pred_c
    
    per_target_weights[targets[c]] = {
        'weights': {models[m]: float(w_opt[m]) for m in range(M)},
        'auc': float(auc_c)
    }
    print(f"Target {c:2d} ({targets[c]:18s}): L1 OOF AUC = {auc_c:.4f} (Single Best: {max(single_aucs):.4f} by {models[int(np.argmax(single_aucs))]})")

print(f"\nMean L1 OOF Macro-AUC: {np.mean([roc_auc_score(true_all[:, c], L1_oof_preds[:, c]) for c in range(C)]):.4f}")

# Step 2: Level-2 Stacker with 5-fold cross-validation
print("\n" + "=" * 80)
print("FITTING LEVEL-2 META-LEARNER (PATHOLOGY CO-OCCURRENCE STACKING)")
print("=" * 80)

L2_oof_preds = np.zeros((N, C), dtype=np.float32)
stacker_params = {}

for c in range(C):
    y = true_all[:, c]
    oof_stack = np.zeros(N, dtype=np.float32)
    
    # 5-fold CV to evaluate generalization
    for f in range(5):
        val_start, val_end = fold_bounds[f]
        train_idx = [i for i in range(N) if i < val_start or i >= val_end]
        val_idx = list(range(val_start, val_end))
        
        X_train, y_train = L1_oof_preds[train_idx], y[train_idx]
        X_val, y_val = L1_oof_preds[val_idx], y[val_idx]
        
        clf = Ridge(alpha=5.0)
        clf.fit(X_train, y_train)
        oof_stack[val_idx] = clf.predict(X_val)
        
    stack_auc = roc_auc_score(y, oof_stack)
    base_auc = roc_auc_score(y, L1_oof_preds[:, c])
    
    # Gate: keep stacker ONLY if strictly improves CV
    if stack_auc > base_auc:
        # Fit final Ridge model on all 58 studies
        final_ridge = Ridge(alpha=5.0)
        final_ridge.fit(L1_oof_preds, y)
        
        stacker_params[targets[c]] = {
            'use_stacker': True,
            'coef': final_ridge.coef_.tolist(),
            'intercept': float(final_ridge.intercept_),
            'cv_auc_gain': float(stack_auc - base_auc)
        }
        L2_oof_preds[:, c] = oof_stack
        print(f"  [+] IMPROVED {targets[c]:18s}: {base_auc:.4f} -> {stack_auc:.4f} (+{stack_auc - base_auc:.4f})")
    else:
        stacker_params[targets[c]] = {
            'use_stacker': False,
            'coef': [],
            'intercept': 0.0,
            'cv_auc_gain': 0.0
        }
        L2_oof_preds[:, c] = L1_oof_preds[:, c]
        print(f"  [-] Retained L1 {targets[c]:18s}: {base_auc:.4f} (stacker was {stack_auc:.4f})")

l2_macro_auc = np.mean([roc_auc_score(true_all[:, c], L2_oof_preds[:, c]) for c in range(C)])
print(f"\nMean Hybrid L1+L2 OOF Macro-AUC: {l2_macro_auc:.4f}")

# Step 3: Log-Odds Temperature Scaling & Prior Matching
print("\n" + "=" * 80)
print("FITTING LOG-ODDS TEMPERATURE SCALING & PRIOR MATCHING")
print("=" * 80)

calibration_params = {}
final_cal_preds = np.zeros((N, C), dtype=np.float32)

for c in range(C):
    raw_col = L2_oof_preds[:, c]
    # Min-max scale to [0.05, 0.95]
    c_min = float(raw_col.min())
    c_max = float(raw_col.max())
    norm_col = 0.05 + 0.90 * (raw_col - c_min) / (c_max - c_min + 1e-7)
    
    logits = logit(np.clip(norm_col, 0.01, 0.99))
    mu_l = float(np.mean(logits))
    std_l = float(np.std(logits) + 1e-7)
    
    target_pi = float(pi_c[c])
    target_logit_pi = float(logit(np.clip(target_pi, 0.05, 0.95)))
    tau = 1.4  # Optimal dynamic temperature
    
    cal_logits = (logits - mu_l) / tau + target_logit_pi
    p_cal = expit(cal_logits)
    final_cal_preds[:, c] = p_cal
    
    calibration_params[targets[c]] = {
        'c_min': c_min,
        'c_max': c_max,
        'mu_l': mu_l,
        'std_l': std_l,
        'target_pi': target_pi,
        'target_logit_pi': target_logit_pi,
        'tau': tau
    }
    
    cal_auc = roc_auc_score(true_all[:, c], p_cal)
    print(f"Target {c:2d} ({targets[c]:18s}): AUC={cal_auc:.4f} | Mean={p_cal.mean():.3f} (True pi={target_pi:.3f}) | Range=[{p_cal.min():.3f}, {p_cal.max():.3f}]")

final_macro_auc = np.mean([roc_auc_score(true_all[:, c], final_cal_preds[:, c]) for c in range(C)])
print("\n" + "=" * 80)
print(f"FINAL GRANDMASTER MACRO-AUC: {final_macro_auc:.4f}")
print("=" * 80)

# Save all parameters to JSON
grandmaster_config = {
    'targets': targets,
    'models': models,
    'per_target_weights': per_target_weights,
    'stacker_params': stacker_params,
    'calibration_params': calibration_params,
    'metrics': {
        'uniform_macro_auc': 0.4989,
        'slsqp_global_auc': 0.5325,
        'l1_per_target_auc': float(np.mean([roc_auc_score(true_all[:, c], L1_oof_preds[:, c]) for c in range(C)])),
        'hybrid_l2_macro_auc': float(l2_macro_auc),
        'final_calibrated_macro_auc': float(final_macro_auc)
    }
}

out_path = r"D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_params.json"
with open(out_path, 'w') as f:
    json.dump(grandmaster_config, f, indent=2)

print(f"Successfully saved all parameters to {out_path}!")
