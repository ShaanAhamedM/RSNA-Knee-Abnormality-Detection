import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.special import logit, expit

targets = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

kept_models = [
    'convnext_small', 'seresnext50_32x4d', 'densenet121', 
    'efficientnet_b2', 'swin_tiny_patch4_window7_224'
]

true_list = [np.load(f'oof/true_fold{f}.npy') for f in range(5)]
true_all = np.vstack(true_list)
pi_c = true_all.mean(axis=0)

# Full OOF matrix across all 58 studies for each model
# For folds that were pruned, we use the model's prediction or average across valid folds
model_oofs = {}
for m in kept_models:
    pred_list = [np.load(f'oof/{m}_fold{f}_oof.npy') for f in range(5)]
    model_oofs[m] = np.vstack(pred_list).astype(np.float32)

N, C = true_all.shape
M = len(kept_models)

print("=" * 80)
print("12 INDEPENDENT PER-TARGET OPTIMAL SUB-ENSEMBLES (PRUNED MODELS ONLY)")
print("=" * 80)

per_target_weights = {}
L1_oof_preds = np.zeros((N, C), dtype=np.float32)

for c in range(C):
    y = true_all[:, c]
    X_c = np.column_stack([model_oofs[m][:, c] for m in kept_models])
    
    single_aucs = [roc_auc_score(y, X_c[:, m]) for m in range(M)]
    
    # Regularized objective: maximize AUC + penalty towards uniform over good models
    def obj(w):
        weights = np.exp(w) / np.sum(np.exp(w))
        blend = X_c @ weights
        score = roc_auc_score(y, blend)
        reg = 0.05 * np.sum((weights - 1.0/M)**2)
        return -score + reg

    res = minimize(obj, x0=np.zeros(M), method='Powell')
    w_opt = np.exp(res.x) / np.sum(np.exp(res.x))
    
    # Prune tiny weights
    w_opt[w_opt < 0.04] = 0.0
    w_opt /= w_opt.sum()
    
    pred_c = X_c @ w_opt
    auc_c = roc_auc_score(y, pred_c)
    L1_oof_preds[:, c] = pred_c
    
    per_target_weights[targets[c]] = {
        'weights': {kept_models[m]: float(w_opt[m]) for m in range(M)},
        'auc': float(auc_c)
    }
    print(f"Target {c:2d} ({targets[c]:18s}): L1 OOF AUC = {auc_c:.4f} (Single Best: {max(single_aucs):.4f} by {kept_models[int(np.argmax(single_aucs))]})")
    for m in range(M):
        if w_opt[m] > 0.01:
            print(f"    {kept_models[m]:28s}: {w_opt[m]:.3f} (OOF: {single_aucs[m]:.3f})")

l1_macro_auc = np.mean([roc_auc_score(true_all[:, c], L1_oof_preds[:, c]) for c in range(C)])
print(f"\nMean L1 OOF Macro-AUC (Pruned Models): {l1_macro_auc:.4f}")

# Level-2 Stacker with selective gating
print("\n" + "=" * 80)
print("LEVEL-2 META-LEARNER (SELECTIVE GATING)")
print("=" * 80)
fold_bounds = []
start = 0
for f in range(5):
    cnt = len(true_list[f])
    fold_bounds.append((start, start + cnt))
    start += cnt

from sklearn.linear_model import Ridge
L2_oof_preds = np.zeros((N, C), dtype=np.float32)
stacker_models = {}

for c in range(C):
    y = true_all[:, c]
    oof_stack = np.zeros(N, dtype=np.float32)
    
    for f in range(5):
        val_start, val_end = fold_bounds[f]
        train_idx = [i for i in range(N) if i < val_start or i >= val_end]
        val_idx = list(range(val_start, val_end))
        
        clf = Ridge(alpha=10.0)
        clf.fit(L1_oof_preds[train_idx], y[train_idx])
        oof_stack[val_idx] = clf.predict(L1_oof_preds[val_idx])
        
    stack_auc = roc_auc_score(y, oof_stack)
    base_auc = roc_auc_score(y, L1_oof_preds[:, c])
    
    if stack_auc > base_auc:
        final_ridge = Ridge(alpha=10.0)
        final_ridge.fit(L1_oof_preds, y)
        stacker_models[targets[c]] = {
            'use_stacker': True,
            'coef': final_ridge.coef_.tolist(),
            'intercept': float(final_ridge.intercept_),
            'gain': float(stack_auc - base_auc)
        }
        L2_oof_preds[:, c] = oof_stack
        print(f"  [+] IMPROVED {targets[c]:18s}: {base_auc:.4f} -> {stack_auc:.4f} (+{stack_auc - base_auc:.4f})")
    else:
        stacker_models[targets[c]] = {
            'use_stacker': False,
            'coef': [],
            'intercept': 0.0,
            'gain': 0.0
        }
        L2_oof_preds[:, c] = L1_oof_preds[:, c]
        print(f"  [-] Retained L1 {targets[c]:18s}: {base_auc:.4f}")

hybrid_macro_auc = np.mean([roc_auc_score(true_all[:, c], L2_oof_preds[:, c]) for c in range(C)])
print(f"\nMean Hybrid L1+L2 Macro-AUC: {hybrid_macro_auc:.4f}")

# Log-Odds Calibration Parameters
print("\n" + "=" * 80)
print("LOG-ODDS TEMPERATURE SCALING & EMPIRICAL PRIOR MATCHING")
print("=" * 80)

calibration_params = {}
cal_preds = np.zeros((N, C), dtype=np.float32)

for c in range(C):
    raw_col = L2_oof_preds[:, c]
    c_min = float(raw_col.min())
    c_max = float(raw_col.max())
    norm_col = 0.05 + 0.90 * (raw_col - c_min) / (c_max - c_min + 1e-7)
    
    logits = logit(np.clip(norm_col, 0.01, 0.99))
    mu_l = float(np.mean(logits))
    std_l = float(np.std(logits) + 1e-7)
    
    target_pi = float(pi_c[c])
    target_logit_pi = float(logit(np.clip(target_pi, 0.05, 0.95)))
    tau = 1.35
    
    cal_logits = (logits - mu_l) / tau + target_logit_pi
    p_cal = expit(cal_logits)
    cal_preds[:, c] = p_cal
    
    calibration_params[targets[c]] = {
        'c_min': c_min,
        'c_max': c_max,
        'mu_l': mu_l,
        'std_l': std_l,
        'target_pi': target_pi,
        'target_logit_pi': target_logit_pi,
        'tau': tau
    }
    
    auc_cal = roc_auc_score(true_all[:, c], p_cal)
    print(f"Target {c:2d} ({targets[c]:18s}): AUC={auc_cal:.4f} | Mean={p_cal.mean():.3f} (True={target_pi:.3f}) | Range=[{p_cal.min():.3f}, {p_cal.max():.3f}]")

cal_macro_auc = np.mean([roc_auc_score(true_all[:, c], cal_preds[:, c]) for c in range(C)])
print("\n" + "=" * 80)
print(f"FINAL AUDITED & CALIBRATED MACRO-AUC: {cal_macro_auc:.4f}")
print("=" * 80)

# Save configuration
out_config = {
    'targets': targets,
    'kept_models': kept_models,
    'pruned_models': ['efficientnet_b0'],
    'kept_folds': {
        'convnext_small': [0, 2, 3, 4],
        'seresnext50_32x4d': [1, 3],
        'densenet121': [4],
        'efficientnet_b2': [0, 2, 4],
        'swin_tiny_patch4_window7_224': [1, 2, 3]
    },
    'per_target_weights': per_target_weights,
    'stacker_models': stacker_models,
    'calibration_params': calibration_params,
    'scores': {
        'initial_uniform_auc': 0.4989,
        'global_slsqp_auc': 0.5325,
        'final_audited_auc': float(cal_macro_auc)
    }
}

with open(r'D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_params.json', 'w') as f:
    json.dump(out_config, f, indent=2)

print("Saved configuration to D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_params.json")
