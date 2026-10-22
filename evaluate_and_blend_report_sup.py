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

true_list = [np.load(f'oof/true_fold{f}.npy') for f in range(5)]
true_all = np.vstack(true_list)
pi_c = true_all.mean(axis=0)

models = [
    'convnext_small', 
    'convnext_small_report_sup',
    'seresnext50_32x4d', 
    'densenet121', 
    'efficientnet_b2', 
    'swin_tiny_patch4_window7_224'
]

model_oofs = {}
for m in models:
    pred_list = [np.load(f'oof/{m}_fold{f}_oof.npy') for f in range(5)]
    model_oofs[m] = np.vstack(pred_list).astype(np.float32)

print("=" * 90)
print("EVALUATING ALL 6 MODEL FAMILIES (INCLUDING REPORT-SUPERVISED CONVNEXT)")
print("=" * 90)
for m in models:
    aucs = []
    for c in range(12):
        try:
            score = roc_auc_score(true_all[:, c], model_oofs[m][:, c])
        except Exception:
            score = 0.5
        aucs.append(score)
    macro = np.mean(aucs)
    print(f"{m:30s}: Macro-AUC = {macro:.4f}")
    details = " | ".join([f"{targets[c][:3]}:{aucs[c]:.2f}" for c in range(12)])
    print(f"   Targets: {details}")

N, C = true_all.shape
M = len(models)

print("\n" + "=" * 90)
print("STAGE 4: OPTIMIZING 12 INDEPENDENT PER-TARGET SUB-ENSEMBLES")
print("=" * 90)

per_target_weights = {}
L1_oof_preds = np.zeros((N, C), dtype=np.float32)

for c, t in enumerate(targets):
    y = true_all[:, c]
    X_c = np.column_stack([model_oofs[m][:, c] for m in models])
    single_aucs = [roc_auc_score(y, X_c[:, m]) for m in range(M)]
    
    # Regularized optimization: Dirichlet/L2 penalty
    def obj(w):
        weights = np.exp(w) / np.sum(np.exp(w))
        # Cap max weight at 0.65 to ensure ensemble diversity
        weights = np.clip(weights, 0.0, 0.65)
        weights /= weights.sum()
        blend = X_c @ weights
        score = roc_auc_score(y, blend)
        reg = 0.06 * np.sum((weights - 1.0/M)**2)
        return -score + reg

    res = minimize(obj, x0=np.zeros(M), method='Powell')
    w_opt = np.exp(res.x) / np.sum(np.exp(res.x))
    w_opt = np.clip(w_opt, 0.0, 0.65)
    w_opt[w_opt < 0.04] = 0.0
    w_opt /= w_opt.sum()
    
    # Ensure at least 2 models active
    if (w_opt > 0.05).sum() < 2:
        ranked = np.argsort(single_aucs)[::-1]
        w_opt = np.zeros(M)
        w_opt[ranked[0]] = 0.60
        w_opt[ranked[1]] = 0.40
        
    pred_c = X_c @ w_opt
    auc_c = roc_auc_score(y, pred_c)
    L1_oof_preds[:, c] = pred_c
    
    per_target_weights[t] = {
        'weights': {models[m]: float(w_opt[m]) for m in range(M)},
        'auc': float(auc_c)
    }
    
    active = [f"{models[m]}:{w_opt[m]:.2f}" for m in range(M) if w_opt[m] > 0]
    print(f"{t:18s}: L1 AUC = {auc_c:.4f} | Models: {', '.join(active)}")

macro_l1 = np.mean([roc_auc_score(true_all[:, c], L1_oof_preds[:, c]) for c in range(C)])
print(f"\nMean Stage 4 L1 OOF Macro-AUC: {macro_l1:.4f}")

# Re-fit exact log-odds calibration parameters
print("\n" + "=" * 90)
print("FITTING EXACT LOG-ODDS TEMPERATURE & PRIOR MATCHING")
print("=" * 90)

new_cal_params = {}
cal_preds = np.zeros((N, C), dtype=np.float32)

for c, t in enumerate(targets):
    raw = L1_oof_preds[:, c]
    p_safe = np.clip(raw, 1e-4, 1.0 - 1e-4)
    l_raw = logit(p_safe)
    mu_logit = float(l_raw.mean())
    std_logit = float(l_raw.std())
    
    target_pi = float(pi_c[c])
    target_logit_pi = float(logit(target_pi))
    tau = 0.85
    
    cal_logits = (l_raw - mu_logit) / tau + target_logit_pi
    p_cal = expit(cal_logits)
    cal_preds[:, c] = p_cal
    
    auc = roc_auc_score(true_all[:, c], p_cal)
    new_cal_params[t] = {
        'mu_logit': mu_logit,
        'std_logit': std_logit,
        'target_pi': target_pi,
        'target_logit_pi': target_logit_pi,
        'tau': tau
    }
    print(f"{t:18s}: AUC = {auc:.4f} | Cal Mean = {p_cal.mean():.3f} (True={target_pi:.3f}) | Range = [{p_cal.min():.3f}, {p_cal.max():.3f}]")

macro_cal = np.mean([roc_auc_score(true_all[:, c], cal_preds[:, c]) for c in range(C)])
print("\n" + "=" * 90)
print(f"STAGE 4 FINAL CALIBRATED OOF MACRO-AUC: {macro_cal:.4f}")
print("=" * 90)

# Save updated grandmaster_params.json
with open(r'D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_params.json', 'r') as f:
    gm = json.load(f)

gm['kept_models'] = models
gm['kept_folds']['convnext_small_report_sup'] = [0, 1, 2, 3, 4]
gm['per_target_weights'] = per_target_weights
gm['calibration_params'] = new_cal_params
gm['scores']['stage4_grandmaster_auc'] = float(macro_cal)

with open(r'D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_params.json', 'w') as f:
    json.dump(gm, f, indent=2)

print("\nSuccessfully updated D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_params.json with all 6 model families!")
