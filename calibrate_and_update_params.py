import json
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score

targets = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

true_list = [np.load(f'oof/true_fold{f}.npy') for f in range(5)]
true_all = np.vstack(true_list)
pi_c = true_all.mean(axis=0)

kept_models = [
    'convnext_small', 'seresnext50_32x4d', 'densenet121', 
    'efficientnet_b2', 'swin_tiny_patch4_window7_224'
]

model_oofs = {
    m: np.vstack([np.load(f'oof/{m}_fold{f}_oof.npy') for f in range(5)]).astype(np.float32) 
    for m in kept_models
}

with open(r'D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_params.json', 'r') as f:
    gm = json.load(f)

per_target_weights = gm['per_target_weights']

N = len(true_all)
L1_preds = np.zeros((N, 12), dtype=np.float32)

for c, t in enumerate(targets):
    w_dict = per_target_weights[t]['weights']
    for m in kept_models:
        L1_preds[:, c] += w_dict.get(m, 0.0) * model_oofs[m][:, c]

print("=" * 90)
print("EXACT LOG-ODDS TEMPERATURE SCALING & PRIOR MATCHING (NO MIN-MAX TRUNCATION)")
print("=" * 90)

new_cal_params = {}
cal_preds = np.zeros((N, 12), dtype=np.float32)

for c, t in enumerate(targets):
    raw = L1_preds[:, c]
    # Smooth clip to avoid -inf/+inf
    p_safe = np.clip(raw, 1e-4, 1.0 - 1e-4)
    l_raw = logit(p_safe)
    mu_logit = float(l_raw.mean())
    std_logit = float(l_raw.std())
    
    target_pi = float(pi_c[c])
    target_logit_pi = float(logit(target_pi))
    
    # Temperature scaling factor tau_c:
    # tau_c in [0.7, 1.0] to naturally stretch dynamic range while preserving ROC-AUC
    # We choose tau = 0.85 to open up the dynamic range from [0.27, 0.59] to [0.05, 0.85]
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
    
    print(f"{t:18s}: raw=[{raw.min():.3f}, {raw.max():.3f}], mu_l={mu_logit:+.3f} -> cal=[{p_cal.min():.3f}, {p_cal.max():.3f}], mean={p_cal.mean():.3f} (True={target_pi:.3f}), AUC={auc:.4f}")

mean_auc = np.mean([roc_auc_score(true_all[:, c], cal_preds[:, c]) for c in range(12)])
print(f"\nFinal Calibrated OOF Macro-AUC: {mean_auc:.4f}")

gm['calibration_params'] = new_cal_params
gm['scores']['final_audited_auc'] = float(mean_auc)

with open(r'D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_params.json', 'w') as f:
    json.dump(gm, f, indent=2)

print("Updated grandmaster_params.json with exact formula!")
