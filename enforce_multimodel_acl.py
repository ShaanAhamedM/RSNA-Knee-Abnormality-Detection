import json
import numpy as np
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score

with open(r'D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_params.json', 'r') as f:
    gm = json.load(f)

# Re-balance ACL weights across 4 architectures
gm['per_target_weights']['ACL']['weights'] = {
    'convnext_small': 0.40,
    'convnext_small_report_sup': 0.15,
    'seresnext50_32x4d': 0.25,
    'efficientnet_b2': 0.20
}

# Verify OOF AUC
true_list = [np.load(f'oof/true_fold{f}.npy') for f in range(5)]
true_all = np.vstack(true_list)
y_acl = true_all[:, 0]

p_conv = np.vstack([np.load(f'oof/convnext_small_fold{f}_oof.npy') for f in range(5)])[:, 0]
p_rep = np.vstack([np.load(f'oof/convnext_small_report_sup_fold{f}_oof.npy') for f in range(5)])[:, 0]
p_ser = np.vstack([np.load(f'oof/seresnext50_32x4d_fold{f}_oof.npy') for f in range(5)])[:, 0]
p_eff = np.vstack([np.load(f'oof/efficientnet_b2_fold{f}_oof.npy') for f in range(5)])[:, 0]

blend_acl = 0.40 * p_conv + 0.15 * p_rep + 0.25 * p_ser + 0.20 * p_eff
auc_acl = roc_auc_score(y_acl, blend_acl)
gm['per_target_weights']['ACL']['auc'] = float(auc_acl)

l_raw = logit(np.clip(blend_acl, 1e-4, 1.0 - 1e-4))
gm['calibration_params']['ACL']['mu_logit'] = float(l_raw.mean())
gm['calibration_params']['ACL']['std_logit'] = float(l_raw.std())

print(f"ACL Multi-Model Blend OOF AUC: {auc_acl:.4f}")

with open(r'D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_params.json', 'w') as f:
    json.dump(gm, f, indent=2)

print("Saved updated grandmaster_params.json!")
