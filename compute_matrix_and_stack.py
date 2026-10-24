import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.stats import rankdata
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

true_list = [np.load(f'oof/true_fold{f}.npy') for f in range(5)]
true_all = np.vstack(true_list)  # (58, 12)
df_true = pd.DataFrame(true_all, columns=targets)

print("=" * 80)
print("1. EMPIRICAL CO-OCCURRENCE / CORRELATION MATRIX (12x12)")
print("=" * 80)
corr = df_true.corr()
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 160)
print(corr.round(2))

# Top correlated pairs
print("\nTop Pathologically Coupled Pairs (Anatomical Synergy):")
pairs = []
for i in range(12):
    for j in range(i + 1, 12):
        pairs.append((targets[i], targets[j], corr.iloc[i, j]))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for p1, p2, val in pairs[:8]:
    print(f"  {p1:18s} <---> {p2:18s} : r = {val:+.2f}")

# Load OOF predictions for each model family
model_oofs = {}
for m in models:
    pred_list = [np.load(f'oof/{m}_fold{f}_oof.npy') for f in range(5)]
    model_oofs[m] = np.vstack(pred_list).astype(np.float32)

M = len(models)
N = len(true_all)

print("\n" + "=" * 80)
print("2. BASELINE: UNIFORM AVERAGE vs CURRENT GLOBAL SLSQP")
print("=" * 80)
uniform_preds = np.mean([model_oofs[m] for m in models], axis=0)
uniform_aucs = [roc_auc_score(true_all[:, c], uniform_preds[:, c]) for c in range(12)]
print(f"Uniform Mean Ensemble Macro-AUC: {np.mean(uniform_aucs):.4f}")

# Prune models with low overall AUC: let's evaluate subsets of models
print("\n" + "=" * 80)
print("3. PER-TARGET INDEPENDENT OPTIMAL SUB-ENSEMBLING (12 OPTIMIZERS)")
print("=" * 80)

# For each target c in 0..11, optimize weights w in R^M to maximize ROC-AUC
# We optimize with Dirichlet/L2 regularization to prevent overfitting on N=58
# AUC is not smooth, so we can use smooth sigmoid surrogate or Nelder-Mead / Powell on negative AUC

best_per_target_weights = {}
per_target_prob_preds = np.zeros_like(uniform_preds)
per_target_rank_preds = np.zeros_like(uniform_preds)

for c in range(12):
    y_true = true_all[:, c]
    # Matrix of shape (N, M) for this column
    X_col = np.column_stack([model_oofs[m][:, c] for m in models])
    
    # Also rank-transformed version:
    X_col_rank = np.column_stack([rankdata(model_oofs[m][:, c]) / len(y_true) for m in models])
    
    # Evaluate individual models on this column
    single_aucs = [roc_auc_score(y_true, X_col[:, m_idx]) for m_idx in range(M)]
    
    def loss_func(w, X, y):
        # Softmax normalize w
        weights = np.exp(w) / np.sum(np.exp(w))
        blend = X @ weights
        try:
            # Negative ROC-AUC + L2 penalty towards uniform (1/M)
            score = roc_auc_score(y, blend)
            reg = 0.05 * np.sum((weights - 1.0/M)**2)
            return -score + reg
        except Exception:
            return 0.0

    # Optimize for probability blend
    res_prob = minimize(
        loss_func, 
        x0=np.zeros(M), 
        args=(X_col, y_true), 
        method='Powell'
    )
    w_prob = np.exp(res_prob.x) / np.sum(np.exp(res_prob.x))
    
    # Optimize for rank blend
    res_rank = minimize(
        loss_func, 
        x0=np.zeros(M), 
        args=(X_col_rank, y_true), 
        method='Powell'
    )
    w_rank = np.exp(res_rank.x) / np.sum(np.exp(res_rank.x))
    
    pred_p = X_col @ w_prob
    pred_r = X_col_rank @ w_rank
    
    auc_p = roc_auc_score(y_true, pred_p)
    auc_r = roc_auc_score(y_true, pred_r)
    
    best_weights = w_rank if auc_r >= auc_p else w_prob
    best_mode = 'rank' if auc_r >= auc_p else 'prob'
    best_pred = pred_r if auc_r >= auc_p else pred_p
    best_auc = max(auc_r, auc_p)
    
    per_target_prob_preds[:, c] = pred_p
    per_target_rank_preds[:, c] = pred_r
    best_per_target_weights[targets[c]] = {
        'weights': {models[m]: float(best_weights[m]) for m in range(M)},
        'mode': best_mode,
        'best_auc': float(best_auc),
        'single_best_auc': float(max(single_aucs)),
        'single_best_model': models[int(np.argmax(single_aucs))]
    }
    
    print(f"Target {c:2d} ({targets[c]:18s}): Best AUC = {best_auc:.4f} (Single Best: {max(single_aucs):.4f} by {models[int(np.argmax(single_aucs))]})")
    for m in range(M):
        if best_weights[m] > 0.05:
            print(f"    {models[m]:25s}: {best_weights[m]:.3f} (OOF: {single_aucs[m]:.3f})")

auc_per_target_prob = [roc_auc_score(true_all[:, c], per_target_prob_preds[:, c]) for c in range(12)]
auc_per_target_rank = [roc_auc_score(true_all[:, c], per_target_rank_preds[:, c]) for c in range(12)]

print("\n" + "=" * 80)
print(f"Summary Per-Target Probability Macro-AUC: {np.mean(auc_per_target_prob):.4f}")
print(f"Summary Per-Target Rank Macro-AUC:        {np.mean(auc_per_target_rank):.4f}")
print("=" * 80)
