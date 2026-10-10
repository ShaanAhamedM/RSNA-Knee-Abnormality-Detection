"""
Mathematical Ensembling & Post-Hoc Probability Calibration
Uses Scipy Optimization (SLSQP / Nelder-Mead) to find optimal model weights w*
that directly maximize Kaggle Macro-ROC AUC across all 5 folds.
"""

import os
import glob
import json
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple

from src.metrics import calculate_macro_roc_auc

def optimize_ensemble_weights(
    oof_dir: str = r"D:\MLprojs\KaggleCompetition\RSNA\oof"
) -> Dict:
    """
    Finds optimal weights w* for all available model families to maximize Macro-ROC AUC.
    """
    # 1. Discover all unique model families in oof_dir
    oof_files = glob.glob(os.path.join(oof_dir, "*_fold*_oof.npy"))
    if not oof_files:
        print("[Ensemble] No OOF prediction files found yet.")
        return {'status': 'empty', 'score': 0.0}

    # Extract model names
    model_names = set()
    for f in oof_files:
        base = os.path.basename(f)
        m_name = base.split('_fold')[0]
        model_names.add(m_name)
    model_names = sorted(list(model_names))
    print(f"[Ensemble] Found {len(model_names)} model families: {model_names}")

    # Check which folds are complete across models
    fold_indices = list(range(5))
    complete_folds = []
    for fold in fold_indices:
        all_models_have_fold = True
        true_file = os.path.join(oof_dir, f"true_fold{fold}.npy")
        if not os.path.exists(true_file):
            continue
        for m_name in model_names:
            pred_file = os.path.join(oof_dir, f"{m_name}_fold{fold}_oof.npy")
            if not os.path.exists(pred_file):
                all_models_have_fold = False
                break
        if all_models_have_fold:
            complete_folds.append(fold)

    if not complete_folds:
        print("[Ensemble] Incomplete fold overlap across models. Evaluating single best model...")
        best_single_model = None
        best_single_auc = 0.0
        for m_name in model_names:
            m_preds = []
            m_trues = []
            for fold in fold_indices:
                p_f = os.path.join(oof_dir, f"{m_name}_fold{fold}_oof.npy")
                t_f = os.path.join(oof_dir, f"true_fold{fold}.npy")
                if os.path.exists(p_f) and os.path.exists(t_f):
                    m_preds.append(np.load(p_f))
                    m_trues.append(np.load(t_f))
            if m_preds and m_trues:
                y_p = np.concatenate(m_preds, axis=0)
                y_t = np.concatenate(m_trues, axis=0)
                score, _ = calculate_macro_roc_auc(y_t, y_p)
                print(f"  Model {m_name}: Partial Macro-AUC = {score:.4f}")
                if score > best_single_auc:
                    best_single_auc = score
                    best_single_model = m_name
        return {
            'status': 'partial',
            'best_model': best_single_model,
            'score': best_single_auc,
            'weights': {best_single_model: 1.0} if best_single_model else {}
        }

    print(f"[Ensemble] Optimizing weights across folds: {complete_folds}")
    
    # Load predictions for all complete folds
    all_y_true = []
    model_predictions = {m: [] for m in model_names}
    for fold in complete_folds:
        t_f = os.path.join(oof_dir, f"true_fold{fold}.npy")
        all_y_true.append(np.load(t_f))
        for m in model_names:
            p_f = os.path.join(oof_dir, f"{m}_fold{fold}_oof.npy")
            model_predictions[m].append(np.load(p_f))

    y_true_combined = np.concatenate(all_y_true, axis=0)
    y_preds_by_model = {m: np.concatenate(model_predictions[m], axis=0) for m in model_names}

    M = len(model_names)
    preds_stacked = np.stack([y_preds_by_model[m] for m in model_names], axis=0) # (M, N, 12)

    # Compute baseline individual model scores
    print("\n--- Baseline Individual Model Scores ---")
    single_scores = {}
    for i, m in enumerate(model_names):
        score, _ = calculate_macro_roc_auc(y_true_combined, y_preds_by_model[m])
        single_scores[m] = score
        print(f"  {m:25s}: Macro-AUC = {score:.4f}")

    if M == 1:
        weights = {model_names[0]: 1.0}
        best_score = single_scores[model_names[0]]
    else:
        # Objective function: negative macro-ROC AUC
        def loss_func(w):
            w = w / (np.sum(w) + 1e-8)
            blended = np.tensordot(w, preds_stacked, axes=([0], [0]))
            auc, _ = calculate_macro_roc_auc(y_true_combined, blended)
            return -auc

        # Initial equal weights
        w0 = np.ones(M) / M
        bounds = [(0.0, 1.0) for _ in range(M)]
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        res = minimize(
            loss_func, w0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-5}
        )

        w_opt = res.x / np.sum(res.x)
        best_score = -res.fun
        weights = {m: float(w_opt[i]) for i, m in enumerate(model_names)}

    print(f"\n[Ensemble Result] Optimal Macro-AUC: {best_score:.4f}")
    print("Optimal Weights:")
    for m, w in weights.items():
        print(f"  {m:25s}: {w:.4f}")

    # Save to disk
    out_file = os.path.join(oof_dir, "best_ensemble_weights.json")
    with open(out_file, 'w') as f:
        json.dump({'score': best_score, 'weights': weights, 'single_scores': single_scores}, f, indent=2)

    return {
        'status': 'optimized',
        'score': best_score,
        'weights': weights,
        'single_scores': single_scores
    }

if __name__ == '__main__':
    optimize_ensemble_weights()
