"""
Evaluation Metrics for RSNA Knee Abnormality Detection
Macro-Averaged Area Under the ROC Curve (Macro-AUC ROC)
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple, List

TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def calculate_macro_roc_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = TARGET_COLUMNS
) -> Tuple[float, Dict[str, float]]:
    """
    Computes exact Kaggle competition metric:
    Macro-Averaged ROC-AUC across all 12 knee abnormality classes.
    """
    per_class_auc = {}
    valid_aucs = []
    
    for i, col in enumerate(target_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        # Check if both classes are present
        if len(np.unique(y_t)) > 1:
            try:
                score = roc_auc_score(y_t, y_p)
                per_class_auc[col] = float(score)
                valid_aucs.append(score)
            except Exception as e:
                per_class_auc[col] = 0.5
                valid_aucs.append(0.5)
        else:
            per_class_auc[col] = np.nan
            
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5
    return macro_auc, per_class_auc
