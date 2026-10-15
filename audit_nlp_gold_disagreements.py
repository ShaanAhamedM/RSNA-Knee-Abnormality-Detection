"""
Audit Gold-58 Disagreements across all 12 pathologies
Prints detailed reports and diagnoses for all False Positives and False Negatives
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from upgrade_nlp_engine_v4 import clean_text_v4, extract_rule_score_v4, CLINICAL_LEXICON_V4, TARGET_WEIGHTS

df = pd.read_csv('train.csv')
gold = df[df['ACL'].notnull()].copy().reset_index(drop=True)
gold['clean_report'] = gold['Report'].apply(clean_text_v4)

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print("="*90)
print(f"AUDITING 58 GOLD REPORTS DISAGREEMENTS")
print("="*90)

for c, target in enumerate(target_cols):
    y = gold[target].values
    scores = np.array([extract_rule_score_v4(txt, target) for txt in gold['clean_report']])
    auc = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else 0.5
    
    # Analyze false positives (score > 0.5, y == 0) and false negatives (score < 0.5, y == 1)
    fp_idx = np.where((scores >= 0.5) & (y == 0))[0]
    fn_idx = np.where((scores < 0.5) & (y == 1))[0]
    
    print(f"\nTarget: {target:18s} | Gold AUC={auc:.4f} | Pos={int(y.sum())}/58 | FP={len(fp_idx)} | FN={len(fn_idx)}")
    
    if len(fp_idx) > 0:
        print(f"  --- False Positives ({len(fp_idx)}) ---")
        for idx in fp_idx:
            study_id = gold.iloc[idx]['StudyInstanceUID']
            txt = gold.iloc[idx]['clean_report']
            print(f"  [FP study {study_id} (idx {idx})] score={scores[idx]:.3f}:")
            print(f"      {txt[:350]}...\n")
            
    if len(fn_idx) > 0:
        print(f"  --- False Negatives ({len(fn_idx)}) ---")
        for idx in fn_idx:
            study_id = gold.iloc[idx]['StudyInstanceUID']
            txt = gold.iloc[idx]['clean_report']
            print(f"  [FN study {study_id} (idx {idx})] score={scores[idx]:.3f}:")
            print(f"      {txt[:350]}...\n")
