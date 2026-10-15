import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tune_precision_lexicon import TARGET_RULES, extract_rule_score, clean_text

df = pd.read_csv('train.csv')
gold_df = df[df['ACL'].notnull()].copy().reset_index(drop=True)
gold_df['clean_report'] = gold_df['Report'].apply(clean_text)

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print(f"Inspecting errors on {len(gold_df)} gold studies:")
for c, target in enumerate(target_cols):
    y = gold_df[target].values
    scores = np.array([extract_rule_score(txt, target) for txt in gold_df['clean_report']])
    try:
        auc = roc_auc_score(y, scores)
    except Exception as e:
        auc = 0.5
    print(f"\n{'='*80}\nTarget: {target} (Current Rule AUC: {auc:.4f}) | Positive: {int(y.sum())}/{len(y)}\n{'='*80}")
    
    fp = np.where((scores > 0.5) & (y == 0))[0]
    fn = np.where((scores < 0.3) & (y == 1))[0]
    if len(fp) > 0:
        print(f"  False Positives ({len(fp)}):")
        for idx in fp[:3]:
            txt = gold_df.iloc[idx]['clean_report']
            print(f"    [FP {idx}] score={scores[idx]:.2f}: {txt[:250]}...")
    if len(fn) > 0:
        print(f"  False Negatives ({len(fn)}):")
        for idx in fn[:3]:
            txt = gold_df.iloc[idx]['clean_report']
            print(f"    [FN {idx}] score={scores[idx]:.2f}: {txt[:250]}...")
