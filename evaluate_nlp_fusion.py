"""
Evaluates Clean Precision Hybrid NLP Teacher (V4 + Transliteration)
Tests whether fusing V4 rules with multilingual transliteration boosts Gold-58 AUC.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from clinical_multilingual_cleaner import clean_clinical_text_v7
from upgrade_nlp_engine_v4 import clean_text_v4, extract_rule_features_v4, CLINICAL_LEXICON_V4

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

df = pd.read_csv('train.csv')
gold_df = df[df['ACL'].notnull()].copy().reset_index(drop=True)
gold_df['clean_v4'] = gold_df['Report'].apply(clean_text_v4)
gold_df['clean_v7'] = gold_df['Report'].apply(clean_clinical_text_v7)

N_gold = len(gold_df)
conf_v4 = np.zeros((N_gold, 12))
pos_v4 = np.zeros((N_gold, 12))
neg_v4 = np.zeros((N_gold, 12))

conf_v7 = np.zeros((N_gold, 12))
pos_v7 = np.zeros((N_gold, 12))
neg_v7 = np.zeros((N_gold, 12))

for i in range(N_gold):
    t4 = gold_df.iloc[i]['clean_v4']
    t7 = gold_df.iloc[i]['clean_v7']
    for c, target in enumerate(target_cols):
        c4, p4, n4 = extract_rule_features_v4(t4, target)
        conf_v4[i, c] = c4; pos_v4[i, c] = p4; neg_v4[i, c] = n4
        c7, p7, n7 = extract_rule_features_v4(t7, target)
        conf_v7[i, c] = c7; pos_v7[i, c] = p7; neg_v7[i, c] = n7

# Transliteration improved foreign studies: take maximum confidence
conf_fused = np.maximum(conf_v4, conf_v7)
# But if either says strongly negative (<= 0.05) and the other is prior (0.20), take negative
for i in range(N_gold):
    for c in range(12):
        if (conf_v4[i, c] <= 0.05 and conf_v7[i, c] == 0.20) or (conf_v7[i, c] <= 0.05 and conf_v4[i, c] == 0.20):
            conf_fused[i, c] = 0.02

# Evaluate pure rule AUC
rule_aucs_v4 = [roc_auc_score(gold_df[target_cols[c]].values, conf_v4[:, c]) for c in range(12)]
rule_aucs_fused = [roc_auc_score(gold_df[target_cols[c]].values, conf_fused[:, c]) for c in range(12)]

print("="*75)
print(f"PURE RULE MACRO-AUC: V4={np.mean(rule_aucs_v4):.4f} -> FUSED={np.mean(rule_aucs_fused):.4f}")
print("="*75)
for c, target in enumerate(target_cols):
    y = gold_df[target].values
    print(f"  {target:18s}: V4={rule_aucs_v4[c]:.4f} | Fused={rule_aucs_fused[c]:.4f} (Pos: {int(y.sum())}/58)")
