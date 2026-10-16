"""
Generate High-Confidence Sharpened Pathology Labels for all 4,349 Unlabeled Reports.
Uses Upgraded NLP Extractor V3 with temperature sharpening (tau = 0.25) and confident clamps (0.02 / 0.98).
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit

from build_upgraded_nlp_extractor import (
    clean_text_v3, extract_rule_features_v3, CLINICAL_LEXICON, target_cols
)

df = pd.read_csv('train.csv')
gold_mask = df['ACL'].notnull()
gold_df = df[gold_mask].copy().reset_index(drop=True)
unlabeled_df = df[~gold_mask].copy().reset_index(drop=True)

gold_df['clean_report'] = gold_df['Report'].apply(clean_text_v3)
unlabeled_df['clean_report'] = unlabeled_df['Report'].apply(clean_text_v3)

print("=" * 80)
print(f"EXTRACTING SHARPENED LABELS FOR {len(unlabeled_df)} CLINICAL REPORTS")
print("=" * 80)

# 1. Rule features
N_gold = len(gold_df)
N_unlabeled = len(unlabeled_df)

gold_conf = np.zeros((N_gold, 12))
gold_pos = np.zeros((N_gold, 12))
gold_neg = np.zeros((N_gold, 12))

unlabeled_conf = np.zeros((N_unlabeled, 12))
unlabeled_pos = np.zeros((N_unlabeled, 12))
unlabeled_neg = np.zeros((N_unlabeled, 12))

print("Extracting clinical rule features on gold reports...")
for i in range(N_gold):
    txt = gold_df.iloc[i]['clean_report']
    for c, target in enumerate(target_cols):
        conf, pos_s, neg_s = extract_rule_features_v3(txt, target)
        gold_conf[i, c] = conf
        gold_pos[i, c] = pos_s
        gold_neg[i, c] = neg_s

print("Extracting clinical rule features on 4,349 unlabeled reports...")
for i in range(N_unlabeled):
    txt = unlabeled_df.iloc[i]['clean_report']
    for c, target in enumerate(target_cols):
        conf, pos_s, neg_s = extract_rule_features_v3(txt, target)
        unlabeled_conf[i, c] = conf
        unlabeled_pos[i, c] = pos_s
        unlabeled_neg[i, c] = neg_s

# 2. Multi-scale TF-IDF
vec_word = TfidfVectorizer(ngram_range=(1, 3), max_features=3500, sublinear_tf=True)
vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', max_features=4500, sublinear_tf=True)

X_word_gold = vec_word.fit_transform(gold_df['clean_report']).toarray()
X_char_gold = vec_char.fit_transform(gold_df['clean_report']).toarray()

X_word_unlabeled = vec_word.transform(unlabeled_df['clean_report']).toarray()
X_char_unlabeled = vec_char.transform(unlabeled_df['clean_report']).toarray()

# 3. Model training on Gold 58 and inference on Unlabeled 4,349
raw_probs = np.zeros((N_unlabeled, 12))

for c, target in enumerate(target_cols):
    y = gold_df[target].values
    X_tr = np.hstack([
        X_word_gold, X_char_gold, 
        gold_conf[:, c:c+1], gold_pos[:, c:c+1], gold_neg[:, c:c+1],
        gold_conf
    ])
    X_te = np.hstack([
        X_word_unlabeled, X_char_unlabeled, 
        unlabeled_conf[:, c:c+1], unlabeled_pos[:, c:c+1], unlabeled_neg[:, c:c+1],
        unlabeled_conf
    ])
    
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    clf.fit(X_tr, y)
    ml_prob = clf.predict_proba(X_te)[:, 1]
    
    # Blend with rule confidence
    w_r = 0.55 if target in ['ACL', 'MCL', "Baker's", 'Fracture'] else 0.40
    raw_probs[:, c] = w_r * unlabeled_conf[:, c] + (1.0 - w_r) * ml_prob

# 4. Temperature Sharpening (tau = 0.25) & Confident Clamping
print("\nApplying Log-Odds Temperature Sharpening (tau = 0.25) and Binary Clamping...")
sharpened_probs = np.zeros_like(raw_probs)

for c in range(12):
    p = np.clip(raw_probs[:, c], 0.01, 0.99)
    # Logit transform
    l = logit(p)
    # Sharpen by dividing log-odds by tau=0.25 (stretches distribution outward)
    p_sharp = expit(l / 0.25)
    
    # For unequivocal rule matches (pos > 0 and neg == 0 with impression pos), clamp to 0.98
    confident_pos = (unlabeled_pos[:, c] >= 6.0) & (unlabeled_neg[:, c] == 0)
    p_sharp[confident_pos] = 0.98
    
    # For unequivocal negatives (neg > 0 and pos == 0), clamp to 0.02
    confident_neg = (unlabeled_neg[:, c] >= 6.0) & (unlabeled_pos[:, c] == 0)
    p_sharp[confident_neg] = 0.02
    
    sharpened_probs[:, c] = p_sharp

# Build DataFrame
out_df = pd.DataFrame({'StudyInstanceUID': unlabeled_df['StudyInstanceUID']})
for c, target in enumerate(target_cols):
    out_df[target] = sharpened_probs[:, c]

out_path = r"D:\MLprojs\KaggleCompetition\RSNA\report_extracted_labels_4349_sharpened.csv"
out_df.to_csv(out_path, index=False)
print(f"\n[SAVED] Sharpened labels saved to {out_path}!")

print("\n--- Sharpened Pathology Distribution Summary (4,349 studies) ---")
for c, target in enumerate(target_cols):
    vals = out_df[target].values
    p_low = np.mean(vals < 0.10) * 100
    p_high = np.mean(vals > 0.90) * 100
    print(f"{target:18s}: Mean = {vals.mean():.3f} | Min={vals.min():.3f}, Max={vals.max():.3f}, Std={vals.std():.3f} | <0.10: {p_low:4.1f}%, >0.90: {p_high:4.1f}%")
