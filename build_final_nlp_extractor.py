import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from scipy.special import expit, logit

df = pd.read_csv('train.csv')
target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

gold_mask = df['ACL'].notnull()
gold_df = df[gold_mask].copy().reset_index(drop=True)
unlabeled_df = df[~gold_mask].copy().reset_index(drop=True)

print(f"Total reports: {len(df)} | Gold labeled: {len(gold_df)} | Unlabeled to extract: {len(unlabeled_df)}")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
    text = text.replace('à', 'a').replace('è', 'e').replace('ù', 'u').replace('â', 'a').replace('ê', 'e').replace('î', 'i').replace('ô', 'o').replace('û', 'u')
    text = text.replace('ä', 'a').replace('ö', 'o').replace('ü', 'u')
    text = re.sub(r'[^a-z0-9\s\.\,\;\:\-\/]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

gold_df['clean_report'] = gold_df['Report'].apply(clean_text)
unlabeled_df['clean_report'] = unlabeled_df['Report'].apply(clean_text)
df['clean_report'] = df['Report'].apply(clean_text)

# Import precision rules from tune_precision_lexicon
from tune_precision_lexicon import extract_rule_score

N_gold = len(gold_df)
gold_rule_scores = np.zeros((N_gold, 12))
for i in range(N_gold):
    txt = gold_df.iloc[i]['clean_report']
    for c, t in enumerate(target_cols):
        gold_rule_scores[i, c] = extract_rule_score(txt, t)

print("\n" + "=" * 90)
print("TRAINING DUAL NLP EXTRACTOR & EVALUATING 5-FOLD CV ON GOLD 58")
print("=" * 90)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Word and char TF-IDF
vec_word = TfidfVectorizer(ngram_range=(1, 3), max_features=3000, sublinear_tf=True)
vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', max_features=5000, sublinear_tf=True)

X_word_gold = vec_word.fit_transform(gold_df['clean_report']).toarray()
X_char_gold = vec_char.fit_transform(gold_df['clean_report']).toarray()

oof_final_preds = np.zeros((N_gold, 12))
per_target_nlp_models = []

for c, target in enumerate(target_cols):
    y = gold_df[target].values
    best_target_auc = 0.0
    best_w_rule = 0.5
    best_c_param = 1.0
    best_oof = None
    
    for C_param in [0.2, 0.5, 1.0, 2.0, 5.0]:
        oof_ml = np.zeros(N_gold)
        for train_idx, val_idx in kf.split(gold_df):
            X_tr = np.hstack([X_word_gold[train_idx], X_char_gold[train_idx], gold_rule_scores[train_idx]])
            X_va = np.hstack([X_word_gold[val_idx], X_char_gold[val_idx], gold_rule_scores[val_idx]])
            
            clf = LogisticRegression(C=C_param, max_iter=1000, class_weight='balanced')
            clf.fit(X_tr, y[train_idx])
            oof_ml[val_idx] = clf.predict_proba(X_va)[:, 1]
            
        for w_r in [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]:
            blend = w_r * gold_rule_scores[:, c] + (1.0 - w_r) * oof_ml
            auc = roc_auc_score(y, blend)
            if auc > best_target_auc:
                best_target_auc = auc
                best_w_rule = w_r
                best_c_param = C_param
                best_oof = blend
                
    oof_final_preds[:, c] = best_oof
    print(f"Target {c:2d} ({target:18s}): 5-Fold OOF AUC = {best_target_auc:.4f} (Rule weight: {best_w_rule:.2f}, ML C={best_c_param})")

final_macro_auc = np.mean([roc_auc_score(gold_df[target_cols[c]].values, oof_final_preds[:, c]) for c in range(12)])
print("\n" + "=" * 90)
print(f">>> FINAL REPORT-TO-GOLD 5-FOLD MACRO-AUC: {final_macro_auc:.4f} <<<")
print("=" * 90)

# Now train full production NLP extractor on ALL 58 gold reports and infer on all 4,349 unlabeled reports!
print("\nFitting full production dual-engine NLP extractor on all 58 gold studies...")
X_full_gold = np.hstack([X_word_gold, X_char_gold, gold_rule_scores])

# Transform 4,349 unlabeled reports
print(f"Transforming {len(unlabeled_df)} unlabeled clinical reports...")
X_word_unlabeled = vec_word.transform(unlabeled_df['clean_report']).toarray()
X_char_unlabeled = vec_char.transform(unlabeled_df['clean_report']).toarray()

unlabeled_rule_scores = np.zeros((len(unlabeled_df), 12))
for i in range(len(unlabeled_df)):
    txt = unlabeled_df.iloc[i]['clean_report']
    for c, t in enumerate(target_cols):
        unlabeled_rule_scores[i, c] = extract_rule_score(txt, t)

X_full_unlabeled = np.hstack([X_word_unlabeled, X_char_unlabeled, unlabeled_rule_scores])

extracted_probs = np.zeros((len(unlabeled_df), 12), dtype=np.float32)

for c, target in enumerate(target_cols):
    y = gold_df[target].values
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    clf.fit(X_full_gold, y)
    probs_ml = clf.predict_proba(X_full_unlabeled)[:, 1]
    
    # Target specific blending with rule engine
    rule_w = 0.50
    if target in ['MCL', "Baker's"]:
        rule_w = 0.70
    elif target in ['Medial Meniscus', 'Medial OA']:
        rule_w = 0.35
        
    blended_probs = rule_w * unlabeled_rule_scores[:, c] + (1.0 - rule_w) * probs_ml
    # Bound to safe range
    extracted_probs[:, c] = np.clip(blended_probs, 0.005, 0.995)

# Build DataFrame with extracted labels
extracted_df = pd.DataFrame({'StudyInstanceUID': unlabeled_df['StudyInstanceUID']})
for c, target in enumerate(target_cols):
    extracted_df[target] = extracted_probs[:, c]

out_csv = r"D:\MLprojs\KaggleCompetition\RSNA\report_extracted_labels_4349.csv"
extracted_df.to_csv(out_csv, index=False)
print(f"\n[SUCCESS] Extracted high-precision labels for all {len(extracted_df)} studies saved to {out_csv}!")

print("\n--- Extracted Pathology Prevalence Summary (4,349 studies) vs. Gold 58 ---")
for c, target in enumerate(target_cols):
    mean_ext = extracted_df[target].mean()
    gold_pi = gold_df[target].mean()
    print(f"{target:18s}: Extracted Mean = {mean_ext*100:5.1f}% (Gold True pi = {gold_pi*100:5.1f}%) | Std = {extracted_df[target].std():.3f}")
