import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.model_selection import KFold
from scipy.special import expit, logit

df = pd.read_csv('train.csv')
target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

gold_df = df[df['ACL'].notnull()].copy().reset_index(drop=True)

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

# We use all 4,407 reports to fit the TF-IDF vocabulary so that the vocabulary has full coverage of all medical terms across Spanish, French, German, and English!
all_reports = df['Report'].apply(clean_text).tolist()

vec_word = TfidfVectorizer(ngram_range=(1, 3), max_features=4000, sublinear_tf=True, min_df=2)
vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', max_features=6000, sublinear_tf=True, min_df=2)

print("Fitting TF-IDF vocabulary across all 4,407 reports to capture full multilingual terminology...")
vec_word.fit(all_reports)
vec_char.fit(all_reports)
print("TF-IDF fitted with full clinical vocabulary!")

# Transform gold reports
X_gold_w = vec_word.transform(gold_df['clean_report']).toarray()
X_gold_c = vec_char.transform(gold_df['clean_report']).toarray()
X_gold = np.hstack([X_gold_w, X_gold_c])

N = len(gold_df)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 90)
print("EVALUATING MULTILINGUAL VOCABULARY ENHANCED NLP TEACHER")
print("=" * 90)

oof_preds = np.zeros((N, 12))
for c, target in enumerate(target_cols):
    y = gold_df[target].values
    best_auc = 0.0
    best_pred = None
    
    for alpha_val in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        oof_col = np.zeros(N)
        for train_idx, val_idx in kf.split(gold_df):
            X_tr, y_tr = X_gold[train_idx], y[train_idx]
            X_va = X_gold[val_idx]
            
            clf = Ridge(alpha=alpha_val)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_va)
            oof_col[val_idx] = preds
            
        auc = roc_auc_score(y, oof_col)
        if auc > best_auc:
            best_auc = auc
            best_pred = oof_col
            
    oof_preds[:, c] = best_pred
    print(f"{target:18s}: OOF AUC = {best_auc:.4f}")

mean_macro_auc = np.mean([roc_auc_score(gold_df[target_cols[c]].values, oof_preds[:, c]) for c in range(12)])
print("\n" + "=" * 90)
print(f"ENHANCED MULTILINGUAL REPORT-TO-GOLD MACRO-AUC: {mean_macro_auc:.4f}")
print("=" * 90)
