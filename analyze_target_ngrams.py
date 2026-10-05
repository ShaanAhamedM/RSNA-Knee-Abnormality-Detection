import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tune_precision_lexicon import clean_text

df = pd.read_csv('train.csv')
gold_df = df[df['ACL'].notnull()].copy().reset_index(drop=True)
gold_df['clean_report'] = gold_df['Report'].apply(clean_text)

vec = TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True)
X = vec.fit_transform(gold_df['clean_report']).toarray()
vocab = np.array(vec.get_feature_names_out())

for target in ['Effusion', 'Contusion', 'Fracture', 'Synovitis']:
    y = gold_df[target].values
    corrs = np.array([np.corrcoef(X[:, j], y)[0, 1] if np.std(X[:, j]) > 1e-6 else 0 for j in range(X.shape[1])])
    
    top_pos_idx = np.argsort(corrs)[-8:][::-1]
    top_neg_idx = np.argsort(corrs)[:8]
    
    print(f"\n{'='*70}\nTOP N-GRAMS FOR {target} (Positives: {int(y.sum())}/58)\n{'='*70}")
    print("Top Positive Correlated N-grams:")
    for idx in top_pos_idx:
        print(f"  + {vocab[idx]:30s}: r = {corrs[idx]:.4f}")
    print("Top Negative Correlated N-grams:")
    for idx in top_neg_idx:
        print(f"  - {vocab[idx]:30s}: r = {corrs[idx]:.4f}")
