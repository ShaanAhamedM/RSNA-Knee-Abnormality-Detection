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

# Expanded keyword dictionary
KEYWORDS = {
    'ACL': ['acl', 'lca', 'anterior cruciate', 'cruzado anterior', 'croise anterieur', 'vorderes kreuzband', 'vkb'],
    'MCL': ['mcl', 'lcm', 'medial collateral', 'colateral medial', 'colateral interno', 'collatéral médial', 'innenband'],
    'Medial Meniscus': ['medial meniscus', 'menisco medial', 'menisco interno', 'menisque medial', 'menisque interne', 'innenmeniskus', 'meniscusscheur/mediaal'],
    'Lateral Meniscus': ['lateral meniscus', 'menisco lateral', 'menisco externo', 'menisque lateral', 'menisque externe', 'aussenmeniskus'],
    'Medial OA': [
        'medial oa', 'medial osteoarth', 'femorotibial medial', 'femorotibial interne', 'artrosis medial', 'gonartrosis medial', 
        'osteoartritis medial', 'arthrose femorotibiale mediale', 'condromalacia medial', 'medial chondrosis', 'medial cartilage',
        'platillo tibial medial', 'condilo femoral medial', 'medial compartment'
    ],
    'Lateral OA': [
        'lateral oa', 'lateral osteoarth', 'femorotibial lateral', 'femorotibial externe', 'artrosis lateral', 'gonartrosis lateral', 
        'osteoartritis lateral', 'arthrose femorotibiale laterale', 'condromalacia lateral', 'lateral chondrosis', 'lateral cartilage',
        'platillo tibial lateral', 'condilo femoral lateral', 'lateral compartment'
    ],
    'PF OA': [
        'pf oa', 'patellofemoral', 'femoropatelar', 'femoropatellaire', 'femoropatellar', 'condromalacia rotuliana', 'condromalacia patelar',
        'chondromalacia patell', 'artrosis patelofemoral', 'retropatellar', 'patellar cartilage', 'cartilago rotuliano', 'trochlear', 'rotula', 'patella'
    ],
    'Effusion': [
        'effusion', 'joint effusion', 'derrame', 'derrame articular', 'epanchement', 'gelenkerguss', 'erguss', 'fluid collection', 
        'excess fluid', 'liquido articular', 'abundant fluid', 'hydrops'
    ],
    'Synovitis': [
        'synovitis', 'sinovitis', 'synovite', 'synovial hypertrophy', 'synovial thickening', 'hipertrofia sinovial', 'engrosamiento sinovial',
        'synovial membrane', 'plica', 'synovium', 'pannus', 'sinovial'
    ],
    "Baker's": [
        'baker', 'popliteal cyst', 'quiste de baker', 'quiste popliteo', 'kyste de baker', 'kyste poplite', 'backer-zyste', 'poplitea'
    ],
    'Contusion': [
        'contusion', 'bone marrow edema', 'bone contusion', 'edema oseo', 'contusion osea', 'oedeme osseux', 'knochenmarkodem', 
        'marrow edema', 'trabecular edema', 'edema medular'
    ],
    'Fracture': [
        'fracture', 'fractura', 'fracturas', 'cortical defect', 'avulsion', 'trabecular fracture', 'fracture trabeculaire', 'cortical disruption'
    ]
}

TEAR_WORDS = [
    'tear', 'torn', 'rupture', 'rotura', 'ruptura', 'scheur', 'fissure', 'defect', 'lesion', 'amputacion', 'amputation', 
    'injury', 'desgarro', 'avulsion', 'edema', 'hypertrophy', 'high signal', 'increased signal', 'aumento de senal',
    'hyperintense', 'chondrosis', 'chondropathy', 'osteophyte', 'narrowing', 'pinzamiento', 'degen', 'irregularity', 'fraying',
    'grade', 'sever', 'moderat', 'leveg'
]

NEGATION_WORDS = [
    'no ', 'not ', 'without ', 'intact', 'normal', 'unremarkable', 'conservado', 'conservada', 'aucun', 'aucune', 'pas de',
    'kein', 'keine', 'geen', 'sem ', 'sin ', 'ausencia de', 'free of', 'negative for', 'preservado', 'preservada', 'preserved'
]

def score_rule_based(text, target):
    clauses = re.split(r'[\.\;\n\*\>\•]', text)
    score = 0.0
    keywords = KEYWORDS[target]
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        has_kw = any(kw in clause for kw in keywords)
        if not has_kw:
            continue
        has_path = any(tw in clause for tw in TEAR_WORDS)
        has_neg = any(nw in clause for nw in NEGATION_WORDS)
        if has_path and not has_neg:
            score += 1.5
        elif has_neg and not has_path:
            score -= 1.5
        elif has_neg and has_path:
            if re.search(r'(no|sin|not|without|sem|kein|aucune)\s+(evidence of\s+)?(tear|rotura|signos de rotura|rupture|lesion|fracture|chondrosis|effusion)', clause):
                score -= 1.5
            else:
                score += 0.5
    return expit(score)

# Engine B: Cross-Validated TF-IDF + Ridge/Logistic Regression
print("\n" + "=" * 80)
print("TRAINING & 5-FOLD CROSS-VALIDATING DUAL-ENGINE NLP EXTRACTOR")
print("=" * 80)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
N = len(gold_df)
oof_nlp_preds = np.zeros((N, 12))

for train_idx, val_idx in kf.split(gold_df):
    train_texts = gold_df.iloc[train_idx]['clean_report'].tolist()
    val_texts = gold_df.iloc[val_idx]['clean_report'].tolist()
    
    # Dual n-gram vectorizer: word n-grams (1-3) and char n-grams (3-5)
    vec_word = TfidfVectorizer(ngram_range=(1, 3), max_features=1500, min_df=1, sublinear_tf=True)
    vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', max_features=2500, min_df=1, sublinear_tf=True)
    
    X_tr_w = vec_word.fit_transform(train_texts).toarray()
    X_va_w = vec_word.transform(val_texts).toarray()
    
    X_tr_c = vec_char.fit_transform(train_texts).toarray()
    X_va_c = vec_char.transform(val_texts).toarray()
    
    # Engine A rule scores as explicit dense features
    tr_rules = np.array([[score_rule_based(t, col) for col in target_cols] for t in train_texts])
    va_rules = np.array([[score_rule_based(t, col) for col in target_cols] for t in val_texts])
    
    X_tr = np.hstack([X_tr_w, X_tr_c, tr_rules])
    X_va = np.hstack([X_va_w, X_va_c, va_rules])
    
    for c, target in enumerate(target_cols):
        y_tr = gold_df.iloc[train_idx][target].values
        
        # Fit L2 Regularized Logistic Regression or Ridge
        model = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
        model.fit(X_tr, y_tr)
        probs_ml = model.predict_proba(X_va)[:, 1]
        
        # Blend Engine A (rule) + Engine B (ML)
        rule_val = va_rules[:, c]
        blended = 0.40 * rule_val + 0.60 * probs_ml
        oof_nlp_preds[val_idx, c] = blended

print("\n--- 5-FOLD OUT-OF-FOLD NLP EXTRACTOR MACRO-AUC ---")
nlp_aucs = []
for c, target in enumerate(target_cols):
    y = gold_df[target].values
    auc = roc_auc_score(y, oof_nlp_preds[:, c])
    nlp_aucs.append(auc)
    print(f"{target:18s}: OOF AUC = {auc:.4f}")

mean_nlp_auc = np.mean(nlp_aucs)
print("=" * 80)
print(f"REPORT-TO-GOLD MACRO-AUC: {mean_nlp_auc:.4f}")
print("=" * 80)
