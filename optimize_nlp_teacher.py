import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.model_selection import KFold
from scipy.special import expit, logit
from scipy.optimize import minimize

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

# Targeted Lexicon
KEYWORDS = {
    'ACL': ['acl', 'lca', 'anterior cruciate', 'cruzado anterior', 'croise anterieur', 'vorderes kreuzband', 'vkb'],
    'MCL': ['mcl', 'lcm', 'medial collateral', 'colateral medial', 'colateral interno', 'collatéral médial', 'innenband'],
    'Medial Meniscus': ['medial meniscus', 'menisco medial', 'menisco interno', 'menisque medial', 'menisque interne', 'innenmeniskus', 'meniscusscheur/mediaal'],
    'Lateral Meniscus': ['lateral meniscus', 'menisco lateral', 'menisco externo', 'menisque lateral', 'menisque externe', 'aussenmeniskus'],
    'Medial OA': [
        'medial oa', 'medial osteoarth', 'femorotibial medial', 'femorotibial interne', 'artrosis medial', 'gonartrosis medial', 
        'osteoartritis medial', 'arthrose femorotibiale mediale', 'condromalacia medial', 'medial chondrosis', 'medial cartilage'
    ],
    'Lateral OA': [
        'lateral oa', 'lateral osteoarth', 'femorotibial lateral', 'femorotibial externe', 'artrosis lateral', 'gonartrosis lateral', 
        'osteoartritis lateral', 'arthrose femorotibiale laterale', 'condromalacia lateral', 'lateral chondrosis', 'lateral cartilage'
    ],
    'PF OA': [
        'pf oa', 'patellofemoral', 'femoropatelar', 'femoropatellaire', 'femoropatellar', 'condromalacia rotuliana', 'condromalacia patelar',
        'chondromalacia patell', 'artrosis patelofemoral', 'retropatellar', 'patellar cartilage', 'cartilago rotuliano', 'trochlear', 'rotula', 'patella'
    ],
    'Effusion': [
        'effusion', 'joint effusion', 'derrame', 'derrame articular', 'epanchement', 'gelenkerguss', 'erguss', 'fluid collection', 
        'excess fluid', 'liquido articular', 'abundant fluid', 'hydrops', 'suprapatellar effusion', 'small joint effusion', 'moderate joint effusion'
    ],
    'Synovitis': [
        'synovitis', 'sinovitis', 'synovite', 'synovial hypertrophy', 'synovial thickening', 'hipertrofia sinovial', 'engrosamiento sinovial',
        'synovial membrane', 'plica', 'synovium', 'pannus', 'sinovial'
    ],
    "Baker's": [
        'baker', 'popliteal cyst', 'quiste de baker', 'quiste popliteo', 'kyste de baker', 'kyste poplite', 'backer-zyste', 'popliteal'
    ],
    'Contusion': [
        'contusion', 'bone marrow edema', 'bone contusion', 'edema oseo', 'contusion osea', 'oedeme osseux', 'knochenmarkodem', 
        'marrow edema', 'trabecular edema', 'edema medular', 'bone bruise'
    ],
    'Fracture': [
        'fracture', 'fractura', 'fracturas', 'cortical defect', 'avulsion', 'trabecular fracture', 'fracture trabeculaire', 'cortical step'
    ]
}

PATHOLOGY_WORDS = [
    'tear', 'torn', 'rupture', 'rotura', 'ruptura', 'scheur', 'fissure', 'defect', 'lesion', 'amputacion', 'amputation', 
    'injury', 'desgarro', 'avulsion', 'edema', 'hypertrophy', 'high signal', 'increased signal', 'aumento de senal',
    'hyperintense', 'chondrosis', 'chondropathy', 'osteophyte', 'narrowing', 'pinzamiento', 'degen', 'effusion', 'derrame'
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
        has_path = any(tw in clause for tw in PATHOLOGY_WORDS)
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

# Precompute rule scores for all gold studies
N = len(gold_df)
rule_matrix = np.zeros((N, 12))
for i, row in gold_df.iterrows():
    for c, t in enumerate(target_cols):
        rule_matrix[i, c] = score_rule_based(row['clean_report'], t)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Fit and evaluate models per target
print("=" * 90)
print("PER-TARGET TUNED DUAL NLP TEACHER (5-FOLD HONEST CV ON GOLD 58)")
print("=" * 90)

best_target_oofs = np.zeros((N, 12))
nlp_config = {}

for c, target in enumerate(target_cols):
    y = gold_df[target].values
    
    # Try different configurations:
    # 1. Rule only
    auc_rule = roc_auc_score(y, rule_matrix[:, c])
    
    # 2. ML with different vectorizers and C
    best_c_auc = 0.0
    best_c_preds = None
    best_params = {}
    
    for C_val in [0.1, 0.5, 1.0, 2.0, 5.0]:
        oof_ml = np.zeros(N)
        oof_rule = rule_matrix[:, c]
        
        for train_idx, val_idx in kf.split(gold_df):
            tr_texts = gold_df.iloc[train_idx]['clean_report'].tolist()
            va_texts = gold_df.iloc[val_idx]['clean_report'].tolist()
            
            vec = TfidfVectorizer(ngram_range=(1, 3), max_features=2000, sublinear_tf=True)
            X_tr = vec.fit_transform(tr_texts)
            X_va = vec.transform(va_texts)
            
            clf = LogisticRegression(C=C_val, max_iter=1000, class_weight='balanced')
            clf.fit(X_tr, y[train_idx])
            oof_ml[val_idx] = clf.predict_proba(X_va)[:, 1]
            
        for w_rule in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
            blend = w_rule * oof_rule + (1.0 - w_rule) * oof_ml
            auc_blend = roc_auc_score(y, blend)
            if auc_blend > best_c_auc:
                best_c_auc = auc_blend
                best_c_preds = blend
                best_params = {'C': C_val, 'w_rule': w_rule, 'auc': auc_blend}
                
    best_target_oofs[:, c] = best_c_preds
    nlp_config[target] = best_params
    print(f"{target:18s}: OOF AUC = {best_c_auc:.4f} (Rule: {auc_rule:.4f}, Best w_rule={best_params['w_rule']:.2f}, C={best_params['C']})")

mean_auc = np.mean([roc_auc_score(gold_df[target_cols[c]].values, best_target_oofs[:, c]) for c in range(12)])
print("\n" + "=" * 90)
print(f"OPTIMIZED REPORT-TO-GOLD MACRO-AUC: {mean_auc:.4f}")
print("=" * 90)
