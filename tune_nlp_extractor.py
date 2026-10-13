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

# Multilingual Comprehensive Medical Lexicon for Engine A
KEYWORDS = {
    'ACL': [
        'acl', 'lca', 'anterior cruciate', 'cruzado anterior', 'croise anterieur', 'vorderes kreuzband', 'vkb',
        'cruzados', 'cruciate'
    ],
    'MCL': [
        'mcl', 'lcm', 'medial collateral', 'colateral medial', 'colateral interno', 'collatéral médial', 'innenband',
        'colaterales', 'collateral'
    ],
    'Medial Meniscus': [
        'medial meniscus', 'menisco medial', 'menisco interno', 'menisque medial', 'menisque interne', 'innenmeniskus', 
        'meniscusscheur/mediaal', 'cuerno posterior del menisco medial', 'cuerno anterior del menisco medial',
        'cuerpo del menisco medial', 'menisco interno', 'compartimento medial', 'medial compartment'
    ],
    'Lateral Meniscus': [
        'lateral meniscus', 'menisco lateral', 'menisco externo', 'menisque lateral', 'menisque externe', 'aussenmeniskus',
        'cuerno posterior del menisco lateral', 'cuerno anterior del menisco lateral', 'cuerpo del menisco lateral',
        'compartimento lateral', 'lateral compartment'
    ],
    'Medial OA': [
        'medial oa', 'medial osteoarth', 'femorotibial medial', 'femorotibial interne', 'artrosis medial', 'gonartrosis medial', 
        'osteoartritis medial', 'arthrose femorotibiale mediale', 'condromalacia medial', 'medial chondrosis', 'medial cartilage',
        'platillo tibial medial', 'condilo femoral medial', 'medial joint space', 'pinzamiento medial', 'medial compartment cartilage'
    ],
    'Lateral OA': [
        'lateral oa', 'lateral osteoarth', 'femorotibial lateral', 'femorotibial externe', 'artrosis lateral', 'gonartrosis lateral', 
        'osteoartritis lateral', 'arthrose femorotibiale laterale', 'condromalacia lateral', 'lateral chondrosis', 'lateral cartilage',
        'platillo tibial lateral', 'condilo femoral lateral', 'lateral joint space', 'pinzamiento lateral', 'lateral compartment cartilage'
    ],
    'PF OA': [
        'pf oa', 'patellofemoral', 'femoropatelar', 'femoropatellaire', 'femoropatellar', 'condromalacia rotuliana', 'condromalacia patelar',
        'chondromalacia patell', 'artrosis patelofemoral', 'retropatellar', 'patellar cartilage', 'cartilago rotuliano', 'trochlear', 
        'rotula', 'patella', 'troclea', 'patellofemoral joint', 'chondromalacia', 'condromalacia'
    ],
    'Effusion': [
        'effusion', 'joint effusion', 'derrame', 'derrame articular', 'epanchement', 'gelenkerguss', 'erguss', 'fluid collection', 
        'excess fluid', 'liquido articular', 'abundant fluid', 'hydrops', 'suprapatellar', 'suprapatelar', 'intraarticular fluid',
        'fluid', 'liquido', 'intra-articular fluid'
    ],
    'Synovitis': [
        'synovitis', 'sinovitis', 'synovite', 'synovial hypertrophy', 'synovial thickening', 'hipertrofia sinovial', 'engrosamiento sinovial',
        'synovial membrane', 'plica', 'synovium', 'pannus', 'sinovial', 'capsular thickening', 'synovial'
    ],
    "Baker's": [
        'baker', 'popliteal cyst', 'quiste de baker', 'quiste popliteo', 'kyste de baker', 'kyste poplite', 'backer-zyste', 
        'poplitea', 'popliteal', 'gastrocnemius-semimembranosus', 'gastrocnemio-semimembranoso', 'semimembranosus bursa', 'popliteo'
    ],
    'Contusion': [
        'contusion', 'bone marrow edema', 'bone contusion', 'edema oseo', 'contusion osea', 'oedeme osseux', 'knochenmarkodem', 
        'marrow edema', 'trabecular edema', 'edema medular', 'trabecular microfracture', 'bone bruise', 'subchondral edema', 'edema subcondral'
    ],
    'Fracture': [
        'fracture', 'fractura', 'fracturas', 'cortical defect', 'avulsion', 'trabecular fracture', 'fracture trabeculaire', 
        'cortical disruption', 'fissure osseuse', 'linea de fractura', 'cortical step'
    ]
}

PATHOLOGY_WORDS = [
    'tear', 'torn', 'rupture', 'rotura', 'ruptura', 'scheur', 'fissure', 'defect', 'lesion', 'amputacion', 'amputation', 
    'injury', 'desgarro', 'avulsion', 'edema', 'hypertrophy', 'high signal', 'increased signal', 'aumento de senal',
    'hyperintense', 'chondrosis', 'chondropathy', 'osteophyte', 'narrowing', 'pinzamiento', 'degen', 'irregularity', 'fraying',
    'grade', 'sever', 'moderat', 'leveg', 'cyst', 'quiste', 'kyste', 'thickening', 'effusion', 'derrame', 'edema oseo', 'marrow edema'
]

NEGATION_PATTERNS = [
    r'no\s+(?:evidence\s+of\s+|signos\s+de\s+|hay\s+)?(?:tear|rotura|rupture|fracture|effusion|derrame|edema|alteracion|lesion)',
    r'sin\s+(?:signos\s+de\s+|evidencia\s+de\s+)?(?:rotura|lesion|alteracion|derrame|edema|fractura)',
    r'without\s+(?:evidence\s+of\s+)?(?:tear|injury|fracture|effusion|edema)',
    r'intact', r'normal', r'unremarkable', r'conservad[oa]', r'preservad[oa]', r'aucun[e]?', r'pas\s+de',
    r'kein[e]?', r'geen', r'sem\s+rotura', r'free\s+of', r'negative\s+for', r'ausencia\s+de'
]

def score_rule_based_v2(text, target):
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
        has_neg = any(re.search(pat, clause) for pat in NEGATION_PATTERNS)
        has_path = any(pw in clause for pw in PATHOLOGY_WORDS)
        
        if has_path and not has_neg:
            score += 2.0
        elif has_neg and not has_path:
            score -= 2.0
        elif has_neg and has_path:
            score -= 1.0
            
    return expit(score)

print("\n" + "=" * 90)
print("TRAINING MULTI-REPRESENTATION NLP EXTRACTOR (TF-IDF + CHAR N-GRAMS + RULES)")
print("=" * 90)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
N = len(gold_df)
oof_preds = np.zeros((N, 12))

# Multi-representation vectorizer
for train_idx, val_idx in kf.split(gold_df):
    tr_df = gold_df.iloc[train_idx]
    va_df = gold_df.iloc[val_idx]
    
    vec_word = TfidfVectorizer(ngram_range=(1, 3), max_features=3000, sublinear_tf=True)
    vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', max_features=5000, sublinear_tf=True)
    
    tr_w = vec_word.fit_transform(tr_df['clean_report']).toarray()
    va_w = vec_word.transform(va_df['clean_report']).toarray()
    
    tr_c = vec_char.fit_transform(tr_df['clean_report']).toarray()
    va_c = vec_char.transform(va_df['clean_report']).toarray()
    
    # Rule features for all 12 targets
    tr_r = np.array([[score_rule_based_v2(t, col) for col in target_cols] for t in tr_df['clean_report']])
    va_r = np.array([[score_rule_based_v2(t, col) for col in target_cols] for t in va_df['clean_report']])
    
    X_tr = np.hstack([tr_w, tr_c, tr_r])
    X_va = np.hstack([va_w, va_c, va_r])
    
    for c, target in enumerate(target_cols):
        y_tr = tr_df[target].values
        
        # Test Ridge vs LogisticRegression
        # Ridge with probability sigmoid mapping is extremely effective on small samples
        clf = Ridge(alpha=2.0)
        clf.fit(X_tr, y_tr)
        preds_raw = clf.predict(X_va)
        
        # Blend with rule score
        rule_score = va_r[:, c]
        blended = 0.50 * rule_score + 0.50 * expit(preds_raw)
        oof_preds[val_idx, c] = blended

print("\n--- 5-FOLD OUT-OF-FOLD REPORT EXTRACTOR MACRO-AUC ---")
aucs = []
for c, target in enumerate(target_cols):
    y = gold_df[target].values
    auc = roc_auc_score(y, oof_preds[:, c])
    aucs.append(auc)
    print(f"{target:18s}: OOF AUC = {auc:.4f}")

mean_auc = np.mean(aucs)
print("=" * 90)
print(f"REPORT-TO-GOLD 5-FOLD MACRO-AUC: {mean_auc:.4f}")
print("=" * 90)
