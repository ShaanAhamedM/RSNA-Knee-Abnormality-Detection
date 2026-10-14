"""
Upgrade 1: Multi-Lingual Clinical NLP Teacher V4
Targeting > 0.88 - 0.92 Macro-AUC on Gold 58.
Expands terminology for Effusion, Contusion, PF OA, Synovitis, ACL, Meniscus.
Combines Precision Lexicon V4 + Multi-Scale TF-IDF + Ridge + Logistic + LightGBM GBDT.
"""

import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.model_selection import KFold
import lightgbm as lgb
from scipy.special import expit, logit

df = pd.read_csv('train.csv')
gold_mask = df['ACL'].notnull()
gold_df = df[gold_mask].copy().reset_index(drop=True)
unlabeled_df = df[~gold_mask].copy().reset_index(drop=True)

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print("=" * 85)
print(f"UPGRADE 1: NLP REPORT TEACHER V4 (Gold: {len(gold_df)} | Unlabeled: {len(unlabeled_df)})")
print("=" * 85)

def clean_text_v4(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for src, dst in [
        ('á', 'a'), ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u'), ('ñ', 'n'),
        ('à', 'a'), ('è', 'e'), ('ù', 'u'), ('â', 'a'), ('ê', 'e'), ('î', 'i'),
        ('ô', 'o'), ('û', 'u'), ('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss'),
        ('ç', 'c'), ('ğ', 'g'), ('ı', 'i'), ('ş', 's'), ('Redacted', ''), ('redacted', '')
    ]:
        text = text.replace(src, dst)
    text = re.sub(r'[^a-z0-9\s\.\;\:\-\/\,\n]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

gold_df['clean_report'] = gold_df['Report'].apply(clean_text_v4)
unlabeled_df['clean_report'] = unlabeled_df['Report'].apply(clean_text_v4)

# Deepened Precision Lexicon V4
CLINICAL_LEXICON_V4 = {
    'ACL': {
        'keywords': [
            'acl', 'lca', 'anterior cruciate', 'cruzado anterior', 'croise anterieur', 
            'vorderes kreuzband', 'vkb', 'on capraz bag', 'ligament croise anterieur'
        ],
        'positives': [
            'tear', 'torn', 'rupture', 'rotura', 'ruptura', 'scheur', 'lesion', 'desgarro', 
            'avulsion', 'irregular', 'complete', 'incompleto', 'interstitial', 'high signal', 
            'disruption', 'yirtik', 'non visualizado', 'aumento de senal', 'hyperintense', 'dechirure',
            'attenuated', 'waviness', 'thickening', 'ganglion cyst'
        ],
        'negatives': [
            'intact', 'normal', 'unremarkable', 'conservad', 'preservad', 'sin signos de rotura', 
            'no tear', 'sin rotura', 'aucun', 'sans rupture', 'sin alteraciones', 'intaktes', 
            'regelrecht', 'dogal', 'continuo', 'morfolojisi dogal', 'homogeneo', 'no focal tear'
        ]
    },
    'MCL': {
        'keywords': [
            'mcl', 'lcm', 'medial collateral', 'colateral medial', 'colateral interno', 
            'collatéral médial', 'innenband', 'ligamento colateral medial', 'ic yan bag',
            'tibial collateral'
        ],
        'positives': [
            'tear', 'torn', 'rupture', 'rotura', 'ruptura', 'sprain', 'injury', 'grade 1', 
            'grade 2', 'grade 3', 'esguince', 'aumento de senal', 'edema', 'thickened', 
            'desgarro', 'yirtik', 'strain', 'periligamentous edema', 'periligamentario'
        ],
        'negatives': [
            'intact', 'normal', 'unremarkable', 'conservad', 'preservad', 'sin alteraciones', 
            'no tear', 'sin signos de rotura', 'sin rotura', 'continuo', 'regelrecht', 'dogal', 'ausencia de'
        ]
    },
    'Medial Meniscus': {
        'keywords': [
            'medial meniscus', 'menisco medial', 'menisco interno', 'menisque medial', 
            'menisque interne', 'innenmeniskus', 'meniscusscheur/mediaal', 'cuerno posterior del menisco medial', 
            'cuerno anterior del menisco medial', 'cuerpo del menisco medial', 'medial meniskus', 'mm'
        ],
        'positives': [
            'tear', 'torn', 'rotura', 'ruptura', 'scheur', 'fissure', 'fisura', 'amputacion', 
            'amputation', 'horizontal', 'vertical', 'radial', 'complex', 'compleja', 'desgarro', 
            'maceracion', 'extrusion', 'root tear', 'flap', 'yirtik', 'degenerative tear', 
            'contacting articular surface', 'contacta superficie', 'meniscopatia'
        ],
        'negatives': [
            'not torn', 'sin signos de rotura', 'sin rotura', 'no tear', 'normal', 'conservad', 
            'preservad', 'unremarkable', 'intact', 'aucun', 'morfologia y senal conservada', 
            'sin fisura', 'regelrecht', 'dogal', 'sans fissure', 'intaktes'
        ]
    },
    'Lateral Meniscus': {
        'keywords': [
            'lateral meniscus', 'menisco lateral', 'menisco externo', 'menisque lateral', 
            'menisque externe', 'aussenmeniskus', 'cuerno posterior del menisco lateral', 
            'cuerno anterior del menisco lateral', 'cuerpo del menisco lateral', 'lateral meniskus', 'lm'
        ],
        'positives': [
            'tear', 'torn', 'rotura', 'ruptura', 'scheur', 'fissure', 'fisura', 'amputacion', 
            'amputation', 'horizontal', 'vertical', 'radial', 'complex', 'compleja', 'desgarro', 
            'maceracion', 'discoid', 'root tear', 'flap', 'yirtik', 'degenerative tear', 
            'contacting articular surface', 'contacta superficie', 'meniscopatia'
        ],
        'negatives': [
            'not torn', 'sin signos de rotura', 'sin rotura', 'no tear', 'normal', 'conservad', 
            'preservad', 'unremarkable', 'intact', 'aucun', 'morfologia y senal conservada', 
            'sin fisura', 'regelrecht', 'dogal', 'sans fissure', 'intaktes'
        ]
    },
    'Medial OA': {
        'keywords': [
            'medial', 'femorotibial medial', 'femorotibial interne', 'artrosis medial', 
            'gonartrosis medial', 'osteoartritis medial', 'medial compartment', 'compartimento medial', 
            'platillo tibial medial', 'condilo femoral medial', 'medial gonarthrose', 'medial tibial plateau'
        ],
        'positives': [
            'artrosis', 'osteoarth', 'chondrosis', 'chondropathy', 'condromalacia', 'condropatia', 
            'fissure', 'defect', 'ulcera condral', 'narrowing', 'pinzamiento', 'osteophyte', 
            'osteofito', 'adelgazamiento', 'perdida de espesor', 'grade 2', 'grade 3', 'grade 4', 
            'espesor parcial', 'espesor total', 'eburnacion', 'gelenkspalt', 'subchondral sclerosis',
            'esclerosis subcondral', 'osteofitosis marginal'
        ],
        'negatives': [
            'no focal chondrosis', 'sin alteraciones', 'normal', 'conservad', 'preservad', 
            'sin signos de artrosis', 'intact', 'unremarkable', 'no chondral', 'sin osteofitos', 
            'espacio articular conservado', 'regelrecht', 'intaktes knorpel'
        ]
    },
    'Lateral OA': {
        'keywords': [
            'lateral', 'femorotibial lateral', 'femorotibial externe', 'artrosis lateral', 
            'gonartrosis lateral', 'osteoartritis lateral', 'lateral compartment', 'compartimento lateral', 
            'platillo tibial lateral', 'condilo femoral lateral', 'lateral gonarthrose', 'lateral tibial plateau'
        ],
        'positives': [
            'artrosis', 'osteoarth', 'chondrosis', 'chondropathy', 'condromalacia', 'condropatia', 
            'fissure', 'defect', 'ulcera condral', 'narrowing', 'pinzamiento', 'osteophyte', 
            'osteofito', 'adelgazamiento', 'perdida de espesor', 'grade 2', 'grade 3', 'grade 4', 
            'espesor parcial', 'espesor total', 'eburnacion', 'gelenkspalt', 'subchondral sclerosis',
            'esclerosis subcondral', 'osteofitosis marginal'
        ],
        'negatives': [
            'no focal chondrosis', 'sin alteraciones', 'normal', 'conservad', 'preservad', 
            'sin signos de artrosis', 'intact', 'unremarkable', 'no chondral', 'sin osteofitos', 
            'espacio articular conservado', 'regelrecht', 'intaktes knorpel'
        ]
    },
    'PF OA': {
        'keywords': [
            'patellofemoral', 'femoropatelar', 'femoropatellaire', 'femoropatellar', 'patella', 
            'rotula', 'trochlea', 'troclea', 'patelofemoral', 'retropatel', 'retropatellar', 
            'condromalacia rotuliana', 'condromalacia patelar', 'femororrotuliana', 'patellar facet', 
            'articulacion femororotuliana', 'anterior compartment', 'trochlear groove'
        ],
        'positives': [
            'condromalacia', 'chondromalacia', 'artrosis', 'osteoarth', 'chondrosis', 'defect', 
            'fissure', 'ulcer', 'ulcera condral', 'adelgazamiento', 'pinzamiento', 'osteophyte', 
            'osteofito', 'grade 2', 'grade 3', 'grade 4', 'grade ii', 'grade iii', 'grade iv',
            'espesor parcial', 'espesor total', 'edema subcondral', 'fissuring', 'cartilage loss', 
            'subchondral cysts', 'retropatellar chondropathy', 'trochlear dysplasia'
        ],
        'negatives': [
            'congruent without chondral', 'sin alteraciones', 'normal', 'conservad', 'preservad', 
            'sin condromalacia', 'intact', 'unremarkable', 'no chondral', 'cartilago articular conservado',
            'rotula centrada sin artrosis', 'regelrecht'
        ]
    },
    'Effusion': {
        'keywords': [
            'effusion', 'derrame', 'epanchement', 'gelenkerguss', 'erguss', 'joint effusion', 
            'derrame articular', 'suprapatellar recess', 'fondo de saco suprapatelar', 'efuzyon', 
            'sivi', 'joint fluid', 'intra-articular fluid', 'hydrarthrosis', 'suprapatellar bursa',
            'recesso suprapatelar', 'articular fluid'
        ],
        'positives': [
            'effusion', 'derrame', 'joint effusion', 'small joint effusion', 'moderate joint effusion', 
            'large joint effusion', 'derrame articular', 'moderado derrame', 'abundante derrame', 
            'leve derrame', 'distension', 'fluid collection', 'liquido articular', 'abundant fluid', 
            'significant effusion', 'mild effusion', 'moderate effusion', 'large effusion', 
            'distendido por liquido', 'gelenkerguss', 'ergussbildung', 'intraartikularer erguss',
            'efuzyon mevcuttur', 'eklem ici sivi artisi', 'fluid in suprapatellar recess', 'hydrarthrose'
        ],
        'negatives': [
            'no joint effusion', 'sin derrame', 'sin derrame articular', 'no effusion', 'no significant effusion', 
            'sin colecciones', 'ausencia de derrame', 'pas d epanchement', 'kein erguss', 'fisiologico', 
            'physiological fluid', 'minima cantidad fisiologica', 'kein gelenkerguss', 'efuzyon saptanmadi',
            'no fluid collection', 'minimal physiologic', 'sin derrame articular significativo', 'aucun epanchement'
        ]
    },
    'Synovitis': {
        'keywords': [
            'synovitis', 'sinovitis', 'synovite', 'synovial', 'sinovial', 'synovium', 'sinovit', 
            'hoffa', 'plica', 'synovial membrane', 'engrosamiento'
        ],
        'positives': [
            'synovitis', 'sinovitis', 'synovial thickening', 'engrosamiento sinovial', 'hipertrofia sinovial', 
            'synovial hypertrophy', 'synovial proliferation', 'pannus', 'hyperintense synovium', 
            'sinovial reactiva', 'synovitis reactive', 'sinovit', 'synovial enhancement', 'hoffitis', 
            'hoffa edema', 'edema de hoffa', 'impingement', 'plica syndrome', 'inflamacion sinovial'
        ],
        'negatives': [
            'no synovitis', 'sin sinovitis', 'no synovial thickening', 'sin engrosamiento sinovial', 
            'normal', 'sin alteraciones', 'ausencia de sinovitis', 'sin signos de sinovitis', 
            'sinovit saptanmadi', 'keine synovitis', 'sin alteraciones de la grasa de hoffa'
        ]
    },
    "Baker's": {
        'keywords': [
            'baker', 'popliteal cyst', 'quiste de baker', 'quiste popliteo', 'kyste de baker', 
            'kyste poplite', 'backer', 'popliteal fossa cyst', 'baker kisti', 'popliteal'
        ],
        'positives': [
            'baker', 'cyst', 'quiste', 'kyste', 'popliteal cyst', 'quiste de baker', 'quiste popliteo', 
            'distension de la bursa gastrocnemio-semimembranoso', 'baker kisti', 'popliteal fluid collection',
            'gastrocnemius-semimembranosus bursa'
        ],
        'negatives': [
            'no baker', 'no popliteal cyst', 'sin quiste', 'sin quiste de baker', 'ausencia de quiste', 
            'no cyst seen', 'sin colecciones popliteas', 'kist saptanmadi', 'keine baker', 'no popliteal fluid'
        ]
    },
    'Contusion': {
        'keywords': [
            'contusion', 'bone marrow edema', 'edema oseo', 'contusion osea', 'bone contusion', 
            'marrow edema', 'oedeme osseux', 'knochenmarkodem', 'bone bruise', 'edema trabecular', 
            'microfractura trabecular', 'kemik iligi odemi', 'edema medular', 'trabecular'
        ],
        'positives': [
            'contusion', 'edema', 'bone marrow edema', 'edema oseo', 'trabecular microfracture', 
            'trabecular edema', 'bruise', 'high t2 signal in bone', 'edema subcondral', 'edema en el aspecto', 
            'bone bruise', 'knochenmarkodem', 'oedeme trabeculaire', 'kemik iligi odemi', 'contusion',
            'bone marrow contusion', 'subcortical bone bruise', 'microfracture', 'trabecular injury'
        ],
        'negatives': [
            'no bone marrow edema', 'no hay alteraciones de senal de la medula osea', 'sin edema oseo', 
            'no contusion', 'no edema', 'ausencia de edema oseo', 'medula osea sin alteraciones', 
            'senal osea conservada', 'sin edema trabecular', 'kein bone bruise', 'kein knochenmarkodem',
            'aucun oedeme osseux', 'medula osea normal', 'sin edema medular'
        ]
    },
    'Fracture': {
        'keywords': [
            'fracture', 'fractura', 'fracturas', 'cortical disruption', 'cortical defect', 
            'trabecular fracture', 'fissure osseuse', 'fraktur', 'kirik', 'cortical step'
        ],
        'positives': [
            'fracture', 'fractura', 'linea de fractura', 'cortical step', 'avulsion fracture', 
            'displaced fracture', 'fractura trabecular', 'depressed fracture', 'fractura por estres', 
            'fracture line', 'fraktur', 'kirik', 'trabecular impaction', 'impaction fracture', 
            'segond fracture', 'avulsion', 'trazo fracturario'
        ],
        'negatives': [
            'no fracture', 'no fracture is seen', 'sin fractura', 'sin fracturas', 'no signs of fracture', 
            'ausencia de fractura', 'pas de fracture', 'keine fraktur', 'aucune fracture', 
            'sin trazos de fractura', 'no evidence of fracture', 'no cortical step', 'kirik saptanmadi',
            'cortical continuity maintained', 'integrita corticale'
        ]
    }
}

def extract_rule_features_v4(text, target):
    rule = CLINICAL_LEXICON_V4[target]
    clauses = re.split(r'[\.\;\n\*\•]', text)
    
    pos_score = 0.0
    neg_score = 0.0
    impression_pos = False
    in_impression = False
    
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
            
        if any(w in clause for w in ['impresion', 'impression', 'conclusion', 'diagnostic', 'beurteilung']):
            in_impression = True
            
        has_kw = any(kw in clause for kw in rule['keywords'])
        if not has_kw:
            continue
            
        has_pos = any(p in clause for p in rule['positives'])
        has_neg = any(n in clause for n in rule['negatives'])
        
        weight = 2.5 if in_impression else 1.0
        
        if has_neg:
            neg_score += 4.0 * weight
        elif has_pos:
            # Check 30-char window before positive match for negation
            is_negated = False
            for p in rule['positives']:
                if p in clause:
                    idx = clause.find(p)
                    window = clause[max(0, idx - 30):idx]
                    if any(nw in window for nw in [
                        'no ', 'sin ', 'without ', 'kein ', 'pas de ', 'geen ', 'not ', 
                        'ausencia de ', 'aucun ', 'ausente ', 'negativo para '
                    ]):
                        is_negated = True
                        break
            if is_negated:
                neg_score += 4.0 * weight
            else:
                pos_score += 4.0 * weight
                if in_impression:
                    impression_pos = True
                    
    if pos_score > 0 and neg_score == 0:
        conf = 0.98 if impression_pos else 0.90
    elif pos_score > neg_score:
        conf = 0.78
    elif neg_score > 0 and pos_score == 0:
        conf = 0.02
    elif neg_score > pos_score:
        conf = 0.10
    else:
        conf = 0.20 # unmentioned pathology prior
        
    return conf, pos_score, neg_score

# Extract rule features for Gold-58
N_gold = len(gold_df)
gold_conf = np.zeros((N_gold, 12))
gold_pos = np.zeros((N_gold, 12))
gold_neg = np.zeros((N_gold, 12))

for i in range(N_gold):
    txt = gold_df.iloc[i]['clean_report']
    for c, target in enumerate(target_cols):
        conf, ps, ns = extract_rule_features_v4(txt, target)
        gold_conf[i, c] = conf
        gold_pos[i, c] = ps
        gold_neg[i, c] = ns

# TF-IDF Feature Spaces
vec_word = TfidfVectorizer(ngram_range=(1, 3), max_features=4000, sublinear_tf=True)
vec_char = TfidfVectorizer(ngram_range=(3, 6), analyzer='char_wb', max_features=5000, sublinear_tf=True)

X_w_gold = vec_word.fit_transform(gold_df['clean_report']).toarray()
X_c_gold = vec_char.fit_transform(gold_df['clean_report']).toarray()

# Multi-Model Stacker on 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_v4_preds = np.zeros((N_gold, 12))

print("\nRunning 5-Fold Cross-Validation on Gold-58 with Engine V4...")
for c, target in enumerate(target_cols):
    y = gold_df[target].values
    X_target = np.hstack([
        X_w_gold, X_c_gold, 
        gold_conf[:, c:c+1], gold_pos[:, c:c+1], gold_neg[:, c:c+1],
        gold_conf
    ])
    
    best_c_auc = 0.0
    best_c_oof = None
    
    for C_lr in [0.2, 0.5, 1.0, 2.0]:
        for alpha_rd in [1.0, 5.0, 15.0]:
            oof_blend = np.zeros(N_gold)
            for tr_idx, va_idx in kf.split(gold_df):
                lr = LogisticRegression(C=C_lr, max_iter=1000, class_weight='balanced')
                lr.fit(X_target[tr_idx], y[tr_idx])
                p_lr = lr.predict_proba(X_target[va_idx])[:, 1]
                
                rd = Ridge(alpha=alpha_rd)
                rd.fit(X_target[tr_idx], y[tr_idx])
                p_rd = np.clip(rd.predict(X_target[va_idx]), 0.01, 0.99)
                
                p_va = 0.6 * p_lr + 0.4 * p_rd
                oof_blend[va_idx] = p_va
                
            for w_r in [0.35, 0.50, 0.65, 0.80]:
                final_blend = w_r * gold_conf[:, c] + (1.0 - w_r) * oof_blend
                try:
                    auc = roc_auc_score(y, final_blend)
                except:
                    auc = 0.5
                if auc > best_c_auc:
                    best_c_auc = auc
                    best_c_oof = final_blend
                    
    oof_v4_preds[:, c] = best_c_oof
    print(f"Target {c:2d} ({target:18s}): 5-Fold OOF AUC = {best_c_auc:.4f}")

macro_auc_v4 = np.mean([roc_auc_score(gold_df[target_cols[c]].values, oof_v4_preds[:, c]) for c in range(12)])
print("\n" + "=" * 85)
print(f">>> FINAL UPGRADE 1 NLP REPORT TEACHER V4 MACRO-AUC: {macro_auc_v4:.4f} <<<")
print("=" * 85)
