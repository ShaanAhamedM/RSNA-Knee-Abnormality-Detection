import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.special import expit

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

# Precision Multilingual Dictionary
TARGET_RULES = {
    'ACL': {
        'keywords': ['acl', 'lca', 'anterior cruciate', 'cruzado anterior', 'croise anterieur', 'vorderes kreuzband', 'vkb'],
        'pathologies': ['tear', 'torn', 'rupture', 'rotura', 'ruptura', 'scheur', 'lesion', 'desgarro', 'avulsion', 'irregular', 'high signal', 'aumento de senal', 'complete', 'incompleto', 'interstitial'],
        'negations': ['intact', 'normal', 'unremarkable', 'conservad', 'preservad', 'no signs of tear', 'sin signos de rotura', 'no tear', 'sin rotura', 'aucun', 'sans rupture']
    },
    'MCL': {
        'keywords': ['mcl', 'lcm', 'medial collateral', 'colateral medial', 'colateral interno', 'collatéral médial', 'innenband', 'ligamento colateral'],
        'pathologies': ['tear', 'torn', 'rupture', 'rotura', 'ruptura', 'sprain', 'injury', 'grade 1', 'grade 2', 'grade 3', 'esguince', 'aumento de senal', 'edema', 'thickened'],
        'negations': ['intact', 'normal', 'unremarkable', 'conservad', 'preservad', 'sin alteraciones', 'no tear', 'sin signos de rotura', 'sin rotura']
    },
    'Medial Meniscus': {
        'keywords': ['medial meniscus', 'menisco medial', 'menisco interno', 'menisque medial', 'menisque interne', 'innenmeniskus', 'meniscusscheur/mediaal', 'cuerno posterior del menisco medial', 'cuerno anterior del menisco medial', 'cuerpo del menisco medial'],
        'pathologies': ['tear', 'torn', 'rotura', 'ruptura', 'scheur', 'fissure', 'fisura', 'amputacion', 'amputation', 'horizontal', 'vertical', 'radial', 'complex', 'compleja', 'desgarro', 'maceracion'],
        'negations': ['not torn', 'sin signos de rotura', 'sin rotura', 'no tear', 'normal', 'conservad', 'preservad', 'unremarkable', 'intact', 'aucun', 'morfologia y senal conservada']
    },
    'Lateral Meniscus': {
        'keywords': ['lateral meniscus', 'menisco lateral', 'menisco externo', 'menisque lateral', 'menisque externe', 'aussenmeniskus', 'cuerno posterior del menisco lateral', 'cuerno anterior del menisco lateral', 'cuerpo del menisco lateral'],
        'pathologies': ['tear', 'torn', 'rotura', 'ruptura', 'scheur', 'fissure', 'fisura', 'amputacion', 'amputation', 'horizontal', 'vertical', 'radial', 'complex', 'compleja', 'desgarro', 'maceracion', 'discoid'],
        'negations': ['not torn', 'sin signos de rotura', 'sin rotura', 'no tear', 'normal', 'conservad', 'preservad', 'unremarkable', 'intact', 'aucun', 'morfologia y senal conservada']
    },
    'Medial OA': {
        'keywords': ['medial oa', 'femorotibial medial', 'femorotibial interne', 'artrosis medial', 'gonartrosis medial', 'osteoartritis medial', 'medial compartment', 'compartimento medial', 'platillo tibial medial', 'condilo femoral medial'],
        'pathologies': ['artrosis', 'osteoarth', 'chondrosis', 'chondropathy', 'condromalacia', 'condropatia', 'fissure', 'defect', 'ulcera condral', 'ulcer', 'narrowing', 'pinzamiento', 'osteophyte', 'osteofito', 'adelgazamiento', 'perdida de espesor', 'grade 2', 'grade 3', 'grade 4', 'espesor parcial', 'espesor total'],
        'negations': ['no focal chondrosis', 'sin alteraciones', 'normal', 'conservad', 'preservad', 'sin signos de artrosis', 'intact', 'unremarkable', 'no chondral']
    },
    'Lateral OA': {
        'keywords': ['lateral oa', 'femorotibial lateral', 'femorotibial externe', 'artrosis lateral', 'gonartrosis lateral', 'osteoartritis lateral', 'lateral compartment', 'compartimento lateral', 'platillo tibial lateral', 'condilo femoral lateral'],
        'pathologies': ['artrosis', 'osteoarth', 'chondrosis', 'chondropathy', 'condromalacia', 'condropatia', 'fissure', 'defect', 'ulcera condral', 'ulcer', 'narrowing', 'pinzamiento', 'osteophyte', 'osteofito', 'adelgazamiento', 'perdida de espesor', 'grade 2', 'grade 3', 'grade 4', 'espesor parcial', 'espesor total'],
        'negations': ['no focal chondrosis', 'sin alteraciones', 'normal', 'conservad', 'preservad', 'sin signos de artrosis', 'intact', 'unremarkable', 'no chondral']
    },
    'PF OA': {
        'keywords': ['pf oa', 'patellofemoral', 'femoropatelar', 'femoropatellaire', 'femoropatellar', 'patella', 'rotula', 'trochlea', 'troclea', 'patelofemoral', 'retropatel', 'retropatellar', 'condromalacia rotuliana', 'condromalacia patelar'],
        'pathologies': ['condromalacia', 'chondromalacia', 'artrosis', 'osteoarth', 'chondrosis', 'defect', 'fissure', 'ulcer', 'ulcera condral', 'adelgazamiento', 'pinzamiento', 'osteophyte', 'osteofito', 'grade 2', 'grade 3', 'grade 4', 'espesor parcial', 'espesor total', 'edema subcondral'],
        'negations': ['congruent without chondral', 'sin alteraciones', 'normal', 'conservad', 'preservad', 'sin condromalacia', 'intact', 'unremarkable', 'no chondral']
    },
    'Effusion': {
        'keywords': ['effusion', 'derrame', 'epanchement', 'gelenkerguss', 'erguss', 'fluid', 'liquido', 'suprapatellar', 'suprapatelar', 'joint effusion', 'derrame articular'],
        'pathologies': ['effusion', 'derrame', 'joint effusion', 'small joint effusion', 'moderate joint effusion', 'large joint effusion', 'derrame articular', 'moderado derrame', 'abundante derrame', 'leve derrame', 'distension', 'fluid collection', 'liquido libre', 'liquido articular', 'abundant fluid'],
        'negations': ['no joint effusion', 'sin derrame', 'sin derrame articular', 'no effusion', 'no fluid', 'ausencia de derrame', 'pas d epanchement', 'kein erguss']
    },
    'Synovitis': {
        'keywords': ['synovitis', 'sinovitis', 'synovite', 'synovial', 'sinovial', 'synovium', 'plica'],
        'pathologies': ['synovitis', 'sinovitis', 'synovial thickening', 'engrosamiento sinovial', 'hipertrofia sinovial', 'synovial hypertrophy', 'synovial proliferation', 'pannus', 'hyperintense synovium', 'sinovial reactiva'],
        'negations': ['no synovitis', 'sin sinovitis', 'no synovial thickening', 'sin engrosamiento sinovial', 'normal', 'sin alteraciones', 'ausencia de sinovitis']
    },
    "Baker's": {
        'keywords': ['baker', 'popliteal cyst', 'quiste de baker', 'quiste popliteo', 'kyste de baker', 'kyste poplite', 'backer', 'popliteal', 'gastrocnemius-semimembranosus', 'gastrocnemio-semimembranoso'],
        'pathologies': ['baker', 'cyst', 'quiste', 'kyste', 'popliteal cyst', 'quiste de baker', 'quiste popliteo', 'collection', 'distension de bursa'],
        'negations': ['no baker', 'no popliteal cyst', 'sin quiste', 'sin quiste de baker', 'ausencia de quiste', 'no cyst seen']
    },
    'Contusion': {
        'keywords': ['contusion', 'bone marrow edema', 'edema oseo', 'contusion osea', 'bone contusion', 'marrow edema', 'oedeme osseux', 'knochenmarkodem', 'bone bruise', 'edema trabecular', 'microfractura trabecular'],
        'pathologies': ['contusion', 'edema', 'bone marrow edema', 'edema oseo', 'trabecular microfracture', 'trabecular edema', 'bruise', 'high t2 signal in bone', 'edema subcondral'],
        'negations': ['no bone marrow edema', 'no hay alteraciones de senal de la medula osea', 'sin edema oseo', 'no contusion', 'no edema', 'ausencia de edema oseo', 'medula osea sin alteraciones']
    },
    'Fracture': {
        'keywords': ['fracture', 'fractura', 'fracturas', 'cortical disruption', 'cortical defect', 'avulsion', 'trabecular fracture', 'fissure osseuse'],
        'pathologies': ['fracture', 'fractura', 'linea de fractura', 'cortical step', 'avulsion fracture', 'displaced fracture', 'fractura trabecular', 'depressed fracture'],
        'negations': ['no fracture', 'no fracture is seen', 'sin fractura', 'sin fracturas', 'no signs of fracture', 'ausencia de fractura', 'pas de fracture', 'keine fraktur', 'aucune fracture']
    }
}

def extract_rule_score(text, target):
    rule = TARGET_RULES[target]
    clauses = re.split(r'[\.\;\n\*\>\•]', text)
    score = 0.0
    
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        has_kw = any(kw in clause for kw in rule['keywords'])
        if not has_kw:
            continue
        has_path = any(p in clause for p in rule['pathologies'])
        has_neg = any(n in clause for n in rule['negations'])
        
        if has_path and not has_neg:
            score += 2.0
        elif has_neg and not has_path:
            score -= 2.0
        elif has_neg and has_path:
            score -= 1.0
            
    return expit(score)

print("=" * 80)
print("EVALUATING PRECISION MULTILINGUAL RULE ENGINE ON 58 GOLD STUDIES")
print("=" * 80)

rule_preds = np.zeros((len(gold_df), 12))
for i, row in gold_df.iterrows():
    text = row['clean_report']
    for c, target in enumerate(target_cols):
        rule_preds[i, c] = extract_rule_score(text, target)

aucs = []
for c, target in enumerate(target_cols):
    y = gold_df[target].values
    auc = roc_auc_score(y, rule_preds[:, c])
    aucs.append(auc)
    print(f"{target:18s}: Rule-Based AUC = {auc:.4f}")

mean_auc = np.mean(aucs)
print("=" * 80)
print(f"PRECISION MULTILINGUAL ENGINE MACRO-AUC: {mean_auc:.4f}")
print("=" * 80)
