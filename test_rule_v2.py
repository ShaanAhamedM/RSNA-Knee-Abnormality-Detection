import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.special import expit
from tune_precision_lexicon import TARGET_RULES, clean_text

df = pd.read_csv('train.csv')
gold_df = df[df['ACL'].notnull()].copy().reset_index(drop=True)
gold_df['clean_report'] = gold_df['Report'].apply(clean_text)

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# Refine keywords and negations
REFINED_RULES = {
    'ACL': {
        'keywords': ['acl', 'lca', 'anterior cruciate', 'cruzado anterior', 'croise anterieur', 'vorderes kreuzband', 'vkb'],
        'pathologies': ['tear', 'torn', 'rupture', 'rotura', 'ruptura', 'scheur', 'lesion', 'desgarro', 'avulsion', 'irregular', 'complete tear', 'partial tear', 'high-grade tear', 'disruption'],
        'negations': ['intact', 'normal', 'unremarkable', 'conservad', 'preservad', 'no signs of tear', 'sin signos de rotura', 'no tear', 'sin rotura', 'aucun', 'sans rupture', 'sin alteraciones']
    },
    'MCL': {
        'keywords': ['mcl', 'lcm', 'medial collateral', 'colateral medial', 'colateral interno', 'collatéral médial', 'innenband', 'ligamento colateral medial'],
        'pathologies': ['tear', 'torn', 'rupture', 'rotura', 'ruptura', 'sprain', 'injury', 'grade 1', 'grade 2', 'grade 3', 'esguince', 'aumento de senal', 'edema', 'thickened', 'desgarro'],
        'negations': ['intact', 'normal', 'unremarkable', 'conservad', 'preservad', 'sin alteraciones', 'no tear', 'sin signos de rotura', 'sin rotura', 'continuo']
    },
    'Medial Meniscus': {
        'keywords': ['medial meniscus', 'menisco medial', 'menisco interno', 'menisque medial', 'menisque interne', 'innenmeniskus', 'meniscusscheur/mediaal', 'cuerno posterior del menisco medial', 'cuerno anterior del menisco medial', 'cuerpo del menisco medial', 'mm'],
        'pathologies': ['tear', 'torn', 'rotura', 'ruptura', 'scheur', 'fissure', 'fisura', 'amputacion', 'amputation', 'horizontal', 'vertical', 'radial', 'complex', 'compleja', 'desgarro', 'maceracion', 'extrusion', 'root tear', 'flap'],
        'negations': ['not torn', 'sin signos de rotura', 'sin rotura', 'no tear', 'normal', 'conservad', 'preservad', 'unremarkable', 'intact', 'aucun', 'morfologia y senal conservada', 'sin fisura']
    },
    'Lateral Meniscus': {
        'keywords': ['lateral meniscus', 'menisco lateral', 'menisco externo', 'menisque lateral', 'menisque externe', 'aussenmeniskus', 'cuerno posterior del menisco lateral', 'cuerno anterior del menisco lateral', 'cuerpo del menisco lateral', 'lm'],
        'pathologies': ['tear', 'torn', 'rotura', 'ruptura', 'scheur', 'fissure', 'fisura', 'amputacion', 'amputation', 'horizontal', 'vertical', 'radial', 'complex', 'compleja', 'desgarro', 'maceracion', 'discoid', 'root tear', 'flap'],
        'negations': ['not torn', 'sin signos de rotura', 'sin rotura', 'no tear', 'normal', 'conservad', 'preservad', 'unremarkable', 'intact', 'aucun', 'morfologia y senal conservada', 'sin fisura']
    },
    'Medial OA': {
        'keywords': ['medial', 'femorotibial medial', 'femorotibial interne', 'artrosis medial', 'gonartrosis medial', 'osteoartritis medial', 'medial compartment', 'compartimento medial', 'platillo tibial medial', 'condilo femoral medial'],
        'pathologies': ['artrosis', 'osteoarth', 'chondrosis', 'chondropathy', 'condromalacia', 'condropatia', 'fissure', 'defect', 'ulcera condral', 'ulcer', 'narrowing', 'pinzamiento', 'osteophyte', 'osteofito', 'adelgazamiento', 'perdida de espesor', 'grade 2', 'grade 3', 'grade 4', 'espesor parcial', 'espesor total', 'eburnacion'],
        'negations': ['no focal chondrosis', 'sin alteraciones', 'normal', 'conservad', 'preservad', 'sin signos de artrosis', 'intact', 'unremarkable', 'no chondral', 'sin osteofitos', 'espacio articular conservado']
    },
    'Lateral OA': {
        'keywords': ['lateral', 'femorotibial lateral', 'femorotibial externe', 'artrosis lateral', 'gonartrosis lateral', 'osteoartritis lateral', 'lateral compartment', 'compartimento lateral', 'platillo tibial lateral', 'condilo femoral lateral'],
        'pathologies': ['artrosis', 'osteoarth', 'chondrosis', 'chondropathy', 'condromalacia', 'condropatia', 'fissure', 'defect', 'ulcera condral', 'ulcer', 'narrowing', 'pinzamiento', 'osteophyte', 'osteofito', 'adelgazamiento', 'perdida de espesor', 'grade 2', 'grade 3', 'grade 4', 'espesor parcial', 'espesor total', 'eburnacion'],
        'negations': ['no focal chondrosis', 'sin alteraciones', 'normal', 'conservad', 'preservad', 'sin signos de artrosis', 'intact', 'unremarkable', 'no chondral', 'sin osteofitos', 'espacio articular conservado']
    },
    'PF OA': {
        'keywords': ['patellofemoral', 'femoropatelar', 'femoropatellaire', 'femoropatellar', 'patella', 'rotula', 'trochlea', 'troclea', 'patelofemoral', 'retropatel', 'retropatellar', 'condromalacia rotuliana', 'condromalacia patelar', 'articulacion femororotuliana'],
        'pathologies': ['condromalacia', 'chondromalacia', 'artrosis', 'osteoarth', 'chondrosis', 'defect', 'fissure', 'ulcer', 'ulcera condral', 'adelgazamiento', 'pinzamiento', 'osteophyte', 'osteofito', 'grade 2', 'grade 3', 'grade 4', 'espesor parcial', 'espesor total', 'edema subcondral', 'fissuring', 'cartilage loss'],
        'negations': ['congruent without chondral', 'sin alteraciones', 'normal', 'conservad', 'preservad', 'sin condromalacia', 'intact', 'unremarkable', 'no chondral', 'cartilago articular conservado']
    },
    'Effusion': {
        'keywords': ['effusion', 'derrame', 'epanchement', 'gelenkerguss', 'erguss', 'joint effusion', 'derrame articular', 'suprapatellar recess', 'fondo de saco suprapatelar'],
        'pathologies': ['effusion', 'derrame', 'joint effusion', 'small joint effusion', 'moderate joint effusion', 'large joint effusion', 'derrame articular', 'moderado derrame', 'abundante derrame', 'leve derrame', 'distension', 'fluid collection', 'liquido articular', 'abundant fluid', 'significant effusion', 'mild effusion', 'moderate effusion', 'large effusion', 'distendido por liquido'],
        'negations': ['no joint effusion', 'sin derrame', 'sin derrame articular', 'no effusion', 'no significant effusion', 'sin colecciones', 'ausencia de derrame', 'pas d epanchement', 'kein erguss', 'fisiologico', 'physiological fluid', 'minima cantidad fisiologica']
    },
    'Synovitis': {
        'keywords': ['synovitis', 'sinovitis', 'synovite', 'synovial', 'sinovial', 'synovium'],
        'pathologies': ['synovitis', 'sinovitis', 'synovial thickening', 'engrosamiento sinovial', 'hipertrofia sinovial', 'synovial hypertrophy', 'synovial proliferation', 'pannus', 'hyperintense synovium', 'sinovial reactiva', 'synovitis reactive'],
        'negations': ['no synovitis', 'sin sinovitis', 'no synovial thickening', 'sin engrosamiento sinovial', 'normal', 'sin alteraciones', 'ausencia de sinovitis', 'sin signos de sinovitis']
    },
    "Baker's": {
        'keywords': ['baker', 'popliteal cyst', 'quiste de baker', 'quiste popliteo', 'kyste de baker', 'kyste poplite', 'backer', 'popliteal fossa cyst'],
        'pathologies': ['baker', 'cyst', 'quiste', 'kyste', 'popliteal cyst', 'quiste de baker', 'quiste popliteo', 'distension de la bursa gastrocnemio-semimembranoso'],
        'negations': ['no baker', 'no popliteal cyst', 'sin quiste', 'sin quiste de baker', 'ausencia de quiste', 'no cyst seen', 'sin colecciones popliteas']
    },
    'Contusion': {
        'keywords': ['contusion', 'bone marrow edema', 'edema oseo', 'contusion osea', 'bone contusion', 'marrow edema', 'oedeme osseux', 'knochenmarkodem', 'bone bruise', 'edema trabecular', 'microfractura trabecular'],
        'pathologies': ['contusion', 'edema', 'bone marrow edema', 'edema oseo', 'trabecular microfracture', 'trabecular edema', 'bruise', 'high t2 signal in bone', 'edema subcondral', 'edema en el aspecto'],
        'negations': ['no bone marrow edema', 'no hay alteraciones de senal de la medula osea', 'sin edema oseo', 'no contusion', 'no edema', 'ausencia de edema oseo', 'medula osea sin alteraciones', 'senal osea conservada', 'sin edema trabecular']
    },
    'Fracture': {
        'keywords': ['fracture', 'fractura', 'fracturas', 'cortical disruption', 'cortical defect', 'trabecular fracture', 'fissure osseuse'],
        'pathologies': ['fracture', 'fractura', 'linea de fractura', 'cortical step', 'avulsion fracture', 'displaced fracture', 'fractura trabecular', 'depressed fracture', 'fractura por estres', 'fracture line'],
        'negations': ['no fracture', 'no fracture is seen', 'sin fractura', 'sin fracturas', 'no signs of fracture', 'ausencia de fractura', 'pas de fracture', 'keine fraktur', 'aucune fracture', 'sin trazos de fractura', 'no evidence of fracture', 'no cortical step']
    }
}

def extract_rule_score_v2(text, target):
    rule = REFINED_RULES[target]
    clauses = re.split(r'[\.\;\n\*\•]', text)
    
    pos_evidence = 0.0
    neg_evidence = 0.0
    kw_found = False
    
    general_negs = [
        'no ', 'sin ', 'without ', 'intact', 'normal', 'unremarkable', 
        'conservad', 'preservad', 'ausencia de', 'pas de', 'kein', 'aucun',
        'sin evidencia de', 'no evidence of', 'negative for'
    ]
    
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        has_kw = any(kw in clause for kw in rule['keywords'])
        if not has_kw:
            continue
        kw_found = True
        
        has_rule_neg = any(n in clause for n in rule['negations'])
        has_gen_neg = any(gn in clause for gn in general_negs)
        has_path = any(p in clause for p in rule['pathologies'])
        
        if has_rule_neg or (has_gen_neg and not has_path):
            neg_evidence += 4.0
        elif has_rule_neg and has_path:
            neg_evidence += 3.0
        elif has_path:
            # Check if there is an explicit negation preceding the pathology word
            # e.g., 'no fracture', 'no tear'
            is_negated = False
            for p in rule['pathologies']:
                if p in clause:
                    p_idx = clause.find(p)
                    window = clause[max(0, p_idx - 25):p_idx]
                    if any(gn in window for gn in ['no ', 'sin ', 'without ', 'not ', 'pas de ', 'kein ']):
                        is_negated = True
                        break
            if is_negated:
                neg_evidence += 4.0
            else:
                pos_evidence += 4.0
                
    if pos_evidence > 0 and neg_evidence == 0:
        score = 4.0
    elif pos_evidence > neg_evidence:
        score = 2.0
    elif neg_evidence > 0:
        score = -4.0
    elif kw_found:
        score = -2.0
    else:
        score = -3.0
        
    return expit(score)

print("=" * 80)
print("EVALUATING REFINED LOGIC ON 58 GOLD STUDIES")
print("=" * 80)
aucs = []
for target in target_cols:
    y = gold_df[target].values
    scores = np.array([extract_rule_score_v2(t, target) for t in gold_df['clean_report']])
    auc = roc_auc_score(y, scores)
    aucs.append(auc)
    print(f"{target:18s}: AUC = {auc:.4f} (Positive: {int(y.sum())}/{len(y)})")

print("=" * 80)
print(f">>> REFINED RULE ENGINE MACRO-AUC: {np.mean(aucs):.4f} <<<")
print("=" * 80)
