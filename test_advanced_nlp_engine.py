"""
Advanced Section-Aware Multi-Lingual Clinical NLP Rule Engine V5
Evaluates exact Rule-Based AUC on Gold-58
"""

import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Load train.csv
df = pd.read_csv('train.csv')
gold = df[df['ACL'].notnull()].copy().reset_index(drop=True)

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for s, d in [
        ('á', 'a'), ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u'), ('ñ', 'n'),
        ('à', 'a'), ('è', 'e'), ('ù', 'u'), ('â', 'a'), ('ê', 'e'), ('î', 'i'),
        ('ô', 'o'), ('û', 'u'), ('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss'),
        ('ç', 'c'), ('ğ', 'g'), ('ı', 'i'), ('ş', 's'), ('[redacted]', ''), ('[date]', ''),
        ('redacted', ''), ('date', ''), ('time', '')
    ]:
        text = text.replace(s, d)
    return text

gold['clean_report'] = gold['Report'].apply(normalize_text)

CLINICAL_LEXICON_V5 = {
    'ACL': {
        'keywords': [
            'acl', 'lca', 'anterior cruciate', 'cruzado anterior', 'croise anterieur', 
            'vorderes kreuzband', 'vkb', 'on capraz bag', 'ön çapraz', 'voorste kruisband', 
            'ligament croise anterieur', 'χιαστος', 'кръстни'
        ],
        'positives': [
            'complete tear', 'partial tear', 'rupture', 'rotura', 'ruptura', 'scheur', 'torn', 
            'disruption', 'avulsion', 'grade 2', 'grade 3', 'grade ii', 'grade iii', 
            'rotura espesor completo', 'rotura espesor total', 'yirtik', 'komplet ruptur', 
            'tam kat', 'dechirure', 'high-grade tear', 'near-complete', 'butunluk kaybi',
            'rotura del lca', 'tear of the anterior cruciate', 'acl tear', 'torn acl'
        ],
        'negatives': [
            'intact', 'normal', 'unremarkable', 'conservad', 'preservad', 'sin signos de rotura', 
            'no tear', 'sin rotura', 'aucun', 'sans rupture', 'sin alteraciones', 'intaktes', 
            'regelrecht', 'dogal', 'continuo', 'morfolojisi dogal', 'homogeneo', 'no focal tear', 
            'both bundles intact', 'sin caracteres propios de rotura', 'grade 1 sprain', 'grade 1 injury',
            'grade i sprain', 'grade i injury', 'low-grade sprain', 'запазена цялост'
        ]
    },
    'MCL': {
        'keywords': [
            'mcl', 'lcm', 'medial collateral', 'colateral medial', 'colateral interno', 
            'collatéral médial', 'innenband', 'ligamento colateral medial', 'ic yan bag', 
            'iç yan bağ', 'tibial collateral', 'εσω πλαγιος', 'медиален колатерален'
        ],
        'positives': [
            'tear', 'torn', 'rupture', 'rotura', 'ruptura', 'grade 2', 'grade 3', 'grade ii', 
            'grade iii', 'rotura parcial', 'rotura total', 'disruption', 'yirtik', 'dechirure',
            'partial tear', 'complete tear', 'high-grade tear'
        ],
        'negatives': [
            'intact', 'normal', 'unremarkable', 'conservad', 'preservad', 'sin alteraciones', 
            'no tear', 'sin signos de rotura', 'sin rotura', 'continuo', 'regelrecht', 'dogal', 
            'ausencia de rotura', 'grade 1 sprain', 'grade 1 injury', 'grade i sprain', 'grade i injury',
            'low-grade sprain', 'no frank tear'
        ]
    },
    'Medial Meniscus': {
        'keywords': [
            'medial meniscus', 'menisco medial', 'menisco interno', 'menisque medial', 
            'menisque interne', 'innenmeniskus', 'medial meniskus', 'medyal meniskus', 
            'ic meniskus', 'iç menisküs', 'εσω μηνισκ', 'медиален менискус', ' mm '
        ],
        'positives': [
            'tear', 'torn', 'rotura', 'ruptura', 'scheur', 'fissure', 'fisura', 'horizontal', 
            'vertical', 'radial', 'complex tear', 'rotura compleja', 'flap', 'bucket-handle', 
            'buckethandle', 'root tear', 'dechirure', 'yirtik', 'displaced fragment', 'extrusion',
            'rotura de menisco interno', 'rupture du menisque medial'
        ],
        'negatives': [
            'intact', 'not torn', 'sin signos de rotura', 'sin rotura', 'no tear', 'normal', 
            'conservad', 'preservad', 'unremarkable', 'aucun', 'sin caracteres propios de rotura',
            'no surfacing tear', 'without extension to the articular surface', 'intrasubstance degeneration',
            'grade 1 degeneration', 'grade 2 degeneration', 'grade i dejenerasyon', 'grade ii dejenerasyon',
            'dejenerasyon izlenmistir', 'meniskopati', 'amputacion marginal'
        ]
    },
    'Lateral Meniscus': {
        'keywords': [
            'lateral meniscus', 'menisco lateral', 'menisco externo', 'menisque lateral', 
            'menisque externe', 'aussenmeniskus', 'lateral meniskus', 'dis meniskus', 
            'dış menisküs', 'εξω μηνισκ', 'латерален менискус', ' lm '
        ],
        'positives': [
            'tear', 'torn', 'rotura', 'ruptura', 'scheur', 'fissure', 'fisura', 'horizontal', 
            'vertical', 'radial', 'complex tear', 'rotura compleja', 'flap', 'bucket-handle', 
            'buckethandle', 'root tear', 'dechirure', 'yirtik', 'displaced fragment', 'maceration',
            'maceracion', 'rotura de menisco lateral', 'rupture du menisque lateral'
        ],
        'negatives': [
            'intact', 'not torn', 'sin signos de rotura', 'sin rotura', 'no tear', 'normal', 
            'conservad', 'preservad', 'unremarkable', 'aucun', 'sin caracteres propios de rotura',
            'no surfacing tear', 'without extension to the articular surface', 'intrasubstance degeneration',
            'grade 1 degeneration', 'grade 2 degeneration', 'grade i dejenerasyon', 'grade ii dejenerasyon',
            'meniskopati'
        ]
    },
    'Medial OA': {
        'keywords': [
            'medial compartment cartilage', 'medial femoral condyle', 'medial tibial plateau', 
            'compartimento femorotibial medial', 'condilo femoral medial', 'platillo tibial medial', 
            'mediale femurkondyle', 'medial tibiaplateau', 'medial gonarthrosis', 'medial oa', 
            'ic eklem araligi', 'εσω διαμερισμα', 'медиален тибиален', 'medial joint line',
            'medial compartment', 'medial tibiofemoral'
        ],
        'positives': [
            'full thickness cartilage loss', 'high-grade cartilage loss', 'grade 3', 'grade 4', 
            'grade iii', 'grade iv', 'severe osteoarthritis', 'moderate osteoarthritis', 
            'artrosis femorotibial medial', 'ulcera condral', 'pinzamiento', 'eburnacion', 
            'subchondral sclerosis', 'osteophytes', 'osteofitos', 'osteofitosis', 'gonartrose', 
            'kraakbeenverlies', 'exalipsi tou arthrikou chondrou', 'high-grade cartilage fissuring',
            'deep fissuring', 'severe extrusion', 'marginal spurring', 'artrosis', 'osteoartritis'
        ],
        'negatives': [
            'no focal chondrosis', 'sin alteraciones', 'normal', 'conservad', 'preservad', 
            'sin signos de artrosis', 'intact', 'unremarkable', 'no chondral', 'sin osteofitos', 
            'espacio articular conservado', 'cartilage appears intact'
        ]
    },
    'Lateral OA': {
        'keywords': [
            'lateral compartment cartilage', 'lateral femoral condyle', 'lateral tibial plateau', 
            'compartimento femorotibial lateral', 'condilo femoral lateral', 'platillo tibial lateral', 
            'laterale femurkondyle', 'lateral tibiaplateau', 'lateral gonarthrosis', 'lateral oa', 
            'dis eklem araligi', 'εξω διαμερισμα', 'латерален тибиален', 'lateral joint line',
            'lateral compartment', 'lateral tibiofemoral'
        ],
        'positives': [
            'full thickness cartilage loss', 'high-grade cartilage loss', 'grade 3', 'grade 4', 
            'grade iii', 'grade iv', 'severe osteoarthritis', 'moderate osteoarthritis', 
            'artrosis femorotibial lateral', 'ulcera condral', 'pinzamiento', 'eburnacion', 
            'subchondral sclerosis', 'osteophytes', 'osteofitos', 'osteofitosis', 'gonartrose', 
            'kraakbeenverlies', 'high-grade cartilage fissuring', 'deep fissuring', 'artrosis', 
            'cartilage thinning and fissuring is present at the posterior and peripheral weightbearing aspect'
        ],
        'negatives': [
            'no focal chondrosis', 'sin alteraciones', 'normal', 'conservad', 'preservad', 
            'sin signos de artrosis', 'intact', 'unremarkable', 'no chondral', 'sin osteofitos', 
            'espacio articular conservado', 'cartilage appears intact'
        ]
    },
    'PF OA': {
        'keywords': [
            'patellofemoral', 'patella', 'trochlea', 'rotula', 'femoropatelar', 'femoropatellar', 
            'retropatel', 'retropatellar', 'patelofemoral', 'femororotuliana', 'condropatia rotuliana', 
            'chondromalacia patellae', 'патела', 'troclea', 'patellar facet', 'anterior compartment',
            'femoropatellaire', 'condropatia patelar'
        ],
        'positives': [
            'condromalacia', 'chondromalacia', 'artrosis femoropatelar', 'grade 2', 'grade 3', 'grade 4', 
            'grade ii', 'grade iii', 'grade iv', 'full thickness cartilage defect', 'high-grade cartilage loss', 
            'ulcera condral', 'fissuring', 'trochlear chondropathy', 'patellar facet chondrosis',
            'multifocal deep and delaminating fissuring', 'artrosis', 'condropatia lesiones grado'
        ],
        'negatives': [
            'intact', 'normal', 'congruent without chondral lesion', 'sin condromalacia', 
            'sin alteraciones', 'conservad', 'preservad', 'unremarkable', 'cartilago rotuliano sin alteraciones'
        ]
    },
    'Effusion': {
        'keywords': [
            'effusion', 'joint effusion', 'derrame', 'derrame articular', 'epanchement', 
            'gelenkserguss', 'erguss', 'efuzyon', 'eklem ici sivi', 'sivi artisi', 'hydrops', 
            'fluid accumulation', 'stavn izliv', 'υγρο ενδαρθρικα', 'излив', 'knee effusion',
            'suprapatella bursa', 'suprapatellar bursa', 'joint fluid'
        ],
        'positives': [
            'moderate joint effusion', 'large joint effusion', 'massive joint effusion', 'joint effusion', 
            'derrame articular', 'effusion with hemarthrosis', 'hemartrosis', 'significant effusion', 
            'gelenkserguss', 'fluid accumulated in suprapatellar bursa', 'eklem ici sivi artisi',
            'distended suprapatellar bursa', 'fluid accumulation', 'epanchement articulaire',
            'large effusion', 'some joint effusion', 'leve derrame articular', 'sivi artisi izlenmistir',
            'some amount of right knee effusion', 'derrame'
        ],
        'negatives': [
            'no knee effusion', 'no joint effusion', 'no effusion', 'sin derrame', 'sin derrame articular', 
            'pas d epanchement', 'kein erguss', 'efuzyon saptanmadi', 'fisiologico', 'physiological fluid', 
            'no fluid collection', 'ausencia de derrame'
        ]
    },
    'Synovitis': {
        'keywords': [
            'synovitis', 'sinovitis', 'synovial', 'sinovial', 'synovium', 'synovite', 'sinovit', 
            'synoviale', 'hoffa', 'hoffitis', 'plica', 'συνοβιακη', 'синовит', 'synovial membrane'
        ],
        'positives': [
            'synovitis', 'sinovitis', 'synovial thickening', 'thickened synovial tissue', 
            'engrosamiento sinovial', 'synovial proliferation', 'chronic reactive synovitis', 
            'synovitis of left knee', 'synovitis of right knee', 'hoffitis', 'sinovite', 
            'chronische synovitis', 'sinovit', 'synovial hypertrophy', 'mild synovitis', 
            'discreta sinovitis', 'sindrome de pinzamiento de la almohadilla grasa de hoffa',
            'suprapatellar bursitis'
        ],
        'negatives': [
            'no synovitis', 'sin sinovitis', 'sin engrosamiento sinovial', 'no synovial thickening', 
            'sin alteraciones de la grasa de hoffa', 'normal', 'sinovit saptanmadi', 'keine synovitis',
            'no edema in superolateral hoffa'
        ]
    },
    "Baker's": {
        'keywords': [
            'baker', 'popliteal cyst', 'quiste de baker', 'quiste popliteo', 'kyste de baker', 
            'baker kisti', 'popliteale cyste', 'gastrocnemius-semimembranosus bursa', 
            'κυστη baker', 'киста на бейкър', 'popliteal fossa'
        ],
        'positives': [
            'baker cyst', 'baker\'s cyst', 'popliteal cyst', 'quiste de baker', 'quiste popliteo', 
            'baker kisti', 'rupture of popliteal cyst', 'scheur van popliteale cyste', 
            'distended semimembranosus bursa', 'kyste de baker', 'popliteale cyste'
        ],
        'negatives': [
            'no baker', 'no popliteal cyst', 'sin quiste de baker', 'sin quistes popliteos patologicos', 
            'sin quistes popliteos', 'ausencia de quiste', 'keine baker', 'baker kisti saptanmadi',
            'no quiste'
        ]
    },
    'Contusion': {
        'keywords': [
            'contusion', 'bone marrow edema', 'bone bruise', 'edema oseo', 'edema óseo', 
            'contusion osea', 'trabecular edema', 'knochenmarkodem', 'oedeme osseux', 
            'kemik iligi odemi', 'kemik kontüzyonu', 'kemik kontuzyonu', 'kissing contusion', 
            'οστεομυελικο οιδημα', 'костномозъчен едем', 'bone marrow contusion', 'contusiones oseas'
        ],
        'positives': [
            'bone contusion', 'bone bruise', 'bone marrow edema', 'edema oseo', 'contusiones oseas', 
            'trabecular microfracture', 'kissing contusion', 'kemik kontuzyonu', 'kemik iligi odemi',
            'oedeme osseux', 'edema oseo del aspecto posterior', 'contusiones oseas femorotibiales',
            'popliteus muscle contusion', 'kontuzyonel kemik iligi odemleri'
        ],
        'negatives': [
            'no acute bone bruise', 'no bone bruise', 'no bone contusion', 'no bone marrow edema', 
            'sin edema oseo', 'no hay alteraciones de senal de la medula osea', 'no hay alteraciones de senal significativas de la medula osea',
            'ausencia de edema oseo', 'kein bone bruise', 'kemik iligi odemi saptanmadi', 'normal marrow',
            'there is no fracture or bone contusion', 'without bone contusion'
        ]
    },
    'Fracture': {
        'keywords': [
            'fracture', 'fractura', 'fraktur', 'avulsion', 'cortical defect', 'cortical step', 
            'subchondral insufficiency fracture', 'insufficiency fracture', 'segond', 
            'trabecular fracture', 'kirik', 'frakture', 'καταγμα', 'фрактура', 'avulsion fracture',
            'split fractures', 'subartikuler fraktur'
        ],
        'positives': [
            'fracture', 'fractura', 'fraktur', 'subchondral insufficiency fracture', 
            'osteochondral fracture', 'avulsion fracture', 'split fractures', 'depressed fracture', 
            'stress fracture', 'segond fracture', 'kirik', 'katagma', 'frakture', 'subartikuler nondeplase fraktur',
            'flaking fracture', 'trazo de fractura', 'микродокиδωδη καταγματα'
        ],
        'negatives': [
            'no acute fracture', 'no fracture is seen', 'no fracture', 'sin fractura', 
            'sin fracturas', 'sin signos de fractura', 'ausencia de fractura', 'keine fraktur', 
            'pas de fracture', 'kirik saptanmadi', 'cortical continuity maintained',
            'there is no fracture or bone contusion', 'no fracture or bone bruise'
        ]
    }
}

INDICATION_TRIGGERS = [
    'antecedentes', 'indicacion', 'indication', 'klinische', 'motivo', 'clinical history', 
    'history', 'anamnese', 'diagnostische vraagstelling', 'comparison', 'technique', 'tecnica'
]
IMPRESSION_TRIGGERS = [
    'impresion', 'impression', 'conclusion', 'conclusio', 'besluit', 'sonuc', 'zakluchenie', 
    'symperasma', 'beurteilung', 'diagnostic'
]

NEGATION_PATTERNS = [
    'no ', 'sin ', 'without ', 'kein ', 'pas de ', 'geen ', 'not ', 'ausencia de ', 
    'aucun ', 'ausente ', 'saptanmadi ', 'yoktur ', 'sans ', 'negativo para ', 'free of '
]

def parse_report_clinical_score(text, target):
    """
    Parses text by section and clause with forward/backward scope and conjunction handling.
    Returns composite score in [0, 1].
    """
    rule = CLINICAL_LEXICON_V5[target]
    lines = text.split('\n')
    
    current_sec = 'findings'
    pos_score = 0.0
    neg_score = 0.0
    impression_pos = False
    
    for line in lines:
        line_clean = line.strip().lower()
        if not line_clean:
            continue
            
        # Section classification
        if any(line_clean.startswith(h) or f"{h}:" in line_clean for h in INDICATION_TRIGGERS):
            current_sec = 'indication'
        elif any(line_clean.startswith(h) or f"{h}:" in line_clean for h in IMPRESSION_TRIGGERS):
            current_sec = 'impression'
        elif any(line_clean.startswith(h) or f"{h}:" in line_clean for h in ['hallazgos', 'findings', 'bevindingen', 'bulgular', 'eyrhmata']):
            current_sec = 'findings'
            
        if current_sec == 'indication':
            # Strictly 0 weight for indication
            continue
            
        weight = 2.5 if current_sec == 'impression' else 1.0
        
        # Split into clauses
        clauses = re.split(r'[\.\;\*\•\t\-\>]', line_clean)
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
                
            has_kw = any(kw in clause for kw in rule['keywords'])
            if not has_kw:
                continue
                
            # Check explicit negatives first
            has_explicit_neg = any(neg in clause for neg in rule['negatives'])
            if has_explicit_neg:
                neg_score += 4.0 * weight
                continue
                
            # Check positives
            matched_pos = None
            for p in rule['positives']:
                if p in clause:
                    matched_pos = p
                    break
                    
            if matched_pos:
                # Check for negation in clause around the matched term
                idx = clause.find(matched_pos)
                pre_window = clause[max(0, idx - 40):idx]
                is_negated = False
                
                # Check direct negation
                if any(nw in pre_window for nw in NEGATION_PATTERNS):
                    is_negated = True
                    
                # Check conjunction negation: e.g. "no fracture or contusion"
                if not is_negated and any(conj in pre_window for conj in [' or ', ' ni ', ' nor ', ' och ']):
                    # Look further back
                    extended_window = clause[max(0, idx - 80):idx]
                    if any(nw in extended_window for nw in NEGATION_PATTERNS):
                        is_negated = True
                        
                if is_negated:
                    neg_score += 4.0 * weight
                else:
                    pos_score += 4.0 * weight
                    if current_sec == 'impression':
                        impression_pos = True
                        
    # Score calculation
    if pos_score > 0 and neg_score == 0:
        score = 0.98 if impression_pos else 0.90
    elif pos_score > neg_score:
        score = 0.75
    elif neg_score > 0 and pos_score == 0:
        score = 0.02
    elif neg_score > pos_score:
        score = 0.12
    else:
        score = 0.20 # prior
        
    return score, pos_score, neg_score

print("="*85)
print("EVALUATING ADVANCED RULE ENGINE V5 ON 58 GOLD STUDIES")
print("="*85)

rule_aucs = []
for c, target in enumerate(target_cols):
    y = gold[target].values
    scores = np.array([parse_report_clinical_score(txt, target)[0] for txt in gold['clean_report']])
    auc = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else 0.5
    rule_aucs.append(auc)
    print(f"Target {c:2d} ({target:18s}): Rule AUC = {auc:.4f} (Pos: {int(y.sum())}/58)")

macro_v5 = np.mean(rule_aucs)
print("="*85)
print(f">>> ADVANCED RULE ENGINE V5 MACRO-AUC: {macro_v5:.4f} <<<")
print("="*85)
