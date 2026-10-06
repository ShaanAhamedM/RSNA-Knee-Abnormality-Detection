"""
Inspect the 4 lowest targets in Gold-58:
Effusion, Synovitis, Contusion, Fracture
"""

import pandas as pd
import numpy as np
from upgrade_nlp_engine_v4 import clean_text_v4, extract_rule_features_v4

df = pd.read_csv('train.csv')
gold = df[df['ACL'].notnull()].copy().reset_index(drop=True)
gold['clean_report'] = gold['Report'].apply(clean_text_v4)

for target in ['Effusion', 'Synovitis', 'Contusion', 'Fracture']:
    print("=" * 80)
    print(f"ANALYZING TARGET: {target}")
    print("=" * 80)
    y = gold[target].values
    for i in range(len(gold)):
        txt = gold.iloc[i]['clean_report']
        conf, ps, ns = extract_rule_features_v4(txt, target)
        # Check disagreement
        if (conf >= 0.5 and y[i] == 0) or (conf < 0.4 and y[i] == 1):
            err_type = "FALSE POSITIVE" if conf >= 0.5 else "FALSE NEGATIVE"
            print(f"[{err_type}] Study {i} (UID: {gold.iloc[i]['StudyInstanceUID'][-6:]}) | True={int(y[i])} Conf={conf:.2f} PosS={ps:.1f} NegS={ns:.1f}")
            # Find lines mentioning target terms
            relevant_lines = [line for line in txt.split('.') if any(w in line for w in ['effusion', 'derrame', 'fluid', 'synov', 'sinov', 'contus', 'edema', 'fractur', 'fraktur', 'bruise'])]
            for r in relevant_lines[:4]:
                print(f"   --> {r.strip()}")
            print()
