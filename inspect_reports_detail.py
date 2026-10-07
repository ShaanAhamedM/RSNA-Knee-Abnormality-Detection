import pandas as pd
from tune_precision_lexicon import clean_text

df = pd.read_csv('train.csv')
gold_df = df[df['ACL'].notnull()].copy().reset_index(drop=True)
gold_df['clean_report'] = gold_df['Report'].apply(clean_text)

for target in ['Effusion', 'Contusion', 'Fracture', 'Synovitis']:
    print(f"\n{'='*80}\nTARGET: {target} (Total 1s: {gold_df[target].sum()}/{len(gold_df)})\n{'='*80}")
    pos_reports = gold_df[gold_df[target] == 1]['clean_report'].values
    neg_reports = gold_df[gold_df[target] == 0]['clean_report'].values
    
    print(f"--- 3 POSITIVE REPORTS FOR {target} ---")
    for r in pos_reports[:3]:
        print(f"  [POS]: {r[:300]}...\n")
        
    print(f"--- 3 NEGATIVE REPORTS FOR {target} ---")
    for r in neg_reports[:3]:
        print(f"  [NEG]: {r[:300]}...\n")
