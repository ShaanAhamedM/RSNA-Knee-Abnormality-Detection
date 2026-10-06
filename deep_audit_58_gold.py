"""
Deep clinical audit of 58 Gold studies across all 12 pathologies.
Analyzes report sections, vocabulary, negations, and true labels.
Writes cleanly to UTF-8 file.
"""

import sys
import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
gold = df[df['ACL'].notnull()].copy().reset_index(drop=True)

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

with open("gold_58_clinical_audit.txt", "w", encoding="utf-8") as f:
    f.write(f"Total Gold Studies: {len(gold)}\n")
    f.write("=" * 100 + "\n")

    for idx, row in gold.iterrows():
        pos_targets = [t for t in target_cols if row[t] == 1]
        neg_targets = [t for t in target_cols if row[t] == 0]
        f.write(f"\n--- STUDY {idx+1}/{len(gold)} [UID: {row['StudyInstanceUID']}] ---\n")
        f.write(f"POS: {pos_targets}\n")
        f.write(f"NEG: {neg_targets}\n")
        f.write("REPORT TEXT:\n")
        f.write(str(row['Report']).strip() + "\n")
        f.write("-" * 100 + "\n")

print(f"Successfully generated gold_58_clinical_audit.txt with {len(gold)} studies in UTF-8.")
