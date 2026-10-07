import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

gold = df[df['ACL'].notnull()].copy()
unlabeled = df[df['ACL'].isnull()].copy()

print(f"Total: {len(df)} | Gold: {len(gold)} | Unlabeled: {len(unlabeled)}")

print("\n--- SAMPLE GOLD REPORTS WITH LABELS ---")
for i in range(8):
    row = gold.iloc[i]
    pos = [c for c in target_cols if row[c] == 1.0]
    print(f"\n[Gold {i}] Positives: {pos}")
    print(row['Report'][:400])

print("\n--- SAMPLE UNLABELED REPORTS ---")
for i in range(6):
    row = unlabeled.iloc[i]
    print(f"\n[Unlabeled {i}] UID: {row['StudyInstanceUID'][:30]}...")
    print(row['Report'][:400])
