import pandas as pd
import numpy as np

train = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\train.csv")
target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

labeled_df = train[train[target_cols].notnull().any(axis=1)].copy()
print(f"Total labeled studies: {len(labeled_df)}")
print("\nLabel distribution in labeled studies:")
for col in target_cols:
    pos = (labeled_df[col] == 1.0).sum()
    neg = (labeled_df[col] == 0.0).sum()
    print(f"  {col:18s}: Pos = {pos:2d} ({pos/len(labeled_df)*100:5.1f}%), Neg = {neg:2d} ({neg/len(labeled_df)*100:5.1f}%)")

print("\nCorrelations between targets in labeled studies:")
print(labeled_df[target_cols].astype(float).corr().round(2))
