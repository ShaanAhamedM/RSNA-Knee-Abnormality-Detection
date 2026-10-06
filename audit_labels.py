import pandas as pd
import numpy as np

train = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\train.csv")
print("Train shape:", train.shape)
print("Columns:", train.columns.tolist())

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# Check how many rows have ANY non-null target
has_target = train[target_cols].notnull().any(axis=1)
print(f"Rows with at least one non-null target: {has_target.sum()} / {len(train)}")

# Check rows where ALL targets are null
null_all = train[target_cols].isnull().all(axis=1)
print(f"Rows where ALL targets are null: {null_all.sum()} / {len(train)}")

# Inspect the rows that HAVE targets
print("\n--- Rows with targets sample ---")
print(train[has_target][['StudyInstanceUID'] + target_cols].head(10))

# Inspect what the rows WITHOUT targets have in 'Report'
print("\n--- Rows without targets sample ---")
print(train[~has_target][['StudyInstanceUID', 'Report']].head(5))

# Also check test.csv and sample_submission.csv
sub = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\sample_submission.csv")
print("\nSample submission shape:", sub.shape)
print("Sample submission columns:", sub.columns.tolist())
print(sub.head())

test = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\test.csv")
print("\nTest shape:", test.shape)
print(test.head())
