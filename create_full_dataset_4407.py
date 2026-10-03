import pandas as pd
import numpy as np

# Load train.csv, folds.csv, and report_extracted_labels_4349.csv
df_train = pd.read_csv('train.csv')
df_folds = pd.read_csv('folds.csv')
df_extracted = pd.read_csv('report_extracted_labels_4349.csv')

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# Identify the 58 gold studies and their folds
gold_mask = df_folds['ACL'].notnull()
gold_studies = df_folds[gold_mask].copy().reset_index(drop=True)
gold_uids = set(gold_studies['StudyInstanceUID'])

print(f"Total Gold Studies: {len(gold_studies)}")
print("Gold fold distribution:\n", gold_studies['fold'].value_counts().sort_index())

# For gold studies: is_gold = 1, weight = 3.0, labels are pure gold 0.0 / 1.0
gold_studies['is_gold'] = 1
gold_studies['sample_weight'] = 3.0

# For the 4,349 report-labeled studies:
# Extract their labels from report_extracted_labels_4349.csv
unlabeled_mask = ~df_train['StudyInstanceUID'].isin(gold_uids)
unlabeled_studies = df_train[unlabeled_mask].copy().reset_index(drop=True)

# Merge with extracted labels
unlabeled_merged = pd.merge(
    unlabeled_studies[['StudyInstanceUID', 'Report']], 
    df_extracted, 
    on='StudyInstanceUID', 
    how='inner'
)

unlabeled_merged['is_gold'] = 0
unlabeled_merged['sample_weight'] = 1.0

# Assign folds to unlabeled studies (0..4) balanced
np.random.seed(42)
unlabeled_merged['fold'] = np.random.randint(0, 5, size=len(unlabeled_merged))

print(f"Total Unlabeled Studies merged with extracted labels: {len(unlabeled_merged)}")

# Combine into master 4,407 table
master_df = pd.concat([gold_studies, unlabeled_merged], ignore_index=True)
print(f"Master Dataset Shape: {master_df.shape}")
print(f"is_gold counts:\n{master_df['is_gold'].value_counts()}")

out_path = r"D:\MLprojs\KaggleCompetition\RSNA\full_dataset_4407_folds.csv"
master_df.to_csv(out_path, index=False)
print(f"Saved master dataset to {out_path}!")
