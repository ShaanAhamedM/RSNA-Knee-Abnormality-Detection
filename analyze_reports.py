import pandas as pd
import re

train = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\folds.csv")
labeled = train[train['ACL'].notnull()].copy()

print(f"Auditing {len(labeled)} gold-standard labeled reports...")

# Inspect a few reports in detail
for i in range(5):
    print(f"\n--- Study {i} (UID: {labeled.iloc[i]['StudyInstanceUID']}) ---")
    pos_targets = [c for c in ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'] if labeled.iloc[i][c] == 1.0]
    print(f"Ground Truth Positives: {pos_targets}")
    print(f"Report Text:\n{labeled.iloc[i]['Report'][:400]}...")
