import pandas as pd
import numpy as np

train = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\train.csv")

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# Separate labeled and unlabeled
labeled_mask = train[target_cols].notnull().any(axis=1)
labeled_df = train[labeled_mask].copy().reset_index(drop=True)
unlabeled_df = train[~labeled_mask].copy().reset_index(drop=True)

print(f"Labeled: {len(labeled_df)}, Unlabeled: {len(unlabeled_df)}")

# Iterative Stratification for Multi-Label Data (Sechidis et al., 2011)
def iterative_multilabel_split(y, n_splits=5, seed=42):
    np.random.seed(seed)
    n_samples, n_labels = y.shape
    
    # Desired per-fold counts
    target_counts = np.zeros((n_splits, n_labels))
    total_pos = y.sum(axis=0)
    for c in range(n_labels):
        target_counts[:, c] = total_pos[c] / n_splits
    
    folds = np.zeros(n_samples, dtype=int) - 1
    fold_counts = np.zeros((n_splits, n_labels))
    fold_sizes = np.zeros(n_splits)
    
    # Order samples by label rarity
    sample_scores = y @ (1.0 / (y.sum(axis=0) + 1e-5))
    order = np.argsort(-sample_scores)
    
    for idx in order:
        sample_y = y[idx]
        if sample_y.sum() == 0:
            # If no positive labels, place into smallest fold
            best_fold = np.argmin(fold_sizes)
        else:
            # Find fold with largest deficit for the sample's positive labels
            deficits = target_counts - fold_counts
            sample_deficits = deficits[:, sample_y == 1].sum(axis=1)
            # Tie breaker: smallest fold size
            sample_deficits = sample_deficits - 0.001 * fold_sizes
            best_fold = np.argmax(sample_deficits)
            
        folds[idx] = best_fold
        fold_counts[best_fold] += sample_y
        fold_sizes[best_fold] += 1
        
    return folds

y_labeled = labeled_df[target_cols].values.astype(int)
labeled_folds = iterative_multilabel_split(y_labeled, n_splits=5, seed=42)
labeled_df['fold'] = labeled_folds

print("\n--- Multi-label distribution across folds (Gold-Standard Labeled Set) ---")
for fold in range(5):
    f_df = labeled_df[labeled_df['fold'] == fold]
    print(f"\nFold {fold} (N={len(f_df)}):")
    for col in target_cols:
        pos = (f_df[col] == 1).sum()
        print(f"  {col:18s}: {pos}", end=" | ")
    print()

# Assign unlabeled studies evenly across the 5 folds
np.random.seed(42)
unlabeled_folds = np.random.permutation(len(unlabeled_df)) % 5
unlabeled_df['fold'] = unlabeled_folds

# Combine back into master folds.csv
all_folds = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
all_folds.to_csv(r"D:\MLprojs\KaggleCompetition\RSNA\folds.csv", index=False)
print(f"\nSuccessfully saved D:\\MLprojs\\KaggleCompetition\\RSNA\\folds.csv with {len(all_folds)} rows!")
