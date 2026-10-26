"""
Clinical NLP Teacher V8 (Contextual Semantic Projection + Hard-Negative Mining)
File: upgrade_nlp_teacher_v8.py
"""

import os
import sys
import shutil
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT_DIR = r"D:\MLprojs\KaggleCompetition\RSNA"
os.chdir(ROOT_DIR)

TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def main():
    print("=" * 80)
    print("STAGE 1: CLINICAL NLP TEACHER V8 (HARD-NEGATIVE MINING & ENRICHMENT)")
    print("=" * 80)

    train_df = pd.read_csv("train.csv")
    gold_df = train_df[train_df['ACL'].notna()].copy()
    gold_uids = set(gold_df['StudyInstanceUID'].values)

    v7_path = "report_extracted_labels_4349_commercial.csv"
    if not os.path.exists(v7_path):
        v7_path = "report_extracted_labels_4349_sharpened.csv"
    v7_df = pd.read_csv(v7_path).set_index("StudyInstanceUID")

    temp_kaggle_dir = "temp_kaggle"
    consensus_path = os.path.join(temp_kaggle_dir, "consensus_labels.csv")
    llm_blend_path = os.path.join(temp_kaggle_dir, "llm_labels_v4_blend.csv")

    has_consensus = os.path.exists(consensus_path)
    has_llm = os.path.exists(llm_blend_path)

    consensus_df = pd.read_csv(consensus_path).set_index("StudyInstanceUID") if has_consensus else None
    llm_df = pd.read_csv(llm_blend_path).set_index("StudyInstanceUID") if has_llm else None

    print(f"Loaded {len(train_df)} total studies ({len(gold_uids)} Gold, {len(train_df) - len(gold_uids)} Silver).")
    print(f"External Priors: Consensus={has_consensus}, LLM Blend={has_llm}")

    v8_records = []
    certainty_masks = {}

    hard_neg_count = {t: 0 for t in TARGET_COLS}
    hard_pos_count = {t: 0 for t in TARGET_COLS}
    uncertain_count = {t: 0 for t in TARGET_COLS}

    for idx, row in train_df.iterrows():
        uid = row['StudyInstanceUID']
        rec = {'StudyInstanceUID': uid}
        masks_for_study = {}

        for t in TARGET_COLS:
            # Baseline from V7
            val_v7 = float(v7_df.loc[uid, t]) if uid in v7_df.index and t in v7_df.columns else 0.25
            mask_col = f"{t}_mask"
            mask_v7 = float(v7_df.loc[uid, mask_col]) if uid in v7_df.index and mask_col in v7_df.columns else 0.0

            # Gather external priors
            ext_vals = []
            if consensus_df is not None and uid in consensus_df.index and t in consensus_df.columns:
                val = consensus_df.loc[uid, t]
                if pd.notna(val):
                    ext_vals.append(float(val))
            if llm_df is not None and uid in llm_df.index and t in llm_df.columns:
                val = llm_df.loc[uid, t]
                if pd.notna(val):
                    ext_vals.append(float(val))

            if len(ext_vals) > 0:
                ext_mean = float(np.mean(ext_vals))
                if mask_v7 == 1.0:
                    blended_val = 0.65 * val_v7 + 0.35 * ext_mean
                else:
                    blended_val = 0.35 * val_v7 + 0.65 * ext_mean
            else:
                blended_val = val_v7

            # Hard Negative Mining & Purity Gating logic
            if blended_val < 0.15:
                final_val = min(blended_val, 0.04)
                cert_mask = 1.0
                hard_neg_count[t] += 1
            elif blended_val > 0.82:
                final_val = max(blended_val, 0.96)
                cert_mask = 1.0
                hard_pos_count[t] += 1
            else:
                final_val = blended_val
                cert_mask = 0.35
                uncertain_count[t] += 1

            final_val = float(np.clip(final_val, 0.005, 0.995))
            rec[t] = final_val
            rec[f"{t}_mask"] = cert_mask
            masks_for_study[t] = cert_mask

        v8_records.append(rec)
        certainty_masks[uid] = masks_for_study

    v8_df = pd.DataFrame(v8_records)
    v8_out_path = "report_extracted_labels_4349_v8.csv"
    v8_df.to_csv(v8_out_path, index=False)
    print(f"[OK] Saved V8 Extracted Labels to {v8_out_path} ({len(v8_df)} rows).")

    masks_out_path = os.path.join("oof", "nlp_teacher_v8_masks.json")
    with open(masks_out_path, "w") as f:
        json.dump(certainty_masks, f)
    print(f"[OK] Saved V8 Certainty Masks to {masks_out_path}")

    # Evaluate on Gold-58 Ground Truth
    merged_eval = pd.merge(gold_df[['StudyInstanceUID'] + TARGET_COLS], v8_df, on='StudyInstanceUID', suffixes=('_true', '_pred'))
    aucs = []
    print("\n--- Evaluation of NLP Teacher V8 on Gold-58 Studies ---")
    for t in TARGET_COLS:
        y_true = merged_eval[t + '_true'].values
        y_pred = merged_eval[t + '_pred'].values
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            print(f"  {t:20s}: AUC = {auc:.4f} (Pos: {hard_pos_count[t]}, Neg: {hard_neg_count[t]}, Unc: {uncertain_count[t]})")
        else:
            aucs.append(0.5)

    macro_auc = float(np.mean(aucs))
    print(f"\n>>> NLP Teacher V8 Overall Macro-AUC on Gold-58: {macro_auc:.4f} (vs V7 0.6793) <<<")

    # Clean up temporary Kaggle folder immediately (ZERO DISK BLOAT RULE)
    if os.path.exists(temp_kaggle_dir):
        shutil.rmtree(temp_kaggle_dir)
        print(f"[OK] Deleted temporary directory '{temp_kaggle_dir}' (Zero disk footprint preserved).")

    return macro_auc

if __name__ == "__main__":
    main()
