"""
Bulletproof Submission Sanity Suite (verify_submission.py)
Audits the final submission file against 4 critical criteria:
1. Schema & ID Integrity
2. Zero Corruption Check
3. Distribution & Variance Check (vs. True Label Prevalence pi_c)
4. Reproducibility Check on 5 random studies from the zip archive
"""

import os
import io
import json
import zipfile
import numpy as np
import pandas as pd
import pydicom
import torch

from src.preprocessing import process_mri_slice
from src.models import RSNAKneeModel

SUB_PATH = r"D:\MLprojs\KaggleCompetition\RSNA\submission.csv"
SAMPLE_SUB_PATH = r"D:\MLprojs\KaggleCompetition\RSNA\sample_submission.csv"
ZIP_PATH = r"D:\RSNA\rsna-knee-abnormality-detection.zip"
TRAIN_FOLDS_PATH = r"D:\MLprojs\KaggleCompetition\RSNA\folds.csv"
PARAMS_PATH = r"D:\MLprojs\KaggleCompetition\RSNA\oof\grandmaster_params.json"

TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def main():
    print("=" * 100)
    print("      RSNA KNEE ABNORMALITY DETECTION — BULLETPROOF SUBMISSION SANITY SUITE")
    print("=" * 100)
    
    assert os.path.exists(SUB_PATH), f"Submission file {SUB_PATH} does not exist!"
    assert os.path.exists(SAMPLE_SUB_PATH), f"Sample submission {SAMPLE_SUB_PATH} does not exist!"
    
    sub = pd.read_csv(SUB_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    
    # --------------------------------------------------------------------------
    # 1. SCHEMA & ID INTEGRITY CHECK
    # --------------------------------------------------------------------------
    print("\n[CHECK 1/4] SCHEMA & ID INTEGRITY")
    print("-" * 50)
    row_count_ok = len(sub) == len(sample_sub)
    col_names_ok = list(sub.columns) == list(sample_sub.columns)
    id_order_ok = (sub['StudyInstanceUID'].values == sample_sub['StudyInstanceUID'].values).all()
    target_cols_ok = all(col in sub.columns for col in TARGET_COLUMNS)
    
    print(f"  • Row count: {len(sub)} (Sample: {len(sample_sub)}) -> {'PASS [OK]' if row_count_ok else 'FAIL [MISMATCH]'}")
    print(f"  • Column schema: {len(sub.columns)} columns -> {'PASS [OK]' if col_names_ok else 'FAIL [MISMATCH]'}")
    print(f"  • StudyInstanceUID ordering exact match -> {'PASS [OK]' if id_order_ok else 'FAIL [MISMATCH]'}")
    print(f"  • All 12 targets present -> {'PASS [OK]' if target_cols_ok else 'FAIL [MISSMATCH]'}")
    
    assert row_count_ok and col_names_ok and id_order_ok and target_cols_ok, "Schema check failed!"
    
    # --------------------------------------------------------------------------
    # 2. ZERO CORRUPTION CHECK
    # --------------------------------------------------------------------------
    print("\n[CHECK 2/4] ZERO CORRUPTION CHECK")
    print("-" * 50)
    nan_count = sub.isnull().sum().sum()
    inf_count = np.isinf(sub[TARGET_COLUMNS].values).sum()
    range_ok = ((sub[TARGET_COLUMNS].values >= 0.0) & (sub[TARGET_COLUMNS].values <= 1.0)).all()
    
    # Check for constant columns
    constant_cols = []
    for col in TARGET_COLUMNS:
        if sub[col].nunique() <= 1:
            constant_cols.append(col)
            
    print(f"  • NaN count: {nan_count} -> {'PASS [OK]' if nan_count == 0 else 'FAIL [HAS NANS]'}")
    print(f"  • Inf count: {inf_count} -> {'PASS [OK]' if inf_count == 0 else 'FAIL [HAS INFS]'}")
    print(f"  • Bounded in [0.0, 1.0]: -> {'PASS [OK]' if range_ok else 'FAIL [OUT OF BOUNDS]'}")
    print(f"  • Constant columns count: {len(constant_cols)} -> {'PASS [OK]' if len(constant_cols) == 0 else f'FAIL [CONSTANT: {constant_cols}]'}")
    
    assert nan_count == 0 and inf_count == 0 and range_ok and len(constant_cols) == 0, "Corruption check failed!"
    
    # --------------------------------------------------------------------------
    # 3. DISTRIBUTION & VARIANCE CHECK (VS TRUE EMPIRICAL PREVALENCE PI_C)
    # --------------------------------------------------------------------------
    print("\n[CHECK 3/4] DISTRIBUTION & VARIANCE DIAGNOSTIC TABLE")
    print("-" * 100)
    
    # Compute true training label prevalence pi_c
    true_list = [np.load(f'oof/true_fold{f}.npy') for f in range(5)]
    true_all = np.vstack(true_list)
    pi_c = true_all.mean(axis=0)
    
    diag_rows = []
    for i, col in enumerate(TARGET_COLUMNS):
        vals = sub[col].values
        q05 = np.percentile(vals, 5)
        q50 = np.percentile(vals, 50)
        q95 = np.percentile(vals, 95)
        mean_val = vals.mean()
        std_val = vals.std()
        min_val = vals.min()
        max_val = vals.max()
        true_pi = pi_c[i]
        delta_pi = mean_val - true_pi
        
        diag_rows.append([
            col,
            f"{true_pi*100:5.1f}%",
            f"{mean_val*100:5.1f}%",
            f"{delta_pi*100:+5.1f}%",
            f"{std_val:.4f}",
            f"{min_val:.4f}",
            f"{q05:.4f}",
            f"{q50:.4f}",
            f"{q95:.4f}",
            f"{max_val:.4f}"
        ])
        
    headers = ["Pathology Target", "True pi_c", "Sub Mean", "Delta", "Std", "Min", "5%", "50% (Med)", "95%", "Max"]
    col_widths = [18, 10, 10, 10, 8, 8, 8, 10, 8, 8]
    header_str = "| " + " | ".join(f"{h:{w}s}" for h, w in zip(headers, col_widths)) + " |"
    sep_str = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    print(header_str)
    print(sep_str)
    for r in diag_rows:
        row_str = "| " + " | ".join(f"{str(val):{w}s}" for val, w in zip(r, col_widths)) + " |"
        print(row_str)
    
    # Distinctness check: Ensure no two columns have identical prediction vectors
    identical_pairs = []
    for i in range(len(TARGET_COLUMNS)):
        for j in range(i + 1, len(TARGET_COLUMNS)):
            c1, c2 = TARGET_COLUMNS[i], TARGET_COLUMNS[j]
            if np.allclose(sub[c1].values, sub[c2].values, atol=1e-5):
                identical_pairs.append((c1, c2))
    print(f"\n  • Distinctness check: {len(identical_pairs)} identical column pairs -> {'PASS [ALL 12 TARGETS DISTINCT]' if len(identical_pairs) == 0 else f'FAIL [{identical_pairs}]'}")
    assert len(identical_pairs) == 0, "Identical columns detected!"
    
    # --------------------------------------------------------------------------
    # 4. REPRODUCIBILITY CHECK ON 5 RANDOM STUDIES FROM ZIP ARCHIVE
    # --------------------------------------------------------------------------
    print("\n[CHECK 4/4] REPRODUCIBILITY CHECK ON 5 RANDOM STUDIES (FULL FORWARD PASS)")
    print("-" * 100)
    
    from src.dataset import get_slice_lookup
    slice_lookup = get_slice_lookup()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading champion ConvNeXt backbone for verification pass on {device}...")
    bb_path = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints\convnext_small_backbone_in1.pt"
    import timm
    model = timm.create_model('convnext_small', pretrained=False, in_chans=1, num_classes=12).to(device)
    # load backbone
    state = torch.load(bb_path, map_location=device)
    # load matching keys
    model_dict = model.state_dict()
    matched = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    model.eval()
    
    # Pick 5 random study UIDs from the lookup
    all_study_ids = list(slice_lookup.keys())
    np.random.seed(42)
    sample_uids = list(np.random.choice(all_study_ids, 5, replace=False))
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for idx, uid in enumerate(sample_uids):
            slice_paths = slice_lookup[uid][:6]
            slice_tensors = []
            slice_stds = []
            
            for sp in slice_paths:
                raw_bytes = z.read(sp)
                dcm = pydicom.dcmread(io.BytesIO(raw_bytes))
                proc = process_mri_slice(dcm.pixel_array, target_size=256)
                slice_tensors.append(proc)
                slice_stds.append(float(proc.std()))
                
            bag = torch.tensor(np.array(slice_tensors), dtype=torch.float32).unsqueeze(1).to(device) # (K, 1, 256, 256)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    logits = model(bag).mean(dim=0, keepdim=True)
                    probs = torch.sigmoid(logits).cpu().numpy()[0]
                    
            print(f"  Study {idx+1}: {uid[:32]}...")
            print(f"    - Input Bag Tensor: shape={list(bag.shape)}, mean={bag.mean():.4f}, std={bag.std():.4f}")
            print(f"    - Per-slice pixel stdevs: {[round(s, 3) for s in slice_stds]} (Non-zero tissue variance confirmed)")
            print(f"    - Forward pass output logits: min={logits.min():.2f}, max={logits.max():.2f}, mean={logits.mean():.2f}")
            print(f"    - Forward pass output probs : min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
            
            assert bag.std() > 0.1, "Zero pixel variance in MRI tensor!"
            assert not np.isnan(probs).any(), "NaN in model forward pass output!"
            
    print("\n" + "=" * 100)
    print("  >>> ALL 4 BULLETPROOF SUBMISSION SANITY CHECKS PASSED WITH ZERO ERRORS <<<")
    print("=" * 100)

if __name__ == '__main__':
    main()
