"""
Production Engine for Upgrade 2 & Upgrade 3:
1. Extracts 12-18 distinct slice tokens per study (4-6 Sagittal, 4-6 Coronal, 4-6 Axial).
2. Preserves slice-level localized tokens with plane embeddings & depth positional encodings.
3. Computes 27 Plane-Specific Radiomic features (fluid fraction, asymmetry, percentiles, gradient energy).
Saves:
- D:\MLprojs\KaggleCompetition\RSNA\dicom_and_radiomics_features.csv (45-column tabular matrix)
- D:\MLprojs\KaggleCompetition\RSNA\multislice_tokens_convnext.pt (Slice-level token database)
"""

import io
import time
import json
import zipfile
import pydicom
import numpy as np
import pandas as pd
import torch
import timm
from src.preprocessing import process_mri_slice

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Extraction Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# Offline ConvNeXt-Small backbone
bb_path = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints\convnext_small_backbone_in1.pt"
backbone = timm.create_model('convnext_small', pretrained=False, in_chans=1, num_classes=0)
backbone.load_state_dict(torch.load(bb_path, map_location=device))
backbone.to(device)
backbone.eval()

with open('study_plane_catalog.json') as f:
    cat = json.load(f)

dicom_base_df = pd.read_csv('dicom_tabular_features.csv')
dicom_dict = dicom_base_df.set_index('StudyInstanceUID').to_dict('index')

print(f"Total studies to process: {len(cat)}")
zip_path = r"D:\RSNA\rsna-knee-abnormality-detection.zip"

all_token_db = {}
radiomics_rows = []
study_uids = list(cat.keys())

def extract_radiomics(img: np.ndarray) -> dict:
    p50, p75, p90, p95, p99 = np.percentile(img, [50, 75, 90, 95, 99])
    bright_fluid_ratio = float(np.mean(img > 0.85))
    mid_w = img.shape[1] // 2
    left_half = img[:, :mid_w]
    right_half = img[:, mid_w:]
    lr_mean_diff = float(np.mean(left_half) - np.mean(right_half))
    lr_p95_diff = float(np.percentile(left_half, 95) - np.percentile(right_half, 95))
    dx = np.diff(img, axis=1)
    dy = np.diff(img, axis=0)
    grad_energy = float(np.mean(dx**2) + np.mean(dy**2))
    return {
        'p50': float(p50), 'p75': float(p75), 'p90': float(p90), 'p95': float(p95), 'p99': float(p99),
        'bright_fluid_ratio': bright_fluid_ratio,
        'lr_mean_diff': lr_mean_diff,
        'lr_p95_diff': lr_p95_diff,
        'grad_energy': grad_energy
    }

t_start = time.time()
with zipfile.ZipFile(zip_path, 'r') as z:
    for idx, uid in enumerate(study_uids):
        entry = cat[uid]
        study_slice_imgs = []
        study_planes = []
        study_depths = []
        
        rad_entry = {'StudyInstanceUID': uid}
        
        for pl_idx, pl in enumerate(['Sagittal', 'Coronal', 'Axial']):
            slices = entry.get(pl, [])
            if not slices:
                for k in ['p50', 'p75', 'p90', 'p95', 'p99', 'bright_fluid_ratio', 'lr_mean_diff', 'lr_p95_diff', 'grad_energy']:
                    rad_entry[f"{pl}_{k}"] = 0.0
                continue
                
            n_sl = len(slices)
            indices = np.linspace(0, n_sl - 1, min(4, n_sl), dtype=int)
            
            # Radiomics on middle slice of this plane
            mid_sp = slices[n_sl // 2]
            try:
                raw_mid = z.read(mid_sp)
                dcm_mid = pydicom.dcmread(io.BytesIO(raw_mid))
                p_mid = process_mri_slice(dcm_mid.pixel_array, target_size=256, use_clahe=False, crop_roi=False)
                rad_vals = extract_radiomics(p_mid)
                for k, v in rad_vals.items():
                    rad_entry[f"{pl}_{k}"] = v
            except Exception:
                for k in ['p50', 'p75', 'p90', 'p95', 'p99', 'bright_fluid_ratio', 'lr_mean_diff', 'lr_p95_diff', 'grad_energy']:
                    rad_entry[f"{pl}_{k}"] = 0.0
                    
            for s_k in indices:
                sp = slices[s_k]
                try:
                    raw = z.read(sp)
                    dcm = pydicom.dcmread(io.BytesIO(raw))
                    p = process_mri_slice(dcm.pixel_array, target_size=256, use_clahe=False, crop_roi=False)
                    study_slice_imgs.append(p)
                    study_planes.append(pl_idx)
                    study_depths.append(s_k / max(1, n_sl - 1))
                except Exception:
                    pass
                    
        # Extract features through ConvNeXt
        if study_slice_imgs:
            t_batch = torch.tensor(np.array(study_slice_imgs), dtype=torch.float32, device=device).unsqueeze(1)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    feats = backbone(t_batch) # (K, 768)
            tokens_fp16 = feats.cpu().to(torch.float16)
            planes_t = torch.tensor(study_planes, dtype=torch.long)
            depths_t = torch.tensor(study_depths, dtype=torch.float32)
        else:
            tokens_fp16 = torch.zeros((1, 768), dtype=torch.float16)
            planes_t = torch.zeros(1, dtype=torch.long)
            depths_t = torch.zeros(1, dtype=torch.float32)
            
        all_token_db[uid] = {
            'tokens': tokens_fp16,
            'planes': planes_t,
            'depths': depths_t
        }
        
        radiomics_rows.append(rad_entry)
        
        if (idx + 1) % 200 == 0 or (idx + 1) == len(study_uids):
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            rem = (len(study_uids) - (idx + 1)) / max(rate, 1e-5)
            print(f"Extraction Progress: {idx + 1:4d} / {len(study_uids)} ({rate:.1f} studies/sec | Elapsed: {elapsed/60:.1f}m | Remaining: {rem/60:.1f}m)")

# Save token database
out_tokens_path = r"D:\MLprojs\KaggleCompetition\RSNA\multislice_tokens_convnext.pt"
torch.save(all_token_db, out_tokens_path)
print(f"\n[SUCCESS] Saved multi-slice tokens for {len(all_token_db)} studies to {out_tokens_path}!")

# Merge with DICOM metadata and save
rad_df = pd.DataFrame(radiomics_rows)
full_tab_df = pd.merge(dicom_base_df, rad_df, on='StudyInstanceUID', how='left')
out_tab_path = r"D:\MLprojs\KaggleCompetition\RSNA\dicom_and_radiomics_features.csv"
full_tab_df.to_csv(out_tab_path, index=False)
print(f"[SUCCESS] Saved combined DICOM + Radiomics table ({full_tab_df.shape[1]} columns) to {out_tab_path}!")
